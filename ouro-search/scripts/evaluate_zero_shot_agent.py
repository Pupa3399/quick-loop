from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_MIRROR = "https://hf-mirror.com"
MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"
MODEL_REVISION = "f1edd81e7ac41355db670500ceaf204e0f73af68"

import httpx  # noqa: E402
import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from ouro_search.agent import AgentRunner  # noqa: E402
from ouro_search.evaluation import aggregate_records, evaluate_trajectory  # noqa: E402
from ouro_search.inference import VllmEngine  # noqa: E402
from ouro_search.search import HttpSearchClient  # noqa: E402
from ouro_search.trajectory import JsonlTrajectoryWriter  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _select_balanced(rows: list[dict[str, Any]], per_dataset: int) -> list[dict[str, Any]]:
    if per_dataset < 1:
        raise ValueError("samples_per_dataset must be positive")
    selected: list[dict[str, Any]] = []
    for dataset in ("nq", "hotpotqa"):
        candidates = [row for row in rows if row["dataset"] == dataset]
        if len(candidates) < per_dataset:
            raise ValueError(
                f"Manifest has {len(candidates)} {dataset} rows, needs {per_dataset}"
            )
        selected.extend(candidates[:per_dataset])
    return selected


def _validate_model(config: DictConfig) -> dict[str, Any]:
    if config.model.model_name != MODEL_ID or config.model.revision != MODEL_REVISION:
        raise RuntimeError("Evaluation must use the pinned original Ouro revision")
    if config.model.backend != "vllm" or int(config.model.fixed_loop_steps) != 3:
        raise RuntimeError("Evaluation requires the inference-only vLLM Ouro R3 view")
    model_path = Path(config.model.model_path)
    weight_path = model_path / "model.safetensors"
    assets_path = model_path / "assets"
    if "checkpoint" in str(model_path).lower() or not weight_path.is_symlink():
        raise RuntimeError("Model path is not the immutable original Ouro metadata view")
    snapshot_dir = assets_path.resolve().parent
    if snapshot_dir.name != MODEL_REVISION or not weight_path.samefile(
        snapshot_dir / "model.safetensors"
    ):
        raise RuntimeError("Ouro weight symlink does not resolve to the pinned HF snapshot")
    stat = weight_path.stat()
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_path": str(model_path),
        "resolved_snapshot": str(snapshot_dir),
        "weight_size_bytes": stat.st_size,
        "weight_mtime_ns_before": stat.st_mtime_ns,
        "loop_steps": 3,
        "inference_only": True,
        "optimizer_created": False,
    }


def _validate_parameters(config: DictConfig) -> None:
    expected = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_search_turns": 4,
        "top_k": 3,
    }
    actual = {
        "temperature": float(config.agent.temperature),
        "top_p": float(config.agent.top_p),
        "max_search_turns": int(config.agent.max_search_turns),
        "top_k": int(config.search.top_k),
    }
    if actual != expected:
        raise RuntimeError(f"Evaluation parameter mismatch: expected={expected}, actual={actual}")


def _retriever_health(config: DictConfig) -> dict[str, Any]:
    response = httpx.get(
        f"{str(config.search.retriever_url).rstrip('/')}/health",
        timeout=float(config.search.timeout_seconds),
    )
    response.raise_for_status()
    health = response.json()
    if (
        health.get("status") != "ok"
        or not health.get("faiss_gpu")
        or health.get("index_size") != 21_015_324
    ):
        raise RuntimeError(f"Real GPU Wiki18 Retriever is not healthy: {health}")
    return health


def _case_summary(record: dict[str, Any]) -> dict[str, Any]:
    trajectory = record["trajectory"]
    return {
        "id": record["id"],
        "dataset": record["dataset"],
        "failure_class": record["failure_class"],
        "question": record["question"],
        "reference_answer": record["reference_answer"],
        "prediction": record["prediction"],
        "raw_generations": trajectory["raw_generations"],
        "action_audits": record["action_audits"],
        "search_queries": [turn["query"] for turn in trajectory["turns"]],
        "retrieved_titles": [
            [document["title"] for document in turn["retrieved_documents"]]
            for turn in trajectory["turns"]
        ],
        "termination_reason": record["termination_reason"],
    }


def _summarize(
    records: list[dict[str, Any]],
    *,
    failure_cases_per_class: int,
) -> dict[str, Any]:
    by_dataset = {
        dataset: aggregate_records(
            [record for record in records if record["dataset"] == dataset]
        )
        for dataset in ("nq", "hotpotqa")
    }
    failure_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    format_failures: list[dict[str, Any]] = []
    for record in records:
        failure_class = str(record["failure_class"])
        if (
            failure_class != "success"
            and len(failure_cases[failure_class]) < failure_cases_per_class
        ):
            failure_cases[failure_class].append(_case_summary(record))
        if record["malformed_action"] and len(format_failures) < failure_cases_per_class:
            format_failures.append(_case_summary(record))
    return {
        "overall": aggregate_records(records),
        "by_dataset": by_dataset,
        "failure_cases": dict(failure_cases),
        "format_failure_cases": format_failures,
    }


@hydra.main(version_base=None, config_path="../configs", config_name="zero_shot_agent_eval")
def main(config: DictConfig) -> None:
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR:
        raise RuntimeError(f"Set HF_ENDPOINT={HF_MIRROR}; overseas fallback is disabled")
    _validate_parameters(config)
    model_identity = _validate_model(config)
    retriever_health = _retriever_health(config)

    manifest_path = PROJECT_ROOT / config.evaluation.manifest
    selected = _select_balanced(
        _load_jsonl(manifest_path), int(config.evaluation.samples_per_dataset)
    )
    profiles = [str(profile) for profile in config.evaluation.profiles]
    if profiles != ["search_r1", "hermes"]:
        raise RuntimeError("This comparison requires profiles=[search_r1, hermes] in that order")
    output_dir = PROJECT_ROOT / config.evaluation.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to mix evaluation runs in non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selected_samples.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    run_config = {
        "model": model_identity,
        "retriever_health": retriever_health,
        "generation": {
            "temperature": float(config.agent.temperature),
            "top_p": float(config.agent.top_p),
            "max_new_tokens_per_turn": int(config.agent.max_new_tokens),
            "max_response_tokens": int(config.agent.max_response_tokens),
            "max_search_turns": int(config.agent.max_search_turns),
            "top_k": int(config.search.top_k),
        },
        "profiles": profiles,
        "sample_ids": [row["id"] for row in selected],
        "hydra_config": OmegaConf.to_container(config, resolve=True),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    engine = VllmEngine(
        config.model.model_path,
        dtype=config.model.dtype,
        trust_remote_code=config.model.trust_remote_code,
        max_model_len=config.model.max_model_len,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
        tensor_parallel_size=config.model.tensor_parallel_size,
        enforce_eager=config.model.enforce_eager,
        use_chat_template=config.model.use_chat_template,
        fixed_loop_steps=config.model.fixed_loop_steps,
    )
    all_summaries: dict[str, Any] = {}
    started = time.perf_counter()
    for profile_name in profiles:
        profile_started = time.perf_counter()
        trajectory_dir = output_dir / "trajectories" / profile_name
        records_path = output_dir / f"{profile_name}_records.jsonl"
        runner = AgentRunner(
            engine,
            HttpSearchClient(
                config.search.retriever_url,
                top_k=config.search.top_k,
                timeout_seconds=config.search.timeout_seconds,
            ),
            max_search_turns=config.agent.max_search_turns,
            default_loop_steps=3,
            max_new_tokens=config.agent.max_new_tokens,
            max_information_tokens=config.agent.max_information_tokens,
            max_response_tokens=config.agent.max_response_tokens,
            temperature=config.agent.temperature,
            top_p=config.agent.top_p,
            trajectory_writer=JsonlTrajectoryWriter(trajectory_dir),
            prompt_profile=profile_name,
        )
        records: list[dict[str, Any]] = []
        with records_path.open("w", encoding="utf-8") as records_file:
            for index, example in enumerate(selected, start=1):
                trajectory = runner.run(
                    example["question"],
                    sample_id=example["id"],
                    reference_answer=example["reference_answer"],
                    answer_aliases=example["answer_aliases"],
                )
                record = evaluate_trajectory(
                    trajectory,
                    dataset=example["dataset"],
                    profile_name=profile_name,
                )
                records.append(record)
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                records_file.flush()
                print(
                    f"profile={profile_name} sample={index}/{len(selected)} "
                    f"id={example['id']} searches={trajectory.num_search_turns} "
                    f"termination={trajectory.termination_reason} em={trajectory.reward}",
                    flush=True,
                )
        summary = _summarize(
            records,
            failure_cases_per_class=int(config.evaluation.failure_cases_per_class),
        )
        summary["elapsed_seconds"] = time.perf_counter() - profile_started
        all_summaries[profile_name] = summary
        (output_dir / f"{profile_name}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    weight_stat_after = (Path(config.model.model_path) / "model.safetensors").stat()
    model_identity["weight_mtime_ns_after"] = weight_stat_after.st_mtime_ns
    model_identity["weight_file_unchanged"] = (
        model_identity["weight_size_bytes"] == weight_stat_after.st_size
        and model_identity["weight_mtime_ns_before"] == weight_stat_after.st_mtime_ns
    )
    report = {
        "experiment": "original-ouro-r3-zero-shot-agent-protocol-comparison",
        "samples": len(selected),
        "model": model_identity,
        "retriever": retriever_health,
        "summaries": all_summaries,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
