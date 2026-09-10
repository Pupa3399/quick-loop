from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_MIRROR = "https://hf-mirror.com"

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from ouro_search.agent import AgentRunner  # noqa: E402
from ouro_search.inference import VllmEngine  # noqa: E402
from ouro_search.search import HttpSearchClient  # noqa: E402
from ouro_search.trajectory import JsonlTrajectoryWriter  # noqa: E402


def _aliases(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("answer_aliases", [])
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence):
        return [str(value) for value in raw]
    return []


def _iter_examples(config: DictConfig) -> Iterator[dict[str, Any]]:
    if not config.eval.data_path:
        yield {
            "id": config.eval.sample_id,
            "question": config.eval.question,
            "reference_answer": config.eval.reference_answer,
            "answer_aliases": list(config.eval.answer_aliases),
        }
        return

    from datasets import load_dataset

    path = Path(config.eval.data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation data does not exist: {path}")
    extension = "json" if path.suffix in {".json", ".jsonl"} else "parquet"
    dataset = load_dataset(extension, data_files=str(path), split="train")
    start = int(config.eval.start_index)
    stop = min(len(dataset), start + int(config.eval.max_samples))
    for index in range(start, stop):
        row = dataset[index]
        aliases = _aliases(row)
        reference = str(row.get("reference_answer", aliases[0] if aliases else ""))
        yield {
            "id": str(row.get("id", f"eval-{index}")),
            "question": str(row["question"]),
            "reference_answer": reference,
            "answer_aliases": aliases or ([reference] if reference else []),
        }


def _make_engine(config: DictConfig) -> VllmEngine:
    if config.model.backend != "vllm":
        raise ValueError(f"Unsupported evaluation backend: {config.model.backend}")
    model_path = Path(config.model.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    return VllmEngine(
        model_path,
        dtype=config.model.dtype,
        trust_remote_code=config.model.trust_remote_code,
        max_model_len=config.model.max_model_len,
        gpu_memory_utilization=config.model.gpu_memory_utilization,
        tensor_parallel_size=config.model.tensor_parallel_size,
        enforce_eager=config.model.enforce_eager,
        use_chat_template=config.model.use_chat_template,
        fixed_loop_steps=config.model.fixed_loop_steps,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(config: DictConfig) -> None:
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR:
        raise RuntimeError(f"Set HF_ENDPOINT={HF_MIRROR}; overseas fallback is disabled")
    engine = _make_engine(config)
    loop_steps = config.model.fixed_loop_steps or config.agent.default_loop_steps
    smoke_started = time.perf_counter()
    smoke_output = engine.generate(
        config.eval.plain_smoke_prompt,
        max_new_tokens=config.eval.plain_smoke_max_tokens,
        temperature=0.0,
        top_p=1.0,
        loop_steps=loop_steps,
    )
    smoke_elapsed = time.perf_counter() - smoke_started
    if not smoke_output:
        raise RuntimeError("Plain vLLM generation produced an empty response")
    runner = AgentRunner(
        engine,
        HttpSearchClient(
            config.search.retriever_url,
            top_k=config.search.top_k,
            timeout_seconds=config.search.timeout_seconds,
        ),
        max_search_turns=config.agent.max_search_turns,
        default_loop_steps=loop_steps,
        max_new_tokens=config.agent.max_new_tokens,
        max_information_tokens=config.agent.max_information_tokens,
        max_response_tokens=config.agent.max_response_tokens,
        temperature=config.agent.temperature,
        top_p=config.agent.top_p,
        trajectory_writer=JsonlTrajectoryWriter(PROJECT_ROOT / config.eval.trajectory_dir),
    )
    sample_reports = []
    for example in _iter_examples(config):
        started = time.perf_counter()
        trajectory = runner.run(
            example["question"],
            sample_id=example["id"],
            reference_answer=example["reference_answer"],
            answer_aliases=example["answer_aliases"],
        )
        elapsed = time.perf_counter() - started
        sample_reports.append(
            {
                "id": trajectory.id,
                "reward": trajectory.reward,
                "search_actions": trajectory.num_search_turns,
                "format_valid": int(trajectory.termination_reason == "answer"),
                "response_token_count": trajectory.response_token_count,
                "elapsed_seconds": elapsed,
                "termination_reason": trajectory.termination_reason,
            }
        )
    if not sample_reports:
        raise RuntimeError("Evaluation selection is empty")
    count = len(sample_reports)
    report = {
        "model": config.model.display_name,
        "model_id": config.model.model_name,
        "revision": config.model.revision,
        "parameter_count": config.model.parameter_count,
        "dtype": config.model.dtype,
        "kv_cache_exercised": True,
        "plain_smoke": {
            "elapsed_seconds": smoke_elapsed,
            "output": smoke_output,
        },
        "metrics": {
            "samples": count,
            "em": sum(item["reward"] for item in sample_reports) / count,
            "search_action_rate": sum(item["search_actions"] > 0 for item in sample_reports)
            / count,
            "format_valid_rate": sum(item["format_valid"] for item in sample_reports) / count,
            "mean_search_turns": sum(item["search_actions"] for item in sample_reports) / count,
            "mean_response_tokens": sum(
                item["response_token_count"] for item in sample_reports
            )
            / count,
            "mean_elapsed_seconds": sum(item["elapsed_seconds"] for item in sample_reports)
            / count,
        },
        "max_search_turns": config.agent.max_search_turns,
        "top_k": config.search.top_k,
        "sample_reports": sample_reports,
    }
    safe_model_name = "".join(
        character.lower() if character.isalnum() else "-"
        for character in config.model.display_name
    ).strip("-")
    report_dir = PROJECT_ROOT / config.eval.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{safe_model_name}-{config.eval.sample_id}.json"
    report["report_path"] = str(report_path)
    report["trajectory_dir"] = str(PROJECT_ROOT / config.eval.trajectory_dir)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
