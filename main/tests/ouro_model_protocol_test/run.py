from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")

# All model and dataset access in this experiment is local-only. If a missing file
# ever triggers a Hugging Face request, it must use the domestic mirror.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "main/models/huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "main/datasets/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DATASETS_OFFLINE", "1")
os.environ.setdefault("VLLM_PLUGINS", "ouro_search")
os.environ.setdefault("VLLM_USE_V1", "1")

from ouro_search.agent.profiles import get_prompt_profile  # noqa: E402
from ouro_search.agent.protocol_audit import audit_generation  # noqa: E402
from ouro_search.agent.runner import AgentRunner  # noqa: E402
from ouro_search.inference.vllm_engine import VllmEngine  # noqa: E402
from ouro_search.rewards.answer_reward import answer_exact_match  # noqa: E402
from ouro_search.search.client import HttpSearchClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Ouro 模型与 Agent 协议横向测试")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", choices=("base", "thinking"), required=True)
    parser.add_argument(
        "--protocol",
        choices=("search_r1", "hermes", "all"),
        default="all",
    )
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_samples(path: Path) -> list[dict[str, Any]]:
    samples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"样本文件必须是非空 JSON 数组: {path}")
    sample_ids = [str(sample["sample_id"]) for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("样本 ID 存在重复")
    return samples


def validate_local_model(model_config: DictConfig) -> Path:
    model_path = project_path(str(model_config.path)).resolve()
    source_snapshot = (model_path / "source_snapshot").resolve()
    expected_revision = str(model_config.source_revision)
    if not model_path.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {model_path}")
    if source_snapshot.name != expected_revision:
        raise RuntimeError(
            f"模型 revision 不匹配: expected={expected_revision}, actual={source_snapshot.name}"
        )
    required = ("config.json", "model.safetensors", "tokenizer.json")
    missing = [name for name in required if not (model_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"模型文件不完整: {missing}")
    return model_path


def validate_retriever(config: DictConfig) -> dict[str, Any]:
    url = str(config.url).rstrip("/")
    response = httpx.get(f"{url}/health", timeout=float(config.timeout_seconds))
    response.raise_for_status()
    health = response.json()
    expected_size = int(config.expected_index_size)
    if int(health.get("index_size", -1)) != expected_size:
        raise RuntimeError(
            f"Retriever index_size 不匹配: expected={expected_size}, "
            f"actual={health.get('index_size')}"
        )
    if bool(config.require_gpu_faiss) and not health.get("faiss_gpu"):
        raise RuntimeError("实验要求 GPU FAISS，但 /health 报告 faiss_gpu=false")
    return health


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def readable_trajectory(
    trajectory: Any,
    *,
    sample: dict[str, Any],
    model_name: str,
    protocol: str,
) -> dict[str, Any]:
    profile = get_prompt_profile(protocol)
    final_answer = profile.parse_final_answer(trajectory.final_model_output) or ""
    aliases = [str(value) for value in sample["answer_aliases"]]
    return {
        "sample_id": str(sample["sample_id"]),
        "model": model_name,
        "protocol": protocol,
        "question": str(sample["question"]),
        "reference_answer": str(sample["reference_answer"]),
        "prompt": trajectory.initial_rendered_prompt,
        "turns": [
            {
                "turn": generation.turn_id,
                "model_output": generation.raw_generation,
                "search_query": generation.parsed_query or "",
                "information": generation.observation_text,
            }
            for generation in trajectory.generations
        ],
        "final_answer": final_answer,
        "em": int(bool(final_answer) and answer_exact_match(final_answer, aliases)),
    }


def ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "count": numerator,
        "total": denominator,
        "rate": round(numerator / denominator, 6) if denominator else None,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(rows)
    format_valid_samples = 0
    search_samples = 0
    retriever_samples = 0
    multi_search_samples = 0
    final_answer_samples = 0
    exact_match_samples = 0
    search_turns = 0
    observation_events = 0
    continued_after_observation = 0
    valid_after_observation = 0
    malformed_generations = 0

    for row in rows:
        audits = [
            audit_generation(str(row["protocol"]), str(turn["model_output"]))
            for turn in row["turns"]
        ]
        format_valid_samples += int(bool(audits) and all(audit.format_valid for audit in audits))
        search_samples += int(
            any(audit.kind == "search" and audit.valid_action for audit in audits)
        )
        malformed_generations += sum(int(audit.malformed) for audit in audits)
        invocation_indexes = [
            index for index, turn in enumerate(row["turns"]) if turn["information"]
        ]
        invocation_count = len(invocation_indexes)
        search_turns += invocation_count
        retriever_samples += int(invocation_count > 0)
        multi_search_samples += int(invocation_count > 1)
        observation_events += invocation_count
        for index in invocation_indexes:
            has_next = index + 1 < len(row["turns"])
            continued_after_observation += int(has_next)
            valid_after_observation += int(has_next and audits[index + 1].valid_action)
        final_answer_samples += int(bool(row["final_answer"]))
        exact_match_samples += int(row["em"])

    return {
        "sample_count": sample_count,
        "format_valid_rate": ratio(format_valid_samples, sample_count),
        "search_tool_call_rate": ratio(search_samples, sample_count),
        "retriever_invocation_rate": ratio(retriever_samples, sample_count),
        "post_observation_continuation_rate": ratio(
            continued_after_observation, observation_events
        ),
        "post_observation_valid_action_rate": ratio(
            valid_after_observation, observation_events
        ),
        "average_search_turns": round(search_turns / sample_count, 6) if sample_count else None,
        "multi_search_rate": ratio(multi_search_samples, sample_count),
        "final_answer_rate": ratio(final_answer_samples, sample_count),
        "exact_match": ratio(exact_match_samples, sample_count),
        "malformed_generation_count": malformed_generations,
    }


def update_summary(
    *,
    config: DictConfig,
    results_dir: Path,
    group_key: str,
    rows: list[dict[str, Any]],
    health: dict[str, Any],
    length_limited_generations: int,
    context_exhausted_samples: int,
) -> None:
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "experiment": str(config.experiment_name),
            "sample_file": str(config.sample_file),
            "sample_ids": [row["sample_id"] for row in rows],
            "shared_inference_config": OmegaConf.to_container(
                config.inference, resolve=True
            ),
            "shared_agent_config": OmegaConf.to_container(config.agent, resolve=True),
            "retriever": {
                "url": str(config.retriever.url),
                "top_k": int(config.retriever.top_k),
                "health": health,
            },
            "groups": {},
        }
    if summary["sample_ids"] != [row["sample_id"] for row in rows]:
        raise RuntimeError("已有汇总的样本 ID 与本次运行不一致")
    group_summary = summarize(rows)
    group_summary["length_limited_generation_count"] = length_limited_generations
    group_summary["context_exhausted_sample_count"] = context_exhausted_samples
    summary["groups"][group_key] = group_summary
    write_json(summary_path, summary)


def run_group(
    *,
    config: DictConfig,
    engine: VllmEngine,
    search_client: HttpSearchClient,
    samples: list[dict[str, Any]],
    model_name: str,
    protocol: str,
    results_dir: Path,
    health: dict[str, Any],
) -> None:
    runner = AgentRunner(
        engine,
        search_client,
        max_search_turns=int(config.agent.max_search_turns),
        default_loop_steps=int(config.inference.loop_steps),
        max_new_tokens=config.inference.max_new_tokens,
        max_information_tokens=config.agent.max_information_tokens,
        max_response_tokens=config.inference.max_response_tokens,
        temperature=float(config.inference.temperature),
        top_p=float(config.inference.top_p),
        seed=int(config.inference.seed),
        prompt_profile=protocol,
    )
    output_path = results_dir / f"{model_name}_{protocol}.json"
    rows: list[dict[str, Any]] = []
    length_limited_generations = 0
    context_exhausted_samples = 0
    for index, sample in enumerate(samples, start=1):
        print(f"[{model_name}/{protocol}] {index}/{len(samples)} {sample['sample_id']}", flush=True)
        trajectory = runner.run(
            str(sample["question"]),
            sample_id=str(sample["sample_id"]),
            reference_answer=str(sample["reference_answer"]),
            answer_aliases=[str(value) for value in sample["answer_aliases"]],
        )
        rows.append(
            readable_trajectory(
                trajectory,
                sample=sample,
                model_name=model_name,
                protocol=protocol,
            )
        )
        length_limited_generations += sum(
            generation.finish_reason == "length" for generation in trajectory.generations
        )
        context_exhausted_samples += int(trajectory.termination_reason == "context_exhausted")
        write_json(output_path, rows)

    update_summary(
        config=config,
        results_dir=results_dir,
        group_key=f"{model_name}_{protocol}",
        rows=rows,
        health=health,
        length_limited_generations=length_limited_generations,
        context_exhausted_samples=context_exhausted_samples,
    )


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    if not isinstance(config, DictConfig):
        raise TypeError("配置文件根节点必须是 mapping")
    samples = load_samples(project_path(str(config.sample_file)))
    model_path = validate_local_model(config.models[args.model])
    health = validate_retriever(config.retriever)
    results_dir = project_path(str(config.results_dir))
    protocols = list(config.agent.protocols) if args.protocol == "all" else [args.protocol]
    for protocol in protocols:
        get_prompt_profile(str(protocol))

    print(f"模型: {args.model} ({model_path})", flush=True)
    print(f"样本: {len(samples)}", flush=True)
    print(f"Retriever: {json.dumps(health, ensure_ascii=False)}", flush=True)
    engine = VllmEngine(
        model_path,
        dtype=str(config.inference.dtype),
        trust_remote_code=bool(config.inference.trust_remote_code),
        max_model_len=int(config.inference.max_model_len),
        gpu_memory_utilization=float(config.inference.gpu_memory_utilization),
        tensor_parallel_size=1,
        fixed_loop_steps=int(config.inference.loop_steps),
    )
    search_client = HttpSearchClient(
        str(config.retriever.url),
        top_k=int(config.retriever.top_k),
        timeout_seconds=float(config.retriever.timeout_seconds),
    )
    for protocol in protocols:
        run_group(
            config=config,
            engine=engine,
            search_client=search_client,
            samples=samples,
            model_name=args.model,
            protocol=str(protocol),
            results_dir=results_dir,
            health=health,
        )


if __name__ == "__main__":
    main()
