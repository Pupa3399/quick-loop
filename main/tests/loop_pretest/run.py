from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import httpx
import torch
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "main/models/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ouro_search.agent.parser import ActionType  # noqa: E402
from ouro_search.agent.profiles import get_prompt_profile  # noqa: E402
from ouro_search.agent.rendering import render_agent_prompt  # noqa: E402
from ouro_search.inference.qwen_latent_loop import (  # noqa: E402
    LatentLoopState,
    LatentLoopTokenRecord,
    QwenLatentLoopEngine,
)
from ouro_search.rewards.answer_reward import answer_exact_match  # noqa: E402
from ouro_search.search.client import HttpSearchClient  # noqa: E402
from ouro_search.search.types import Document  # noqa: E402
from ouro_search.trajectory.schema import TurnRecord  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Search-R1 Qwen3B R1/R2 turn 预实验")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    if len(cases) != 20:
        raise ValueError(f"预实验固定输入应为 20 条，实际为 {len(cases)} 条")
    sample_ids = [str(case["sample_id"]) for case in cases]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("筛选数据包含重复 sample_id")
    for case in cases:
        annotation = case.get("human_annotation", {})
        target_turn = int(annotation.get("target_turn", case["target_turn"]))
        prefix = case.get("prefix_history", [])
        if annotation.get("decision") != "keep":
            raise ValueError(f"{case['sample_id']} 不是人工 keep 样本")
        if target_turn != int(case["target_turn"]) or target_turn != len(prefix):
            raise ValueError(f"{case['sample_id']} 的 target_turn 与 prefix_history 不一致")
        if case.get("target_action") != "search":
            raise ValueError(f"{case['sample_id']} 的目标 action 不是 search")
    return cases


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_retriever(config: DictConfig) -> dict[str, Any]:
    response = httpx.get(
        f"{str(config.url).rstrip('/')}/health",
        timeout=float(config.timeout_seconds),
    )
    response.raise_for_status()
    health = response.json()
    if int(health.get("index_size", -1)) != int(config.expected_index_size):
        raise RuntimeError("Retriever index_size 与 Wiki18 Flat 基线不一致")
    if bool(config.require_gpu_faiss) and not health.get("faiss_gpu"):
        raise RuntimeError("预实验要求真实 GPU FAISS Retriever")
    return health


def prefix_turns(case: dict[str, Any]) -> list[TurnRecord]:
    turns: list[TurnRecord] = []
    for turn_id, source in enumerate(case["prefix_history"]):
        thinking = str(source.get("thinking", ""))
        query = str(source.get("query", ""))
        snippets = str(source.get("snippets", ""))
        documents = [
            Document(
                id=str(document.get("docid", document.get("id", ""))),
                title=str(document.get("title", "")),
                text=str(document.get("text", "")),
                score=float(document.get("score", 0.0)),
            )
            for document in source.get("retrieved_docs", [])
        ]
        turns.append(
            TurnRecord(
                turn_id=turn_id,
                loop_steps=1,
                think=thinking,
                query=query,
                information=f"<information>{snippets}</information>",
                retrieved_documents=documents,
                model_output=f"<think>{thinking}</think>\n<search>{query}</search>",
            )
        )
    return turns


def create_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def generate_action(
    engine: QwenLatentLoopEngine,
    state: LatentLoopState,
    *,
    depth: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
    stop_sequences: tuple[str, ...],
) -> tuple[str, list[LatentLoopTokenRecord], str]:
    token_ids: list[int] = []
    records: list[LatentLoopTokenRecord] = []
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        record = engine.step(
            state,
            depth=depth,
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        records.append(record)
        token_ids.append(record.token_id)
        if engine.tokenizer.eos_token_id == record.token_id:
            stop_reason = "eos"
            break
        text = engine.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        matched = next((stop for stop in stop_sequences if stop in text), None)
        if matched is not None:
            stop_reason = matched
            break
    text = engine.tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return text, records, stop_reason


def append_observation(
    engine: QwenLatentLoopEngine,
    state: LatentLoopState,
    information: str,
    *,
    max_tokens: int,
) -> tuple[str, int]:
    token_ids = engine.tokenizer(
        information,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][:, :max_tokens]
    if token_ids.shape[1] < 1:
        raise RuntimeError("Retriever observation 编码后为空")
    rendered = engine.tokenizer.decode(
        token_ids[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    engine.append_visible_tokens(state, token_ids)
    return rendered, int(token_ids.shape[1])


def token_diagnostics(records: list[LatentLoopTokenRecord]) -> list[dict[str, Any]]:
    return [
        {
            "token_index": index,
            "token": record.final_output_text,
            "depth": record.depth,
            "r1_top1": record.r1_top1_text,
            "r2_top1": record.r2_top1_text,
            "final_token": record.final_output_text,
            "logical_position": record.logical_position,
        }
        for index, record in enumerate(records)
    ]


def run_branch(
    *,
    case: dict[str, Any],
    branch_depth: int,
    draw_index: int,
    seed: int,
    engine: QwenLatentLoopEngine,
    search_client: HttpSearchClient,
    config: DictConfig,
) -> dict[str, Any]:
    profile = get_prompt_profile("search_r1")
    fixed_prefix = prefix_turns(case)
    target_turn = int(case["target_turn"])
    prompt = profile.build_prompt(str(case["question"]), fixed_prefix)
    rendered_prompt = render_agent_prompt(engine.tokenizer, prompt, use_chat_template=True)
    input_ids = engine.tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    state = engine.prefill(input_ids)
    generator = create_generator(engine.device, seed)
    response_tokens = 0
    search_turns = len(fixed_prefix)
    generated_turns: list[dict[str, Any]] = []
    termination_reason = "max_search_turns"
    final_answer = ""

    for generated_turn_index in range(int(config.agent.max_search_turns) + 1):
        turn_id = target_turn + generated_turn_index
        depth = branch_depth if turn_id == target_turn else 1
        remaining = int(config.sampling.max_response_tokens) - response_tokens
        if remaining <= 0:
            termination_reason = "max_response_tokens"
            break
        max_new_tokens = min(int(config.sampling.max_new_tokens_per_turn), remaining)
        output, records, stop_reason = generate_action(
            engine,
            state,
            depth=depth,
            max_new_tokens=max_new_tokens,
            temperature=float(config.sampling.temperature),
            top_p=float(config.sampling.top_p),
            generator=generator,
            stop_sequences=tuple(profile.stop_sequences),
        )
        response_tokens += len(records)
        parsed = profile.parse_action(output)
        turn = {
            "turn": turn_id,
            "loop_depth": depth,
            "model_output": output,
            "action": parsed.action.value,
            "search_query": parsed.query or "",
            "information": "",
            "generation_stop_reason": stop_reason,
            "num_generated_tokens": len(records),
            "token_diagnostics": token_diagnostics(records) if turn_id == target_turn else [],
        }
        generated_turns.append(turn)

        if parsed.action is ActionType.ANSWER:
            final_answer = parsed.answer or ""
            termination_reason = "answer"
            break
        if parsed.action is not ActionType.SEARCH:
            termination_reason = "unparseable_output"
            break
        if search_turns >= int(config.agent.max_search_turns):
            termination_reason = "max_search_turns"
            break

        remaining_information_tokens = (
            int(config.sampling.max_response_tokens) - response_tokens
        )
        if remaining_information_tokens <= 0:
            termination_reason = "max_response_tokens"
            break

        response = search_client.search(parsed.query or "")
        information = profile.format_observation(response)
        information, information_tokens = append_observation(
            engine,
            state,
            information,
            max_tokens=min(
                int(config.agent.max_information_tokens),
                remaining_information_tokens,
            ),
        )
        response_tokens += information_tokens
        turn["information"] = information
        search_turns += 1

    aliases = [str(value) for value in case["reference_answers"]]
    prefix_output = [
        {
            "turn": turn.turn_id,
            "loop_depth": None,
            "model_output": turn.model_output,
            "action": "search",
            "search_query": turn.query,
            "information": turn.information,
            "generation_stop_reason": "fixed_prefix",
            "num_generated_tokens": None,
            "token_diagnostics": [],
        }
        for turn in fixed_prefix
    ]
    all_turns = prefix_output + generated_turns
    return {
        "trajectory_id": f"{case['sample_id']}-r{branch_depth}-sample-{draw_index}",
        "sample_id": str(case["sample_id"]),
        "candidate_type": str(case["candidate_type"]),
        "target_turn": target_turn,
        "target_depth": branch_depth,
        "sample_index": draw_index,
        "seed": seed,
        "question": str(case["question"]),
        "reference_answers": aliases,
        "initial_prompt": rendered_prompt,
        "turns": all_turns,
        "num_agent_turns": len(all_turns),
        "num_generated_agent_turns": len(generated_turns),
        "num_search_turns": search_turns,
        "final_answer": final_answer,
        "em": int(bool(final_answer) and answer_exact_match(final_answer, aliases)),
        "termination_reason": termination_reason,
        "response_tokens": response_tokens,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "trajectory_count": count,
        "sample_count": len({row["sample_id"] for row in rows}),
        "answer_rate": sum(row["termination_reason"] == "answer" for row in rows) / count,
        "em": sum(int(row["em"]) for row in rows) / count,
        "average_agent_turns": sum(int(row["num_agent_turns"]) for row in rows) / count,
        "average_generated_agent_turns": sum(
            int(row["num_generated_agent_turns"]) for row in rows
        )
        / count,
        "average_search_turns": sum(int(row["num_search_turns"]) for row in rows) / count,
        "termination_reasons": {
            reason: sum(row["termination_reason"] == reason for row in rows)
            for reason in sorted({str(row["termination_reason"]) for row in rows})
        },
    }


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    if not isinstance(config, DictConfig):
        raise TypeError("配置文件根节点必须是 mapping")
    if str(config.agent.prompt_profile) != "search_r1":
        raise ValueError("本预实验只允许 Search-R1 protocol")
    if list(config.sampling.depths) != [1, 2]:
        raise ValueError("本预实验必须同时运行 R1 与 R2")
    if int(config.sampling.trajectories_per_depth) != 4:
        raise ValueError("每个 depth 必须采样 4 条轨迹")

    cases = load_cases(project_path(str(config.input_file)))
    health = validate_retriever(config.retriever)
    engine = QwenLatentLoopEngine.from_pretrained(
        project_path(str(config.model.path)),
        dtype=getattr(torch, str(config.model.dtype)),
        device=str(config.model.device),
        local_files_only=bool(config.model.local_files_only),
    )
    search_client = HttpSearchClient(
        str(config.retriever.url),
        top_k=int(config.retriever.top_k),
        timeout_seconds=float(config.retriever.timeout_seconds),
    )
    results_dir = project_path(str(config.results_dir))
    rows_by_depth: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    draws = int(config.sampling.trajectories_per_depth)

    for case_index, case in enumerate(cases):
        for draw_index in range(draws):
            seed = int(config.sampling.base_seed) + case_index * draws + draw_index
            for depth in (1, 2):
                print(
                    f"[{case_index + 1}/{len(cases)}] {case['sample_id']} "
                    f"R{depth} sample={draw_index} seed={seed}",
                    flush=True,
                )
                row = run_branch(
                    case=case,
                    branch_depth=depth,
                    draw_index=draw_index,
                    seed=seed,
                    engine=engine,
                    search_client=search_client,
                    config=config,
                )
                rows_by_depth[depth].append(row)
                write_json(results_dir / f"r{depth}_trajectories.json", rows_by_depth[depth])

    summary = {
        "experiment": str(config.experiment_name),
        "input_file": str(config.input_file),
        "retriever_health": health,
        "sampling": OmegaConf.to_container(config.sampling, resolve=True),
        "r1": summarize(rows_by_depth[1]),
        "r2": summarize(rows_by_depth[2]),
    }
    write_json(results_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
