from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
ORIGINAL_SEARCH_LIMIT_TEXT = "You can search at most four times."

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "main/models/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from ouro_search.agent.parser import ActionType, parse_agent_output  # noqa: E402
from ouro_search.agent.profiles import get_prompt_profile  # noqa: E402
from ouro_search.search.client import HttpSearchClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重新运行 Search-R1 Qwen3B 的 20 条样本")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        cases = [json.loads(line) for line in handle if line.strip()]
    if len(cases) != 20:
        raise ValueError(f"rerollout 固定要求 20 条样本，实际为 {len(cases)} 条")
    sample_ids = [str(case["sample_id"]) for case in cases]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("rerollout 输入包含重复 sample_id")
    return cases


def load_original_messages(
    path: Path,
    cases: list[dict[str, Any]],
) -> dict[str, list[dict[str, str]]]:
    table = pq.read_table(path, columns=["dataset", "prompt", "extra_info"])
    wanted = {
        (str(case["source"]), int(case["orig_index"])): str(case["sample_id"])
        for case in cases
    }
    messages_by_sample: dict[str, list[dict[str, str]]] = {}
    for row in table.to_pylist():
        extra_info = row.get("extra_info") or {}
        key = (str(row.get("dataset", "")), int(extra_info.get("index", -1)))
        sample_id = wanted.get(key)
        if sample_id is None:
            continue
        messages = [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in row["prompt"]
        ]
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError(f"{sample_id} 的原始 Prompt 不是单条 user message")
        if ORIGINAL_SEARCH_LIMIT_TEXT not in messages[0]["content"]:
            raise ValueError(f"{sample_id} 未使用原始四次 Search Prompt")
        messages_by_sample[sample_id] = messages
    missing = sorted(set(wanted.values()) - set(messages_by_sample))
    if missing:
        raise ValueError(f"Search-R1 parquet 缺少样本: {missing}")
    return messages_by_sample


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
        raise RuntimeError("rerollout 要求真实 GPU FAISS Retriever")
    return health


def render_initial_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    encoded_text = list(tokenizer.encode(text, add_special_tokens=False))
    if token_ids != encoded_text:
        raise RuntimeError("chat template 的文本与 token 渲染结果不一致")
    return text, token_ids


def initialize_trajectories(
    cases: list[dict[str, Any]],
    messages_by_sample: dict[str, list[dict[str, str]]],
    tokenizer: Any,
    config: DictConfig,
) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    draws = int(config.sampling.trajectories_per_sample)
    for case_index, case in enumerate(cases):
        sample_id = str(case["sample_id"])
        prompt_text, prompt_token_ids = render_initial_prompt(
            tokenizer,
            messages_by_sample[sample_id],
        )
        for draw_index in range(draws):
            seed = int(config.sampling.base_seed) + case_index * draws + draw_index
            trajectories.append(
                {
                    "sample_id": sample_id,
                    "seed": seed,
                    "prompt_text": prompt_text,
                    "prompt_token_ids": prompt_token_ids,
                    "turns": [],
                    "original_target_turn": int(case["target_turn"]),
                    "original_target_action": str(case["target_action"]),
                    "_prefix_token_ids": list(prompt_token_ids),
                    "_response_token_count": 0,
                    "_done": False,
                    "_termination_reason": "max_search_actions",
                }
            )
    return trajectories


def public_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in trajectory.items() if not key.startswith("_")}


def save_trajectories(path: Path, trajectories: list[dict[str, Any]]) -> None:
    write_json(path, [public_trajectory(trajectory) for trajectory in trajectories])


def make_sampling_params(config: DictConfig, seed: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=float(config.sampling.temperature),
        top_p=float(config.sampling.top_p),
        max_tokens=int(config.sampling.max_generation_tokens),
        seed=seed,
        stop=list(config.agent.stop_sequences),
        include_stop_str_in_output=True,
    )


def run_rollouts(
    llm: Any,
    tokenizer: Any,
    trajectories: list[dict[str, Any]],
    search_client: HttpSearchClient,
    config: DictConfig,
    results_path: Path,
) -> None:
    from vllm.inputs import TokensPrompt

    profile = get_prompt_profile("search_r1")
    max_search_actions = int(config.agent.max_search_actions)
    max_information_tokens = int(config.agent.max_information_tokens)
    max_response_tokens = int(config.sampling.max_response_tokens)

    for turn_id in range(max_search_actions + 1):
        active = [trajectory for trajectory in trajectories if not trajectory["_done"]]
        if not active:
            break
        prompts = [
            TokensPrompt(prompt_token_ids=trajectory["_prefix_token_ids"])
            for trajectory in active
        ]
        params = [make_sampling_params(config, int(trajectory["seed"])) for trajectory in active]
        outputs = llm.generate(prompts, params, use_tqdm=True)
        if len(outputs) != len(active):
            raise RuntimeError("vLLM 返回的序列数与请求数不一致")

        for trajectory, request_output in zip(active, outputs, strict=True):
            output = request_output.outputs[0]
            generated_token_ids = [int(token_id) for token_id in output.token_ids]
            raw_generation_text = tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            parse_text = tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            parsed = parse_agent_output(parse_text)
            prefix_token_ids = list(trajectory["_prefix_token_ids"])
            full_prefix_text = tokenizer.decode(
                prefix_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            turn = {
                "turn": turn_id,
                "raw_generation_text": raw_generation_text,
                "generated_token_ids": generated_token_ids,
                "information_text": "",
                "information_token_ids": [],
                "full_prefix_text": full_prefix_text,
                "full_prefix_token_ids": prefix_token_ids,
                "thinking": parsed.think,
                "query": parsed.query or "",
                "answer": parsed.answer or "",
                "action": parsed.action.value,
            }
            trajectory["turns"].append(turn)
            trajectory["_prefix_token_ids"].extend(generated_token_ids)
            trajectory["_response_token_count"] += len(generated_token_ids)

            if parsed.action is ActionType.ANSWER:
                trajectory["_done"] = True
                trajectory["_termination_reason"] = "answer"
                continue
            if parsed.action is not ActionType.SEARCH:
                trajectory["_done"] = True
                trajectory["_termination_reason"] = "unparseable_output"
                continue
            if turn_id == max_search_actions:
                trajectory["_done"] = True
                trajectory["_termination_reason"] = "max_search_actions"
                continue

            search_result = search_client.search(parsed.query or "")
            information = profile.format_observation(search_result)
            information_token_ids = list(
                tokenizer.encode(information, add_special_tokens=False)
            )[:max_information_tokens]
            information_text = tokenizer.decode(
                information_token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            turn["information_text"] = information_text
            turn["information_token_ids"] = information_token_ids
            trajectory["_prefix_token_ids"].extend(information_token_ids)
            trajectory["_response_token_count"] += len(information_token_ids)
            if trajectory["_response_token_count"] >= max_response_tokens:
                trajectory["_done"] = True
                trajectory["_termination_reason"] = "max_response_tokens"

        save_trajectories(results_path, trajectories)

    for trajectory in trajectories:
        trajectory["termination_reason"] = trajectory["_termination_reason"]
    save_trajectories(results_path, trajectories)


def information_boundary_summary(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    complete_forms: Counter[tuple[str, tuple[int, ...], str]] = Counter()
    missing_close = 0
    for trajectory in trajectories:
        for turn in trajectory["turns"]:
            text = str(turn["information_text"])
            if not text:
                continue
            closing = "</information>"
            closing_position = text.rfind(closing)
            if closing_position < 0:
                missing_close += 1
                continue
            before_close = text[:closing_position]
            linebreak_match = re.search(r"[\r\n]*$", before_close)
            linebreaks = linebreak_match.group(0) if linebreak_match else ""
            after_close = text[closing_position + len(closing) :]
            complete_forms[
                (
                    linebreaks,
                    tuple(int(value) for value in turn["information_token_ids"][-4:]),
                    after_close,
                )
            ] += 1
    return {
        "count": sum(complete_forms.values()) + missing_close,
        "complete_close_count": sum(complete_forms.values()),
        "missing_close_due_to_500_token_truncation_count": missing_close,
        "complete_forms": [
            {
                "count": count,
                "linebreaks_before_information_close": linebreaks,
                "last_4_information_token_ids": list(token_ids),
                "text_after_information_close": after_close,
            }
            for (linebreaks, token_ids, after_close), count in complete_forms.most_common()
        ],
    }


def summarize(
    trajectories: list[dict[str, Any]],
    retriever_health: dict[str, Any],
) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    reached = 0
    reproduced = 0
    trajectories_with_search = 0
    trajectories_with_answer = 0
    for trajectory in trajectories:
        actions = [str(turn["action"]) for turn in trajectory["turns"]]
        action_counts.update(actions)
        trajectories_with_search += int("search" in actions)
        trajectories_with_answer += int("answer" in actions)
        target_turn = int(trajectory["original_target_turn"])
        if target_turn < len(trajectory["turns"]):
            reached += 1
            target_action = str(trajectory["turns"][target_turn]["action"])
            target_counts[target_action] += 1
            reproduced += int(target_action == trajectory["original_target_action"])

    trajectory_count = len(trajectories)
    action_total = sum(action_counts.values())
    return {
        "trajectory_count": trajectory_count,
        "sample_count": len({trajectory["sample_id"] for trajectory in trajectories}),
        "action_counts": dict(sorted(action_counts.items())),
        "search_action_rate": action_counts["search"] / action_total if action_total else 0.0,
        "answer_action_rate": action_counts["answer"] / action_total if action_total else 0.0,
        "trajectory_search_rate": trajectories_with_search / trajectory_count,
        "trajectory_answer_rate": trajectories_with_answer / trajectory_count,
        "target_turn_reached": reached,
        "target_turn_action_counts": dict(sorted(target_counts.items())),
        "original_target_action_reproduction_count": reproduced,
        "original_target_action_reproduction_rate_all": reproduced / trajectory_count,
        "original_target_action_reproduction_rate_reached": (
            reproduced / reached if reached else 0.0
        ),
        "termination_reasons": dict(
            sorted(
                Counter(
                    str(trajectory["termination_reason"])
                    for trajectory in trajectories
                ).items()
            )
        ),
        "information_boundaries": information_boundary_summary(trajectories),
        "retriever_health": retriever_health,
    }


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    if not isinstance(config, DictConfig):
        raise TypeError("rerollout 配置根节点必须是 mapping")
    if int(config.sampling.trajectories_per_sample) != 4:
        raise ValueError("rerollout 固定要求每题 4 条轨迹")
    if int(config.agent.max_search_actions) != 4:
        raise ValueError("rerollout 固定要求最多 4 次 Search")

    cases = load_cases(project_path(str(config.input_file)))
    messages = load_original_messages(
        project_path(str(config.searchr1_data_file)),
        cases,
    )
    retriever_health = validate_retriever(config.retriever)

    from vllm import LLM

    llm = LLM(
        model=str(project_path(str(config.model.path))),
        tokenizer=str(project_path(str(config.model.path))),
        dtype=str(config.model.dtype),
        trust_remote_code=bool(config.model.trust_remote_code),
        max_model_len=int(config.model.max_model_len),
        gpu_memory_utilization=float(config.model.gpu_memory_utilization),
        tensor_parallel_size=int(config.model.tensor_parallel_size),
        enforce_eager=bool(config.model.enforce_eager),
    )
    tokenizer = llm.get_tokenizer()
    trajectories = initialize_trajectories(cases, messages, tokenizer, config)
    search_client = HttpSearchClient(
        str(config.retriever.url),
        top_k=int(config.retriever.top_k),
        timeout_seconds=float(config.retriever.timeout_seconds),
    )
    results_path = project_path(str(config.results_file))
    run_rollouts(
        llm,
        tokenizer,
        trajectories,
        search_client,
        config,
        results_path,
    )
    summary = summarize(trajectories, retriever_health)
    write_json(project_path(str(config.summary_file)), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
