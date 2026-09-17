from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "main/models/searchr1/searchr1-qwen2.5-3b-grpo-v0.2"
INPUT_PATH = (
    PROJECT_ROOT
    / "main/data/qwen_prefilt/search_r1_qwen3b_decision_turns_human_20.jsonl"
)
OUTPUT_PATH = Path(__file__).with_name("results") / "single_token_smoke.json"

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ouro_search.agent.profiles import get_prompt_profile  # noqa: E402
from ouro_search.agent.rendering import render_agent_prompt  # noqa: E402
from ouro_search.inference.qwen_hidden_loop import QwenHiddenLoopEngine  # noqa: E402
from ouro_search.inference.qwen_latent_loop import QwenLatentLoopEngine  # noqa: E402
from ouro_search.search.types import Document  # noqa: E402
from ouro_search.trajectory.schema import TurnRecord  # noqa: E402


def load_first_case() -> dict[str, Any]:
    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def build_target_prompt(engine: QwenHiddenLoopEngine, case: dict[str, Any]) -> str:
    prefix: list[TurnRecord] = []
    for turn_id, source in enumerate(case["prefix_history"]):
        thinking = str(source["thinking"])
        query = str(source["query"])
        snippets = str(source["snippets"])
        documents = [
            Document(
                id=str(document["docid"]),
                title=str(document["title"]),
                text=str(document["text"]),
                score=0.0,
            )
            for document in source["retrieved_docs"]
        ]
        prefix.append(
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
    profile = get_prompt_profile("search_r1")
    prompt = profile.build_prompt(str(case["question"]), prefix)
    return render_agent_prompt(engine.tokenizer, prompt, use_chat_template=True)


def token_result(result: Any) -> dict[str, Any]:
    record = result.records[0]
    return {
        "token_id": record.final_output_token_id,
        "token": record.final_output_text,
        "r1_top1": record.r1_top1_text,
        "r2_top1": record.r2_top1_text,
        "position_ids_by_depth": record.position_ids_by_depth,
        "logits_max_abs_difference": getattr(
            record,
            "logits_max_abs_difference",
            None,
        ),
    }


def main() -> None:
    hidden_engine = QwenHiddenLoopEngine.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device="cuda:0",
        local_files_only=True,
        alpha=0.5,
    )
    case = load_first_case()
    prompt = build_target_prompt(hidden_engine, case)
    input_ids = hidden_engine.tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]
    r1 = hidden_engine.generate_ids(input_ids, max_new_tokens=1, depth=1, eos_token_id=-1)
    hidden_r2 = hidden_engine.generate_ids(
        input_ids,
        max_new_tokens=1,
        depth=2,
        eos_token_id=-1,
    )
    old_engine = QwenLatentLoopEngine(hidden_engine.model, hidden_engine.tokenizer)
    soft_r2 = old_engine.generate_ids(input_ids, max_new_tokens=1, depth=2, eos_token_id=-1)
    result = {
        "sample_id": case["sample_id"],
        "alpha": hidden_engine.alpha,
        "r1": token_result(r1),
        "hidden_r2": token_result(hidden_r2),
        "old_soft_token_r2": token_result(soft_r2),
        "hidden_r2_preserves_first_token": hidden_r2.token_ids == r1.token_ids,
        "old_phase_shift_reproduced": soft_r2.token_ids != r1.token_ids,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if hidden_r2.token_ids != r1.token_ids:
        raise RuntimeError("hidden-state R2 changed the Search-R1 tag's first greedy token")


if __name__ == "__main__":
    main()
