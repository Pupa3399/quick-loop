from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ouro_search.agent.profiles.search_r1 import SEARCH_R1_PROMPT


def _aliases(example: Mapping[str, Any]) -> list[str]:
    raw = example.get("golden_answers", example.get("answers", example.get("answer", [])))
    if isinstance(raw, Mapping):
        raw = raw.get("text", [])
    if isinstance(raw, str):
        values: Sequence[Any] = [raw]
    elif isinstance(raw, Sequence):
        values = raw
    else:
        values = []
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def normalize_example(
    example: Mapping[str, Any], *, dataset: str, split: str, index: int
) -> dict[str, Any] | None:
    question = str(example.get("question", "")).strip()
    aliases = _aliases(example)
    if not question or not aliases:
        return None
    if not question.endswith("?"):
        question += "?"
    sample_id = f"{dataset}-{split}-{index}"
    return {
        "id": sample_id,
        "question": question,
        "reference_answer": aliases[0],
        "answer_aliases": aliases,
        "dataset": dataset,
        "data_source": dataset,
        "prompt": [{"role": "user", "content": SEARCH_R1_PROMPT.format(question=question)}],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": aliases}},
        "extra_info": {"split": split, "index": index, "id": sample_id},
    }
