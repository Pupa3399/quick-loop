from __future__ import annotations

import re
import string
from collections.abc import Sequence
from typing import Any

_ANSWER_RE = re.compile(r"<answer\b[^>]*>(.*?)(?:</answer>|$)", re.IGNORECASE | re.DOTALL)


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def extract_final_answer(model_output: str) -> str | None:
    matches = list(_ANSWER_RE.finditer(model_output))
    return matches[-1].group(1).strip() if matches else None


def answer_exact_match(prediction: str, aliases: str | Sequence[str]) -> bool:
    values = [aliases] if isinstance(aliases, str) else aliases
    normalized_prediction = normalize_answer(prediction)
    return any(normalized_prediction == normalize_answer(alias) for alias in values)


def compute_answer_reward(model_output: str, aliases: str | Sequence[str]) -> float:
    """Return pure final-answer EM reward; protocol/search behavior earns no bonus."""
    answer = extract_final_answer(model_output)
    return float(answer is not None and answer_exact_match(answer, aliases))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | Sequence[str] | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """veRL custom reward entry point using pure final-answer exact match."""
    del data_source, extra_info, kwargs
    aliases = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
    return compute_answer_reward(solution_str, aliases)
