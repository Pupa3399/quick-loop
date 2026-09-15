from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GenerationResult:
    """Lossless inference result used by evaluation and replay tooling."""

    text: str
    rendered_prompt: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    finish_reason: str | None
    stop_reason: str | int | None
    latency_seconds: float
    chosen_token_logprobs: list[float | None] = field(default_factory=list)
    token_top_logprobs: list[list[dict[str, Any]]] = field(default_factory=list)

