from __future__ import annotations

import re
from typing import Any

_OPEN_RE = re.compile(r"<information\b[^>]*>", re.IGNORECASE)
_CLOSE_RE = re.compile(r"</information\s*>", re.IGNORECASE)


def information_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    while opening := _OPEN_RE.search(text, cursor):
        closing = _CLOSE_RE.search(text, opening.end())
        end = closing.end() if closing else len(text)
        spans.append((opening.start(), end))
        cursor = end
        if not closing:
            break
    return spans


def build_information_token_mask(
    tokenizer: Any, text: str, *, add_special_tokens: bool = False
) -> list[int]:
    """Build a policy mask: generated tokens=1 and retriever information tokens=0."""
    encoded = tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        return_offsets_mapping=True,
    )
    offsets = encoded["offset_mapping"]
    if offsets and isinstance(offsets[0], list):
        if len(offsets) != 1:
            raise ValueError("build_information_token_mask accepts one text at a time")
        offsets = offsets[0]
    spans = information_spans(text)
    return [
        int(
            start == end
            or not any(
                start < span_end and end > span_start for span_start, span_end in spans
            )
        )
        for start, end in offsets
    ]
