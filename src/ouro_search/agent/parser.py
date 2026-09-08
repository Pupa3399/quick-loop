from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    SEARCH = "search"
    ANSWER = "answer"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ParsedOutput:
    action: ActionType
    think: str
    query: str | None = None
    answer: str | None = None
    raw_output: str = ""


_OPEN_TAG = re.compile(r"<(think|search|answer)>", re.IGNORECASE)


def _extract_last_tag(text: str, tag: str) -> tuple[str | None, int]:
    openings = list(re.finditer(rf"<{tag}>\s*", text, flags=re.IGNORECASE))
    if not openings:
        return None, -1
    opening = openings[-1]
    closing = re.search(rf"\s*</{tag}>", text[opening.end() :], flags=re.IGNORECASE)
    if closing:
        end = opening.end() + closing.start()
    else:
        next_tag = _OPEN_TAG.search(text, opening.end())
        end = next_tag.start() if next_tag else len(text)
    return text[opening.end() : end].strip(), opening.start()


def parse_agent_output(text: str) -> ParsedOutput:
    """Parse the latest action, tolerating a missing closing tag."""
    raw = text.strip()
    think, _ = _extract_last_tag(raw, "think")
    search, search_position = _extract_last_tag(raw, "search")
    answer, answer_position = _extract_last_tag(raw, "answer")

    candidates: list[tuple[int, ActionType, str]] = []
    if search is not None and search.strip():
        candidates.append((search_position, ActionType.SEARCH, search.strip()))
    if answer is not None and answer.strip():
        candidates.append((answer_position, ActionType.ANSWER, answer.strip()))

    if not candidates:
        return ParsedOutput(action=ActionType.UNKNOWN, think=think or "", raw_output=raw)

    _, action, value = max(candidates, key=lambda candidate: candidate[0])
    if action is ActionType.SEARCH:
        return ParsedOutput(
            action=action,
            think=think or "",
            query=value,
            raw_output=raw,
        )
    return ParsedOutput(
        action=action,
        think=think or "",
        answer=value,
        raw_output=raw,
    )
