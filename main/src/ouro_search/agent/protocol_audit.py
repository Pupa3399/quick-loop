from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from ouro_search.agent.parser import ActionType
from ouro_search.agent.profiles import get_prompt_profile

_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
_SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*</search>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_SEARCH_R1_CONTROL_RE = re.compile(r"</?(?:think|search|information|answer)\b", re.I)
_HERMES_CONTROL_RE = re.compile(r"</?(?:tool_call|tool_response)\b", re.I)
_SEARCH_R1_STRICT_RE = re.compile(
    r"\s*<think>\s*.+?\s*</think>\s*"
    r"<(?P<action>search|answer)>\s*.+?\s*</(?P=action)>\s*",
    re.IGNORECASE | re.DOTALL,
)
_HERMES_TOOL_STRICT_RE = re.compile(
    r"\s*<tool_call>\s*\{.*\}\s*</tool_call>\s*", re.IGNORECASE | re.DOTALL
)


@dataclass(frozen=True, slots=True)
class GenerationAudit:
    kind: str
    valid_action: bool
    format_valid: bool
    malformed: bool
    attempted_search: bool
    reason: str
    explicit_protocol_format: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _audit_search_r1(text: str) -> GenerationAudit:
    raw = text.strip()
    parsed = get_prompt_profile("search_r1").parse_action(raw)
    searches = [match for match in _SEARCH_RE.finditer(raw) if match.group(1).strip()]
    answers = [match for match in _ANSWER_RE.finditer(raw) if match.group(1).strip()]
    actions = [(match.start(), "search") for match in searches]
    actions.extend((match.start(), "answer") for match in answers)
    attempted_search = "<search" in raw.lower()
    if parsed.action is ActionType.UNKNOWN:
        reason = "missing_action"
        if _SEARCH_R1_CONTROL_RE.search(raw):
            reason = "unclosed_or_empty_tag"
        return GenerationAudit(
            kind="unknown",
            valid_action=False,
            format_valid=False,
            malformed=True,
            attempted_search=attempted_search,
            reason=reason,
            explicit_protocol_format=False,
        )
    kind = parsed.action.value
    action_position = raw.lower().rfind(f"<{kind}>")
    thinks = [match for match in _THINK_RE.finditer(raw) if match.group(1).strip()]
    has_prior_think = any(match.start() < action_position for match in thinks)
    strict_match = _SEARCH_R1_STRICT_RE.fullmatch(raw)
    strict_format = bool(
        strict_match
        and strict_match.group("action").lower() == kind
        and len(thinks) == 1
        and len(searches) + len(answers) == 1
    )
    return GenerationAudit(
        kind=kind,
        valid_action=True,
        format_valid=strict_format,
        malformed=not strict_format,
        attempted_search=attempted_search,
        reason=(
            "ok"
            if strict_format
            else "missing_closed_think_before_action"
            if not has_prior_think
            else "multiple_actions"
            if len(actions) > 1
            else "unclosed_or_empty_tag"
            if not any(action_kind == kind for _, action_kind in actions)
            else "noncanonical_layout"
        ),
        explicit_protocol_format=strict_format,
    )


def _audit_hermes(text: str) -> GenerationAudit:
    raw = text.strip()
    parsed = get_prompt_profile("hermes").parse_action(raw)
    attempted_search = "<tool_call" in raw.lower()
    if parsed.action is ActionType.SEARCH:
        strict_format = bool(_HERMES_TOOL_STRICT_RE.fullmatch(raw))
        return GenerationAudit(
            "search",
            True,
            strict_format,
            not strict_format,
            True,
            "ok" if strict_format else "noncanonical_tool_call_layout",
            explicit_protocol_format=strict_format,
        )
    if parsed.action is ActionType.ANSWER:
        if _SEARCH_R1_CONTROL_RE.search(raw) or _HERMES_CONTROL_RE.search(raw):
            return GenerationAudit(
                "unknown",
                False,
                False,
                True,
                attempted_search,
                "foreign_or_stray_control_tag",
                explicit_protocol_format=False,
            )
        return GenerationAudit(
            "answer", True, True, False, False, "ok", explicit_protocol_format=False
        )
    reason = "malformed_tool_call" if attempted_search else "missing_action"
    return GenerationAudit(
        "unknown", False, False, True, attempted_search, reason, explicit_protocol_format=False
    )


def audit_generation(profile_name: str, text: str) -> GenerationAudit:
    if profile_name == "search_r1":
        return _audit_search_r1(text)
    if profile_name == "hermes":
        return _audit_hermes(text)
    raise ValueError(f"Unsupported prompt profile: {profile_name}")
