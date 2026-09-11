from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from ouro_search.agent.parser import ActionType
from ouro_search.agent.profiles import get_prompt_profile
from ouro_search.trajectory.schema import Trajectory

_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
_SEARCH_RE = re.compile(r"<search>\s*(.*?)\s*</search>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_SEARCH_R1_CONTROL_RE = re.compile(r"</?(?:think|search|information|answer)\b", re.I)
_HERMES_CONTROL_RE = re.compile(r"</?(?:tool_call|tool_response)\b", re.I)


@dataclass(frozen=True, slots=True)
class GenerationAudit:
    kind: str
    valid_action: bool
    format_valid: bool
    malformed: bool
    attempted_search: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _audit_search_r1(text: str) -> GenerationAudit:
    raw = text.strip()
    searches = [match for match in _SEARCH_RE.finditer(raw) if match.group(1).strip()]
    answers = [match for match in _ANSWER_RE.finditer(raw) if match.group(1).strip()]
    actions = [(match.start(), "search") for match in searches]
    actions.extend((match.start(), "answer") for match in answers)
    attempted_search = "<search" in raw.lower()
    if len(actions) != 1:
        reason = "missing_action" if not actions else "multiple_actions"
        if _SEARCH_R1_CONTROL_RE.search(raw) and not actions:
            reason = "unclosed_or_empty_tag"
        return GenerationAudit(
            kind="unknown",
            valid_action=False,
            format_valid=False,
            malformed=True,
            attempted_search=attempted_search,
            reason=reason,
        )
    action_position, kind = actions[0]
    thinks = [match for match in _THINK_RE.finditer(raw) if match.group(1).strip()]
    has_prior_think = any(match.start() < action_position for match in thinks)
    return GenerationAudit(
        kind=kind,
        valid_action=True,
        format_valid=has_prior_think,
        malformed=not has_prior_think,
        attempted_search=attempted_search,
        reason="ok" if has_prior_think else "missing_closed_think_before_action",
    )


def _audit_hermes(text: str) -> GenerationAudit:
    raw = text.strip()
    parsed = get_prompt_profile("hermes").parse_action(raw)
    attempted_search = "<tool_call" in raw.lower()
    if parsed.action is ActionType.SEARCH:
        return GenerationAudit("search", True, True, False, True, "ok")
    if parsed.action is ActionType.ANSWER:
        # Search-R1 action tags are not valid Hermes final responses.
        if _SEARCH_R1_CONTROL_RE.search(raw) or _HERMES_CONTROL_RE.search(raw):
            return GenerationAudit(
                "unknown", False, False, True, attempted_search, "foreign_or_stray_control_tag"
            )
        return GenerationAudit("answer", True, True, False, False, "ok")
    reason = "malformed_tool_call" if attempted_search else "missing_action"
    return GenerationAudit("unknown", False, False, True, attempted_search, reason)


def audit_generation(profile_name: str, text: str) -> GenerationAudit:
    if profile_name == "search_r1":
        return _audit_search_r1(text)
    if profile_name == "hermes":
        return _audit_hermes(text)
    raise ValueError(f"Unsupported evaluation profile: {profile_name}")


def _normalized(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", value.lower()).split())


def _retrieved_answer_evidence(trajectory: Trajectory) -> bool:
    aliases = [_normalized(alias) for alias in trajectory.answer_aliases]
    aliases = [alias for alias in aliases if len(alias) >= 3 and alias not in {"yes", "no"}]
    for turn in trajectory.turns:
        for document in turn.retrieved_documents:
            evidence = _normalized(f"{document.title} {document.text}")
            if any(alias in evidence for alias in aliases):
                return True
    return False


def _failure_class(
    trajectory: Trajectory,
    audits: list[GenerationAudit],
) -> str:
    if trajectory.reward == 1.0:
        return "success"
    malformed_search_attempt = any(audit.attempted_search and audit.malformed for audit in audits)
    if malformed_search_attempt:
        return "protocol_failure"
    if not trajectory.turns:
        return "no_search"
    if audits and audits[-1].malformed:
        return "protocol_failure"
    if _retrieved_answer_evidence(trajectory):
        return "information_utilization_failure"
    return "search_decision_failure"


def evaluate_trajectory(
    trajectory: Trajectory,
    *,
    dataset: str,
    profile_name: str,
) -> dict[str, Any]:
    audits = [audit_generation(profile_name, output) for output in trajectory.raw_generations]
    final_answer = bool(audits and audits[-1].kind == "answer" and audits[-1].valid_action)
    valid_observation_followups = sum(
        int(index + 1 < len(audits) and audits[index + 1].valid_action)
        for index in range(len(trajectory.turns))
    )
    return {
        "id": trajectory.id,
        "dataset": dataset,
        "profile": profile_name,
        "question": trajectory.question,
        "reference_answer": trajectory.reference_answer,
        "answer_aliases": trajectory.answer_aliases,
        "prediction": trajectory.prediction,
        "em": float(trajectory.reward or 0.0),
        "termination_reason": trajectory.termination_reason,
        "response_token_count": trajectory.response_token_count,
        "num_generations": len(trajectory.raw_generations),
        "num_search_turns": trajectory.num_search_turns,
        "retriever_invoked": bool(trajectory.turns),
        "valid_search_action": any(audit.kind == "search" for audit in audits),
        "valid_final_answer": final_answer,
        "sample_format_valid": bool(
            audits and final_answer and all(audit.format_valid for audit in audits)
        ),
        "malformed_action": any(audit.malformed for audit in audits),
        "valid_action_count": sum(audit.valid_action for audit in audits),
        "valid_search_action_count": sum(audit.kind == "search" for audit in audits),
        "observation_count": len(trajectory.turns),
        "valid_observation_followup_count": valid_observation_followups,
        "retrieved_answer_evidence": _retrieved_answer_evidence(trajectory),
        "failure_class": _failure_class(trajectory, audits),
        "action_audits": [audit.to_dict() for audit in audits],
        "trajectory": trajectory.to_dict(),
    }


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    generations = sum(int(record["num_generations"]) for record in records)
    observations = sum(int(record["observation_count"]) for record in records)
    searched = [record for record in records if record["retriever_invoked"]]
    not_searched = [record for record in records if not record["retriever_invoked"]]
    failures = Counter(str(record["failure_class"]) for record in records)
    return {
        "samples": count,
        "format_valid_rate": _ratio(
            sum(bool(record["sample_format_valid"]) for record in records), count
        ),
        "valid_action_rate": _ratio(
            sum(int(record["valid_action_count"]) for record in records), generations
        ),
        "malformed_action_rate": _ratio(
            sum(bool(record["malformed_action"]) for record in records), count
        ),
        "valid_search_tool_action_rate": _ratio(
            sum(bool(record["valid_search_action"]) for record in records), count
        ),
        "valid_search_tool_generation_rate": _ratio(
            sum(int(record["valid_search_action_count"]) for record in records), generations
        ),
        "retriever_invocation_rate": _ratio(len(searched), count),
        "post_observation_valid_continuation_rate": _ratio(
            sum(int(record["valid_observation_followup_count"]) for record in records),
            observations,
        ),
        "mean_search_turns": _ratio(
            sum(int(record["num_search_turns"]) for record in records), count
        ),
        "multi_turn_search_rate": _ratio(
            sum(int(record["num_search_turns"]) >= 2 for record in records), count
        ),
        "four_turn_limit_rate": _ratio(
            sum(int(record["num_search_turns"]) >= 4 for record in records), count
        ),
        "final_answer_rate": _ratio(
            sum(bool(record["valid_final_answer"]) for record in records), count
        ),
        "em": _ratio(sum(float(record["em"]) for record in records), count),
        "searched_em": _ratio(sum(float(record["em"]) for record in searched), len(searched)),
        "not_searched_em": _ratio(
            sum(float(record["em"]) for record in not_searched), len(not_searched)
        ),
        "mean_response_tokens": _ratio(
            sum(int(record["response_token_count"]) for record in records), count
        ),
        "failure_counts": dict(sorted(failures.items())),
    }
