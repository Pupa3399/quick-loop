from __future__ import annotations

import pytest

from ouro_search.evaluation import aggregate_records, audit_generation, evaluate_trajectory
from ouro_search.search.types import Document
from ouro_search.trajectory.schema import Trajectory, TurnRecord


def test_search_r1_strict_audit_distinguishes_action_and_format() -> None:
    valid = audit_generation(
        "search_r1", "<think>Need evidence.</think><search>Hamlet author</search>"
    )
    missing_think = audit_generation("search_r1", "<answer>Shakespeare</answer>")
    unclosed = audit_generation("search_r1", "<think>Need evidence.</think><search>Hamlet")

    assert valid.kind == "search" and valid.format_valid
    assert missing_think.kind == "answer" and missing_think.valid_action
    assert not missing_think.format_valid and missing_think.malformed
    assert unclosed.kind == "unknown" and unclosed.attempted_search


def test_hermes_audit_accepts_official_actions_and_rejects_foreign_tags() -> None:
    tool = audit_generation(
        "hermes",
        '<tool_call>{"arguments": {"query": "Hamlet"}, "name": "search"}</tool_call>',
    )
    answer = audit_generation("hermes", "William Shakespeare")
    foreign = audit_generation("hermes", "<answer>William Shakespeare</answer>")

    assert tool.kind == "search" and tool.format_valid
    assert answer.kind == "answer" and answer.format_valid
    assert foreign.malformed and foreign.kind == "unknown"


def test_trajectory_metrics_include_observation_followup_and_failure_class() -> None:
    document = Document(
        id="wiki-1",
        title="Hamlet",
        text="Hamlet was written by William Shakespeare.",
        score=1.0,
    )
    trajectory = Trajectory(
        id="hotpotqa-test-1",
        question="Who wrote Hamlet?",
        prediction="Christopher Marlowe",
        reference_answer="William Shakespeare",
        answer_aliases=["William Shakespeare"],
        reward=0.0,
        num_search_turns=1,
        turns=[
            TurnRecord(
                turn_id=0,
                loop_steps=3,
                think="Need evidence.",
                query="Hamlet author",
                information="<information>William Shakespeare</information>",
                retrieved_documents=[document],
                model_output="<think>Need evidence.</think><search>Hamlet author</search>",
            )
        ],
        termination_reason="answer",
        final_model_output=(
            "<think>Use evidence.</think><answer>Christopher Marlowe</answer>"
        ),
        raw_generations=[
            "<think>Need evidence.</think><search>Hamlet author</search>",
            "<think>Use evidence.</think><answer>Christopher Marlowe</answer>",
        ],
        response_token_count=20,
    )

    record = evaluate_trajectory(
        trajectory, dataset="hotpotqa", profile_name="search_r1"
    )
    metrics = aggregate_records([record])

    assert record["failure_class"] == "information_utilization_failure"
    assert metrics["format_valid_rate"] == pytest.approx(1.0)
    assert metrics["valid_search_tool_action_rate"] == pytest.approx(1.0)
    assert metrics["post_observation_valid_continuation_rate"] == pytest.approx(1.0)
    assert metrics["em"] == pytest.approx(0.0)
