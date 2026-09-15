from __future__ import annotations

from ouro_search.trajectory.prescreen import build_candidate, prescreen, token_jaccard


def _doc(docid: str, text: str, title: str = "Title") -> dict[str, str]:
    return {"docid": docid, "title": title, "text": text}


def _record(
    sample_id: str,
    turns: list[dict[str, object]],
    *,
    answer: str = "Ada Lovelace",
    em: int = 0,
    source: str = "nq",
    finish_reason: str = "answer_generated",
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "source": source,
        "orig_index": 1,
        "question": "Who wrote the first computer program?",
        "answer": answer,
        "pred_answer": "wrong",
        "em": em,
        "finish_reason": finish_reason,
        "trajectory": turns,
    }


def test_jaccard_ignores_common_question_words() -> None:
    assert token_jaccard("Who wrote the program?", "program author") == 1 / 3


def test_over_search_targets_first_search_after_answer_string() -> None:
    record = _record(
        "a",
        [
            {
                "query": "first computer program author",
                "retrieved_docs": [_doc("1", "Ada Lovelace wrote it")],
            },
            {"query": "Ada Lovelace biography", "retrieved_docs": [_doc("2", "Biography")]},
            {"query": "", "thinking": "answer", "retrieved_docs": []},
        ],
        em=1,
    )
    candidate, exclusion = build_candidate(record, {"path": "part.jsonl", "line_number": 1})
    assert exclusion is None
    assert candidate is not None
    assert candidate["candidate_type"] == "over_search"
    assert candidate["target_turn"] == 1
    assert candidate["rule_signals"]["prefix_search_count"] == 1
    assert candidate["rule_signals"]["gold_answer_string_hit_before"] is True


def test_stagnation_detects_repeated_query() -> None:
    record = _record(
        "b",
        [
            {"query": "program author", "retrieved_docs": [_doc("1", "No answer")]},
            {"query": "program author", "retrieved_docs": [_doc("2", "Still no answer")]},
            {"query": "", "retrieved_docs": []},
        ],
    )
    candidate, _ = build_candidate(record, {})
    assert candidate is not None
    assert candidate["candidate_type"] == "search_stagnation"
    assert candidate["rule_signals"]["query_exact_match_previous"] is True


def test_bad_next_hop_is_explicitly_low_confidence() -> None:
    record = _record(
        "c",
        [
            {
                "query": "computer program history",
                "retrieved_docs": [_doc("1", "Partial context", "Computing")],
            },
            {"query": "unrelated subject", "retrieved_docs": [_doc("2", "No answer", "Other")]},
            {
                "query": "first programmer",
                "retrieved_docs": [_doc("3", "The answer is Ada Lovelace")],
            },
            {"query": "", "retrieved_docs": []},
        ],
        source="hotpotqa",
    )
    candidate, _ = build_candidate(record, {})
    assert candidate is not None
    assert candidate["candidate_type"] == "bad_next_hop"
    assert candidate["target_turn"] == 1
    assert candidate["confidence"] <= 0.64
    assert candidate["rule_signals"]["later_search_first_brought_gold_string"] is True


def test_evidence_use_failure_targets_answer_turn() -> None:
    record = _record(
        "d",
        [
            {"query": "first programmer", "retrieved_docs": [_doc("1", "It was Ada Lovelace")]},
            {"query": "", "thinking": "It was someone else", "retrieved_docs": []},
        ],
    )
    candidate, _ = build_candidate(record, {})
    assert candidate is not None
    assert candidate["candidate_type"] == "evidence_use_failure"
    assert candidate["target_action"] == "answer"
    assert candidate["target_turn"] == 1


def test_excludes_empty_retrieval_and_length_failure() -> None:
    empty = _record("empty", [{"query": "search", "retrieved_docs": []}])
    truncated = _record(
        "length",
        [{"query": "search", "retrieved_docs": [_doc("1", "text")]}],
        finish_reason="length_no_action",
    )
    assert build_candidate(empty, {}) == (None, "retriever_anomaly")
    assert build_candidate(truncated, {}) == (None, "truncation")


def test_prescreen_balances_available_source_em_strata() -> None:
    records = []
    for index, (source, em) in enumerate([("nq", 0), ("nq", 1), ("hotpotqa", 0), ("hotpotqa", 1)]):
        record = _record(
            str(index),
            [
                {"query": "author", "retrieved_docs": [_doc("1", "Ada Lovelace")]},
                {"query": "more", "retrieved_docs": [_doc("2", "more")]},
                {"query": "", "retrieved_docs": []},
            ],
            source=source,
            em=em,
        )
        records.append((record, {"line_number": index + 1}))
    candidates, stats = prescreen(records, per_type=4)
    selected = [row for row in candidates if row["candidate_type"] == "over_search"]
    assert len(selected) == 4
    assert {(row["source"], row["baseline_em"]) for row in selected} == {
        ("nq", 0),
        ("nq", 1),
        ("hotpotqa", 0),
        ("hotpotqa", 1),
    }
    assert stats.records_seen == 4
