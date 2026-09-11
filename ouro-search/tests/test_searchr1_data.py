from ouro_search.data.searchr1 import SEARCH_R1_PROMPT, normalize_example


def test_normalized_record_preserves_aliases_and_verl_fields() -> None:
    record = normalize_example(
        {"question": "Capital of France", "golden_answers": ["Paris", "The Paris"]},
        dataset="nq",
        split="train",
        index=7,
    )
    assert record is not None
    assert record["id"] == "nq-train-7"
    assert record["question"] == "Capital of France?"
    assert record["reference_answer"] == "Paris"
    assert record["answer_aliases"] == ["Paris", "The Paris"]
    assert record["reward_model"]["ground_truth"]["target"] == ["Paris", "The Paris"]
    assert "<search> query </search>" in SEARCH_R1_PROMPT


def test_empty_question_or_answers_are_filtered() -> None:
    assert normalize_example({}, dataset="nq", split="train", index=0) is None
