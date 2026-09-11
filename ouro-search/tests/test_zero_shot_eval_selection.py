from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_SPEC = importlib.util.spec_from_file_location(
    "prepare_zero_shot_agent_eval",
    Path(__file__).parents[1] / "scripts" / "prepare_zero_shot_agent_eval.py",
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)
select_examples = MODULE.select_examples


def _row(index: int, dataset: str, answer: str = "answer") -> dict[str, object]:
    return {
        "id": f"{dataset}-test-{index}",
        "dataset": dataset,
        "question": (
            f"Which historical person connected subject {index} to another documented event?"
        ),
        "reference_answer": answer,
        "answer_aliases": [answer],
    }


def test_selection_is_balanced_deterministic_and_leak_free() -> None:
    rows = [_row(index, "nq") for index in range(80)]
    rows.extend(_row(index, "hotpotqa") for index in range(80))
    rows.extend(_row(index + 100, "hotpotqa", "yes") for index in range(30))
    rows.append(
        {
            **_row(999, "nq", "leaked answer"),
            "question": "Which person has the name leaked answer in this difficult question?",
        }
    )

    first = select_examples(rows)
    second = select_examples(list(reversed(rows)))

    assert first == second
    assert len(first) == 100
    assert sum(row["dataset"] == "nq" for row in first) == 50
    assert sum(row["dataset"] == "hotpotqa" for row in first) == 50
    assert sum(row["reference_answer"] == "yes" for row in first) == 10
    assert all(row["id"] != "nq-test-999" for row in first)
