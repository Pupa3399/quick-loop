from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/data2/wuguanting/quick_loop/data/searchr1/test.parquet")
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "zero_shot_agent_v1.jsonl"
SELECTION_VERSION = "zero-shot-agent-v1"
TEMPORAL_PATTERN = re.compile(
    r"\b(current|currently|latest|next|now|recent|today|tonight|upcoming)\b",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"\b[\w'-]+\b")


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", value.lower()).split())


def _rank(row: dict[str, Any]) -> str:
    value = f"{SELECTION_VERSION}\0{row['id']}".encode()
    return hashlib.sha256(value).hexdigest()


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_answer_leakage(row: dict[str, Any]) -> bool:
    question = _normalize(str(row["question"]))
    return any(
        len(normalized := _normalize(str(alias))) >= 3 and normalized in question
        for alias in row["answer_aliases"]
    )


def _eligible(rows: Iterable[dict[str, Any]], *, min_words: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for row in rows:
        question = str(row["question"]).strip()
        normalized_question = _normalize(question)
        aliases = [str(alias).strip() for alias in row["answer_aliases"] if str(alias).strip()]
        if (
            not question
            or not aliases
            or len(WORD_PATTERN.findall(question)) < min_words
            or TEMPORAL_PATTERN.search(question)
            or normalized_question in seen_questions
        ):
            continue
        normalized_row = {**row, "question": question, "answer_aliases": aliases}
        if _has_answer_leakage(normalized_row):
            continue
        seen_questions.add(normalized_question)
        selected.append(normalized_row)
    return selected


def select_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    nq = _eligible((row for row in rows if row["dataset"] == "nq"), min_words=8)
    hotpot = _eligible(
        (row for row in rows if row["dataset"] == "hotpotqa"), min_words=10
    )
    hotpot_binary = [
        row for row in hotpot if _normalize(str(row["reference_answer"])) in {"yes", "no"}
    ]
    hotpot_span = [
        row for row in hotpot if _normalize(str(row["reference_answer"])) not in {"yes", "no"}
    ]

    strata = [
        ("nq_external", nq, 50, ["external_knowledge", "hard_without_search"]),
        (
            "hotpotqa_span_multihop",
            hotpot_span,
            40,
            ["external_knowledge", "hard_without_search", "multi_hop"],
        ),
        (
            "hotpotqa_binary_multihop",
            hotpot_binary,
            10,
            ["comparison", "external_knowledge", "hard_without_search", "multi_hop"],
        ),
    ]
    output: list[dict[str, Any]] = []
    for stratum, candidates, count, tags in strata:
        ranked = sorted(candidates, key=_rank)
        if len(ranked) < count:
            raise RuntimeError(
                f"Selection stratum {stratum} has {len(ranked)} eligible rows, needs {count}"
            )
        for row in ranked[:count]:
            output.append(
                {
                    "id": str(row["id"]),
                    "dataset": str(row["dataset"]),
                    "question": str(row["question"]),
                    "reference_answer": str(row["reference_answer"]),
                    "answer_aliases": list(row["answer_aliases"]),
                    "selection_stratum": stratum,
                    "difficulty_tags": tags,
                }
            )
    return sorted(output, key=lambda row: (row["dataset"], row["id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the zero-shot Agent evaluation set")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(f"Evaluation source does not exist: {args.source}")

    table = pq.read_table(
        args.source,
        columns=["id", "dataset", "question", "reference_answer", "answer_aliases"],
    )
    examples = select_examples(table.to_pylist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n" for example in examples
    )
    args.output.write_text(content, encoding="utf-8")

    counts: dict[str, int] = {}
    strata: dict[str, int] = {}
    for example in examples:
        counts[example["dataset"]] = counts.get(example["dataset"], 0) + 1
        stratum = example["selection_stratum"]
        strata[stratum] = strata.get(stratum, 0) + 1
    manifest_sha256 = hashlib.sha256(content.encode()).hexdigest()
    metadata = {
        "selection_version": SELECTION_VERSION,
        "source": str(args.source),
        "source_sha256": _source_sha256(args.source),
        "manifest": str(args.output),
        "manifest_sha256": manifest_sha256,
        "samples": len(examples),
        "dataset_counts": counts,
        "stratum_counts": strata,
        "rules": {
            "common": [
                "non-empty question and answer aliases",
                "deduplicate normalized questions",
                "exclude time-sensitive cue words",
                "exclude answer aliases copied verbatim in the question",
                "rank candidates by SHA-256(selection_version + NUL + source ID)",
            ],
            "nq_external": "50 NQ rows with at least 8 words",
            "hotpotqa_span_multihop": "40 non-binary HotpotQA rows with at least 10 words",
            "hotpotqa_binary_multihop": "10 yes/no HotpotQA rows with at least 10 words",
        },
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
