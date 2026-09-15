#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ouro_search.trajectory.prescreen import (  # noqa: E402
    iter_jsonl_records,
    load_reference_index,
    prescreen,
    write_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rule-based prescreening of Search-R1 decision turns"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/Search-R1_3B_rollout",
        help="A JSONL file or a directory containing JSONL shards",
    )
    parser.add_argument(
        "--reference-parquet",
        type=Path,
        default=PROJECT_ROOT / "data/searchr1/train.parquet",
        help="Optional normalized Search-R1 parquet used to enrich answer aliases",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/qwen_prefilt/search_r1_qwen3b_decision_turns_v1.jsonl",
    )
    parser.add_argument("--per-type", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_type <= 0:
        parser.error("--per-type must be positive")

    reference_index = load_reference_index(
        args.reference_parquet if args.reference_parquet.exists() else None
    )
    candidates, stats = prescreen(
        iter_jsonl_records(args.input),
        reference_index=reference_index,
        per_type=args.per_type,
        seed=args.seed,
    )
    write_candidates(args.output, candidates)
    selected_counts = Counter(row["candidate_type"] for row in candidates)
    strata = Counter(
        f"{row['candidate_type']}|{row['source']}|em={row['baseline_em']}" for row in candidates
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.resolve()),
        "reference_parquet": str(args.reference_parquet.resolve())
        if args.reference_parquet.exists()
        else None,
        "output": str(args.output.resolve()),
        "per_type_requested": args.per_type,
        "seed": args.seed,
        "stats": stats.to_dict(),
        "selected_counts": dict(selected_counts),
        "selected_strata": dict(sorted(strata.items())),
        "notes": [
            "Gold evidence is a normalized reference-answer substring proxy, not a semantic "
            "sufficiency label.",
            "bad_next_hop is deliberately low confidence and uses later retrieval outcome "
            "as hindsight.",
            "Each source trajectory contributes at most one, earliest-priority candidate.",
        ],
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
