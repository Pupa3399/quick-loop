from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset
from ouro_search.data.searchr1 import normalize_example

HF_MIRROR = "https://hf-mirror.com"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_REPO = "RUC-NLPIR/FlashRAG_datasets"


def require_hf_mirror() -> None:
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint != HF_MIRROR:
        raise RuntimeError(f"Set HF_ENDPOINT={HF_MIRROR}; overseas fallback is disabled")


def prepare_split(
    dataset_name: str, split: str, cache_dir: Path
) -> tuple[Dataset, int, str]:
    source_splits = load_dataset(SOURCE_REPO, dataset_name, cache_dir=str(cache_dir))
    candidates = (split,) if split == "train" else ("test", "dev", "train")
    source_split = next((candidate for candidate in candidates if candidate in source_splits), None)
    if source_split is None:
        raise RuntimeError(f"{dataset_name} has none of the expected splits: {candidates}")
    source = source_splits[source_split]
    records: list[dict[str, Any]] = []
    filtered = 0
    for index, example in enumerate(source):
        normalized = normalize_example(
            example, dataset=dataset_name, split=split, index=index
        )
        if normalized is None:
            filtered += 1
        else:
            records.append(normalized)
    return Dataset.from_list(records), filtered, source_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Search-R1 NQ + HotpotQA parquet")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "main/data/searchr1",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "main/datasets/huggingface",
    )
    args = parser.parse_args()
    require_hf_mirror()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"source": SOURCE_REPO, "datasets": {}, "filtered": 0}
    for split, output_name in (("train", "train.parquet"), ("test", "test.parquet")):
        prepared: list[Dataset] = []
        split_total = 0
        for dataset_name in ("nq", "hotpotqa"):
            dataset, filtered, source_split = prepare_split(
                dataset_name, split, args.cache_dir
            )
            prepared.append(dataset)
            split_total += len(dataset)
            report["filtered"] += filtered
            report["datasets"].setdefault(dataset_name, {})[split] = len(dataset)
            report["datasets"][dataset_name][f"{split}_filtered"] = filtered
            report["datasets"][dataset_name][f"{split}_source_split"] = source_split
        concatenate_datasets(prepared).to_parquet(args.output_dir / output_name)
        report[split] = split_total
    report["total"] = report["train"] + report["test"]
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
