from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

HF_MIRROR = "https://hf-mirror.com"
MODEL_ROOT = Path("/data2/wuguanting/quick_loop/data/models")
MODELS = {
    "3b": {
        "model_id": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-grpo-v0.2",
        "revision": "7ff32234abe1e1bf1dbbbf3a1385686b52ecfea2",
        "path": MODEL_ROOT / "searchr1-qwen2.5-3b-grpo-v0.2",
    },
    "7b": {
        "model_id": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-grpo-v0.2",
        "revision": "deaa9d14b92dd4d481414e77b9f733c934df830a",
        "path": MODEL_ROOT / "searchr1-qwen2.5-7b-grpo-v0.2",
    },
}


def _artifact_bytes(path: Path) -> int:
    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file() and ".cache" not in file.relative_to(path).parts
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pinned Search-R1 v0.2 baselines")
    parser.add_argument("models", nargs="*", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR:
        raise RuntimeError(f"Set HF_ENDPOINT={HF_MIRROR}; overseas fallback is disabled")

    reports = []
    for name in args.models:
        spec = MODELS[name]
        local_path = Path(spec["path"])
        snapshot_download(
            repo_id=str(spec["model_id"]),
            revision=str(spec["revision"]),
            local_dir=local_path,
            endpoint=HF_MIRROR,
            max_workers=args.max_workers,
        )
        reports.append(
            {
                **spec,
                "path": str(local_path),
                "artifact_bytes": _artifact_bytes(local_path),
            }
        )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
