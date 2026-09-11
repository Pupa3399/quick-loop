from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

HF_MIRROR = "https://hf-mirror.com"
MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"
MODEL_REVISION = "f1edd81e7ac41355db670500ceaf204e0f73af68"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed-depth vLLM view of Ouro")
    parser.add_argument("--loop-steps", type=int, choices=(3, 4), default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/huggingface"))
    args = parser.parse_args()
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR:
        raise RuntimeError(f"Set HF_ENDPOINT={HF_MIRROR}; overseas fallback is disabled")
    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=args.cache_dir,
            endpoint=HF_MIRROR,
        )
    )
    original_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    vllm_config = dict(original_config)
    logical_layers = int(vllm_config["num_hidden_layers"])
    loop_steps = args.loop_steps
    vllm_config["ouro_num_hidden_layers"] = logical_layers
    vllm_config["num_hidden_layers"] = logical_layers * loop_steps
    vllm_config["total_ut_steps"] = loop_steps
    vllm_config["early_exit_threshold"] = 1.0

    output_dir = args.output_dir or Path(
        f"/data2/wuguanting/quick_loop/data/vllm-ouro-r{loop_steps}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in snapshot.iterdir():
        if source.name == "config.json":
            continue
        destination = output_dir / source.name
        if not destination.exists():
            destination.symlink_to(source.resolve())
    (output_dir / "config.ouro-original.json").write_text(
        json.dumps(original_config, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(vllm_config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"prepared={output_dir}")
    print(f"logical_layers={logical_layers} effective_cache_layers={logical_layers * loop_steps}")


if __name__ == "__main__":
    main()
