from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

HF_MIRROR = "https://hf-mirror.com"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "main/models"
MODEL_ID = "ByteDance/Ouro-2.6B-Thinking"
MODEL_REVISION = "f1edd81e7ac41355db670500ceaf204e0f73af68"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed-depth vLLM view of Ouro")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--loop-steps", type=int, choices=(3, 4), default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=MODEL_ROOT / "huggingface",
    )
    args = parser.parse_args()
    if os.environ.get("HF_ENDPOINT") != HF_MIRROR:
        raise RuntimeError(f"Set HF_ENDPOINT={HF_MIRROR}; overseas fallback is disabled")
    snapshot = Path(
        snapshot_download(
            args.model_id,
            revision=args.revision,
            cache_dir=args.cache_dir,
            endpoint=HF_MIRROR,
            local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
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

    output_dir = args.output_dir or MODEL_ROOT / f"vllm-ouro-r{loop_steps}"
    output_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot = output_dir / "source_snapshot"
    if source_snapshot.is_symlink() and source_snapshot.resolve() != snapshot.resolve():
        source_snapshot.unlink()
    if not source_snapshot.exists():
        source_snapshot.symlink_to(snapshot.resolve(), target_is_directory=True)
    for source in snapshot.iterdir():
        if source.name == "config.json":
            continue
        destination = output_dir / source.name
        if destination.is_symlink() and destination.resolve() != source.resolve():
            destination.unlink()
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
