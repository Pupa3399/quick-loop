from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from ouro_search.inference import TransformersEngine  # noqa: E402
from ouro_search.models import OuroModel  # noqa: E402


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    if os.environ["HF_ENDPOINT"] != config.model.hf_endpoint:
        raise RuntimeError("HF_ENDPOINT must use the configured domestic mirror")
    model = OuroModel(
        model_name=config.model.model_name,
        revision=config.model.revision,
        dtype=config.model.dtype,
        trust_remote_code=config.model.trust_remote_code,
        total_ut_steps=config.model.total_ut_steps,
        freeze_exit_gate=config.model.freeze_exit_gate,
        device=config.model.device,
        cache_dir=str(PROJECT_ROOT / config.model.cache_dir),
    )
    engine = TransformersEngine(model, use_cache=config.model.use_cache)
    if not model.exit_gate_is_frozen():
        raise RuntimeError("exit gate is not frozen")

    for loop_steps in (3, 4):
        output = engine.generate(
            "What is 12 multiplied by 7? Answer briefly.",
            max_new_tokens=128,
            temperature=0.0,
            top_p=1.0,
            greedy=True,
            loop_steps=loop_steps,
        )
        if not output:
            raise RuntimeError(f"R{loop_steps} produced an empty output")
        print(f"[R{loop_steps}] {output}")


if __name__ == "__main__":
    main()
