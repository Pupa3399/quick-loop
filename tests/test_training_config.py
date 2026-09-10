from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from verl.utils.import_utils import import_external_libs
from verl.workers.config import FSDPEngineConfig


def test_actor_fsdp_config_matches_pinned_verl() -> None:
    config_path = Path("configs/train/verl_searchr1_grpo.yaml")
    config = OmegaConf.load(config_path)
    fsdp_config = OmegaConf.to_container(
        config.actor_rollout_ref.actor.fsdp_config,
        resolve=True,
    )

    assert isinstance(fsdp_config, dict)
    assert "grad_offload" not in fsdp_config
    FSDPEngineConfig(**fsdp_config)


def test_hf_training_uses_available_sdpa_backend() -> None:
    config = OmegaConf.load("configs/train/verl_searchr1_grpo.yaml")

    assert config.actor_rollout_ref.model.override_config.attn_implementation == "sdpa"


def test_verl_external_lib_is_a_single_importable_module() -> None:
    config = OmegaConf.load("configs/train/verl_searchr1_grpo.yaml")
    external_lib = config.actor_rollout_ref.model.external_lib

    assert isinstance(external_lib, str)
    import_external_libs(external_lib)
