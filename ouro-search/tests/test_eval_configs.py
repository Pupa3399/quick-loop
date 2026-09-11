from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

MODEL_CONFIGS = (
    "searchr1_qwen2_5_3b_grpo_v0_2",
    "searchr1_qwen2_5_7b_grpo_v0_2",
    "ouro_2_6b_r3",
    "ouro_2_6b_r4",
)


def test_unified_eval_model_configs_share_experiment_controls() -> None:
    agent = OmegaConf.load("configs/agent/default.yaml")
    search = OmegaConf.load("configs/search/e5_wiki18.yaml")

    assert agent.max_search_turns == 4
    assert agent.max_new_tokens == 500
    assert agent.max_information_tokens == 500
    assert agent.max_response_tokens == 3000
    assert search.top_k == 3
    assert search.timeout_seconds == 120.0
    for name in MODEL_CONFIGS:
        model = OmegaConf.load(Path("configs/model") / f"{name}.yaml")
        assert model.backend == "vllm"
        assert model.dtype == "bfloat16"
        assert model.max_model_len == 7096
        assert model.hf_endpoint == "https://hf-mirror.com"


def test_search_r1_revisions_are_immutable_shas() -> None:
    for name in MODEL_CONFIGS[:2]:
        config = OmegaConf.load(Path("configs/model") / f"{name}.yaml")
        assert len(config.revision) == 40
        assert all(character in "0123456789abcdef" for character in config.revision)
