import pickle
from types import SimpleNamespace

from transformers import PretrainedConfig

from ouro_search.vllm_model.config import stable_worker_vllm_config


class _DynamicOuroConfig(PretrainedConfig):
    model_type = "ouro"


def test_stable_worker_config_replaces_only_the_transmitted_hf_config() -> None:
    hf_config = _DynamicOuroConfig(
        architectures=["OuroForCausalLM"],
        total_ut_steps=3,
        ouro_num_hidden_layers=48,
    )
    model_config = SimpleNamespace(hf_config=hf_config, hf_text_config=hf_config)
    vllm_config = SimpleNamespace(model_config=model_config, marker="engine-owned")

    worker_config = stable_worker_vllm_config(vllm_config)

    assert worker_config is not vllm_config
    assert worker_config.model_config is not model_config
    assert vllm_config.model_config.hf_config is hf_config
    assert type(worker_config.model_config.hf_config) is PretrainedConfig
    assert worker_config.model_config.hf_text_config is worker_config.model_config.hf_config
    assert worker_config.model_config.hf_config.model_type == "ouro"
    assert worker_config.model_config.hf_config.total_ut_steps == 3
    assert worker_config.marker == "engine-owned"
    pickle.dumps(worker_config)


def test_stable_worker_config_leaves_other_models_untouched() -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=PretrainedConfig(model_type="llama"))
    )

    assert stable_worker_vllm_config(vllm_config) is vllm_config
