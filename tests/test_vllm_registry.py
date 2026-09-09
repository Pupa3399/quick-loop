import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="vLLM extra not installed")
def test_ouro_can_be_registered_out_of_tree() -> None:
    from vllm.model_executor.models import ModelRegistry

    from ouro_search.vllm_model import register_ouro_vllm

    register_ouro_vllm()
    assert "OuroForCausalLM" in ModelRegistry.get_supported_archs()
