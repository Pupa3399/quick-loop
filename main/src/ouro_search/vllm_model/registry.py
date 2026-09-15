from __future__ import annotations


def register_ouro_vllm() -> None:
    """Register Ouro lazily so parent processes do not initialize CUDA."""
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "OuroForCausalLM", "ouro_search.vllm_model.ouro:OuroForCausalLM"
    )
