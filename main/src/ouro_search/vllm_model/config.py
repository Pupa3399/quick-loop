from __future__ import annotations

import copy
from typing import Any

from transformers import PretrainedConfig


def stable_worker_vllm_config(vllm_config: Any) -> Any:
    """Return a worker-safe copy when an Ouro config comes from remote code."""
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if getattr(hf_config, "model_type", None) != "ouro":
        return vllm_config
    if type(hf_config) is PretrainedConfig:
        return vllm_config

    worker_config = copy.copy(vllm_config)
    worker_model_config = copy.copy(model_config)
    stable_hf_config = PretrainedConfig.from_dict(hf_config.to_dict())
    stable_hf_config.model_type = "ouro"
    for attribute, value in vars(worker_model_config).items():
        if value is hf_config:
            setattr(worker_model_config, attribute, stable_hf_config)
    worker_config.model_config = worker_model_config
    return worker_config
