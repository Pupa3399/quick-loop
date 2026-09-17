"""Inference engines."""

from ouro_search.inference.qwen_hidden_loop import (
    HIDDEN_LATEST_DEPTH_KV,
    HiddenLoopGeneration,
    HiddenLoopState,
    HiddenLoopTokenRecord,
    QwenHiddenLoopEngine,
)
from ouro_search.inference.qwen_latent_loop import (
    LATEST_DEPTH_KV,
    LatentLoopGeneration,
    LatentLoopState,
    LatentLoopTokenRecord,
    QwenLatentLoopEngine,
    SoftFeedback,
)
from ouro_search.inference.transformers_engine import TransformersEngine
from ouro_search.inference.types import GenerationResult
from ouro_search.inference.vllm_engine import VllmEngine

__all__ = [
    "LATEST_DEPTH_KV",
    "GenerationResult",
    "HIDDEN_LATEST_DEPTH_KV",
    "HiddenLoopGeneration",
    "HiddenLoopState",
    "HiddenLoopTokenRecord",
    "LatentLoopGeneration",
    "LatentLoopState",
    "LatentLoopTokenRecord",
    "QwenLatentLoopEngine",
    "QwenHiddenLoopEngine",
    "SoftFeedback",
    "TransformersEngine",
    "VllmEngine",
]
