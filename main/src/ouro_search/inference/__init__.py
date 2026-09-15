"""Inference engines."""

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
    "LatentLoopGeneration",
    "LatentLoopState",
    "LatentLoopTokenRecord",
    "QwenLatentLoopEngine",
    "SoftFeedback",
    "TransformersEngine",
    "VllmEngine",
]
