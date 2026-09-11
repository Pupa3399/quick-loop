from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "ByteDance/Ouro-2.6B-Thinking"


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    supported = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return supported[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose one of {sorted(supported)}") from exc


class OuroModel:
    """Thin wrapper around the official Hugging Face Ouro implementation."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        revision: str | None = None,
        dtype: str | torch.dtype = "bfloat16",
        trust_remote_code: bool = True,
        total_ut_steps: int = 3,
        freeze_exit_gate: bool = True,
        device: str | torch.device | None = None,
        cache_dir: str | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not trust_remote_code:
            raise ValueError("Ouro requires trust_remote_code=True")

        self.model_name = model_name
        self.dtype = _resolve_dtype(dtype)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        kwargs = dict(model_kwargs or {})

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            revision=revision,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            cache_dir=cache_dir,
            revision=revision,
            **kwargs,
        )
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.set_loop_steps(total_ut_steps)
        if freeze_exit_gate:
            self.freeze_exit_gate()
        self.print_runtime_state()

    def _exit_gate(self) -> nn.Module:
        decoder = getattr(self.model, "model", None)
        gate = getattr(decoder, "early_exit_gate", None)
        if not isinstance(gate, nn.Module):
            raise RuntimeError(
                "Official Ouro exit gate was not found at model.early_exit_gate; "
                "the remote implementation may have changed."
            )
        if not list(gate.parameters()):
            raise RuntimeError("Ouro model.early_exit_gate exists but has no parameters")
        return gate

    def set_loop_steps(self, steps: int) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        decoder = getattr(self.model, "model", None)
        if decoder is None or not hasattr(decoder, "total_ut_steps"):
            raise RuntimeError("Official Ouro decoder does not expose total_ut_steps")

        self.model.config.total_ut_steps = steps
        decoder.config.total_ut_steps = steps
        decoder.total_ut_steps = steps
        if self.total_ut_steps != steps:
            raise RuntimeError(f"Failed to set total_ut_steps to {steps}")
        print(f"[OuroModel] total_ut_steps={self.total_ut_steps}")

    @property
    def total_ut_steps(self) -> int:
        decoder = getattr(self.model, "model", None)
        value = getattr(decoder, "total_ut_steps", None)
        if value is None:
            raise RuntimeError("Official Ouro decoder does not expose total_ut_steps")
        return int(value)

    def freeze_exit_gate(self) -> None:
        gate = self._exit_gate()
        for parameter in gate.parameters():
            parameter.requires_grad_(False)
        states = self.exit_gate_parameter_states()
        if not states or any(states.values()):
            raise RuntimeError(f"Failed to freeze all exit gate parameters: {states}")
        print(f"[OuroModel] exit_gate_requires_grad={states}")

    def exit_gate_parameter_states(self) -> dict[str, bool]:
        gate = self._exit_gate()
        return {
            f"model.early_exit_gate.{name}": parameter.requires_grad
            for name, parameter in gate.named_parameters()
        }

    def exit_gate_is_frozen(self) -> bool:
        states = self.exit_gate_parameter_states()
        return bool(states) and not any(states.values())

    def print_runtime_state(self) -> None:
        print(f"[OuroModel] current total_ut_steps={self.total_ut_steps}")
        print(f"[OuroModel] exit gate parameters={self.exit_gate_parameter_states()}")

    def generate(self, **generation_inputs: Any) -> torch.Tensor:
        with torch.inference_mode():
            return self.model.generate(**generation_inputs)
