from __future__ import annotations

from typing import Any

from torch import nn


def prepare_ouro_hf_config(config: Any) -> bool:
    """Restore logical layers before the recurrent Ouro HF model is constructed."""
    if getattr(config, "model_type", None) != "ouro":
        return False

    steps = int(getattr(config, "total_ut_steps", 0))
    logical_layers = int(
        getattr(config, "ouro_num_hidden_layers", config.num_hidden_layers)
    )
    effective_layers = int(config.num_hidden_layers)
    if steps != 3:
        raise RuntimeError(f"GRPO is fixed to Ouro R3, got total_ut_steps={steps}")
    if effective_layers not in {logical_layers, logical_layers * steps}:
        raise RuntimeError(
            "Ouro layer metadata is inconsistent: "
            f"logical={logical_layers}, effective={effective_layers}, steps={steps}"
        )

    config.ouro_num_hidden_layers = logical_layers
    config.num_hidden_layers = logical_layers
    layer_types = getattr(config, "layer_types", None)
    if isinstance(layer_types, list) and len(layer_types) != logical_layers:
        config.layer_types = layer_types[:logical_layers]
    return True


def freeze_ouro_exit_gate(model: nn.Module) -> dict[str, bool]:
    """Freeze and verify the real exit gate exposed by official Ouro code."""
    decoder = getattr(model, "model", None)
    gate = getattr(decoder, "early_exit_gate", None)
    if not isinstance(gate, nn.Module):
        raise RuntimeError("Ouro exit gate was not found at model.early_exit_gate")
    parameters = list(gate.named_parameters())
    if not parameters:
        raise RuntimeError("Ouro exit gate exists but has no parameters")
    for _, parameter in parameters:
        parameter.requires_grad_(False)
    states = {
        f"model.early_exit_gate.{name}": parameter.requires_grad
        for name, parameter in parameters
    }
    if any(states.values()):
        raise RuntimeError(f"Failed to freeze Ouro exit gate: {states}")
    return states
