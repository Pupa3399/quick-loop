from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(slots=True)
class OuroUpdateEvidence:
    parameter_name: str | None
    parameter: nn.Parameter | None
    value_before_step: torch.Tensor | None
    trunk_grad_nonzero: bool


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


def capture_ouro_update_evidence(model: nn.Module) -> OuroUpdateEvidence | None:
    """Validate the frozen gate and snapshot a small parameter with a nonzero gradient."""
    named_parameters = list(model.named_parameters())
    gate_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if "early_exit_gate." in name
    ]
    if not gate_parameters:
        return None
    invalid_gate = {
        name: {
            "requires_grad": parameter.requires_grad,
            "grad_is_none": parameter.grad is None,
        }
        for name, parameter in gate_parameters
        if parameter.requires_grad or parameter.grad is not None
    }
    if invalid_gate:
        raise RuntimeError(f"Ouro exit gate changed during backward: {invalid_gate}")

    candidates = sorted(
        (
            (name, parameter)
            for name, parameter in named_parameters
            if "early_exit_gate." not in name
            and parameter.grad is not None
            and parameter.numel() > 0
        ),
        key=lambda item: item[1].numel(),
    )
    for name, parameter in candidates:
        if torch.count_nonzero(parameter.grad).item() > 0:
            return OuroUpdateEvidence(
                parameter_name=name,
                parameter=parameter,
                value_before_step=parameter.detach().clone(),
                trunk_grad_nonzero=True,
            )
    return OuroUpdateEvidence(
        parameter_name=None,
        parameter=None,
        value_before_step=None,
        trunk_grad_nonzero=False,
    )


def ouro_parameter_was_updated(evidence: OuroUpdateEvidence | None) -> bool:
    if evidence is None or evidence.parameter is None:
        return False
    assert evidence.value_before_step is not None
    return not torch.equal(evidence.value_before_step, evidence.parameter.detach())
