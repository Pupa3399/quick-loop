from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ouro_search.verl.model_hooks import (
    capture_ouro_update_evidence,
    freeze_ouro_exit_gate,
    ouro_parameter_was_updated,
    prepare_ouro_hf_config,
)


def test_prepare_ouro_hf_config_restores_logical_layers() -> None:
    config = SimpleNamespace(
        model_type="ouro",
        total_ut_steps=3,
        ouro_num_hidden_layers=48,
        num_hidden_layers=144,
        layer_types=["full_attention"] * 144,
    )

    assert prepare_ouro_hf_config(config)
    assert config.num_hidden_layers == 48
    assert len(config.layer_types) == 48


def test_prepare_ouro_hf_config_rejects_non_r3() -> None:
    config = SimpleNamespace(
        model_type="ouro", total_ut_steps=4, num_hidden_layers=48
    )
    with pytest.raises(RuntimeError, match="fixed to Ouro R3"):
        prepare_ouro_hf_config(config)


def test_freeze_ouro_exit_gate() -> None:
    model = nn.Module()
    model.model = nn.Module()
    model.model.early_exit_gate = nn.Linear(4, 1)

    states = freeze_ouro_exit_gate(model)

    assert states
    assert not any(states.values())


def test_ouro_update_evidence_checks_gradients_and_parameter_change() -> None:
    model = nn.Module()
    model.model = nn.Module()
    model.model.trunk = nn.Linear(4, 4)
    model.model.early_exit_gate = nn.Linear(4, 1)
    freeze_ouro_exit_gate(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    model.model.trunk(torch.ones(1, 4)).sum().backward()
    evidence = capture_ouro_update_evidence(model)
    optimizer.step()

    assert evidence is not None
    assert evidence.trunk_grad_nonzero
    assert ouro_parameter_was_updated(evidence)
    assert all(parameter.grad is None for parameter in model.model.early_exit_gate.parameters())


def test_ouro_update_evidence_rejects_gate_gradient() -> None:
    model = nn.Module()
    model.model = nn.Module()
    model.model.trunk = nn.Linear(4, 4)
    model.model.early_exit_gate = nn.Linear(4, 1)
    model.model.early_exit_gate.weight.grad = torch.ones_like(
        model.model.early_exit_gate.weight
    )

    with pytest.raises(RuntimeError, match="exit gate changed during backward"):
        capture_ouro_update_evidence(model)
