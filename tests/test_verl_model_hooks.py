from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch import nn

from ouro_search.verl.model_hooks import freeze_ouro_exit_gate, prepare_ouro_hf_config


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
