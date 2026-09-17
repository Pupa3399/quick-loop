from __future__ import annotations

from typing import Any

import pytest
import torch
from ouro_search.inference.qwen_hidden_loop import QwenHiddenLoopEngine
from transformers import Qwen2Config, Qwen2ForCausalLM


class NumericTokenizer:
    eos_token_id = None

    @staticmethod
    def decode(token_ids: list[int], **_: Any) -> str:
        return "".join(f"<{token_id}>" for token_id in token_ids)


@pytest.fixture(scope="module")
def tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(config).eval()


@pytest.fixture()
def input_ids() -> torch.Tensor:
    return torch.tensor([[1, 5, 7, 9]], dtype=torch.long)


def cache_snapshot(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (layer.keys.detach().clone(), layer.values.detach().clone())
        for layer in cache.layers
    ]


def test_depth_one_matches_hugging_face_greedy(
    tiny_model: Qwen2ForCausalLM,
    input_ids: torch.Tensor,
) -> None:
    engine = QwenHiddenLoopEngine(tiny_model, NumericTokenizer())
    expected = tiny_model.generate(
        input_ids,
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=0,
    )[0, input_ids.shape[1] :].tolist()
    actual = engine.generate_ids(
        input_ids,
        max_new_tokens=5,
        depth=1,
        eos_token_id=-1,
    )
    assert actual.token_ids == expected


def test_alpha_zero_repeats_r1_logits(
    tiny_model: Qwen2ForCausalLM,
    input_ids: torch.Tensor,
) -> None:
    engine = QwenHiddenLoopEngine(tiny_model, NumericTokenizer(), alpha=0.0)
    record = engine.step(engine.prefill(input_ids), depth=2)
    assert record.r2_top1_token_id == record.r1_top1_token_id
    assert record.logits_max_abs_difference is not None
    assert record.logits_max_abs_difference <= 1e-6


def test_rms_scale_uses_fp32_formula(
    tiny_model: Qwen2ForCausalLM,
) -> None:
    engine = QwenHiddenLoopEngine(tiny_model, NumericTokenizer())
    hidden = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.bfloat16)
    reference = torch.tensor([[[0.25, 0.5, 0.75, 1.0]]], dtype=torch.bfloat16)
    actual = engine.rms_scale(hidden, reference=reference)
    hidden_fp32 = hidden.float()
    reference_fp32 = reference.float()
    expected = hidden_fp32 * torch.sqrt(reference_fp32.square().mean(dim=-1, keepdim=True))
    expected /= torch.sqrt(hidden_fp32.square().mean(dim=-1, keepdim=True)) + 1e-6
    assert actual.dtype == reference.dtype
    torch.testing.assert_close(actual, expected.to(reference.dtype))


def test_r2_reuses_position_and_replaces_current_kv(
    tiny_model: Qwen2ForCausalLM,
    input_ids: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = QwenHiddenLoopEngine(tiny_model, NumericTokenizer(), alpha=0.5)
    state = engine.prefill(input_ids)
    initial_cache_length = state.past_key_values.get_seq_length()
    forward_snapshots: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
    original_forward = engine.decoder.forward

    def capture_forward(*args: Any, **kwargs: Any) -> Any:
        output = original_forward(*args, **kwargs)
        forward_snapshots.append(cache_snapshot(output.past_key_values))
        return output

    monkeypatch.setattr(engine.decoder, "forward", capture_forward)
    record = engine.step(state, depth=2)

    assert record.position_ids_by_depth == [input_ids.shape[1] - 1] * 2
    assert state.past_key_values.get_seq_length() == initial_cache_length + 1
    assert len(forward_snapshots) == 2
    final_snapshot = cache_snapshot(state.past_key_values)
    differs_from_r1 = False
    for final_layer, r1_layer, r2_layer in zip(
        final_snapshot,
        forward_snapshots[0],
        forward_snapshots[1],
        strict=True,
    ):
        torch.testing.assert_close(final_layer[0], r2_layer[0])
        torch.testing.assert_close(final_layer[1], r2_layer[1])
        differs_from_r1 |= not torch.allclose(final_layer[0][..., -1, :], r1_layer[0][..., -1, :])
    assert differs_from_r1


def test_depth_by_token_is_zero_based(
    tiny_model: Qwen2ForCausalLM,
    input_ids: torch.Tensor,
) -> None:
    engine = QwenHiddenLoopEngine(tiny_model, NumericTokenizer())
    result = engine.generate_ids(
        input_ids,
        max_new_tokens=3,
        depth=1,
        depth_by_token={1: 2},
        eos_token_id=-1,
    )
    assert [record.depth for record in result.records] == [1, 2, 1]


def test_engine_adds_no_trainable_parameters(tiny_model: Qwen2ForCausalLM) -> None:
    original_parameter_ids = {id(parameter) for parameter in tiny_model.parameters()}
    engine = QwenHiddenLoopEngine(tiny_model, NumericTokenizer())
    assert {id(parameter) for parameter in engine.model.parameters()} == original_parameter_ids
    assert all(not parameter.requires_grad for parameter in engine.model.parameters())
    assert isinstance(engine.alpha, float)
