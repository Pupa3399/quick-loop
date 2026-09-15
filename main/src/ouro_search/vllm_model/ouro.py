from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.attention import Attention, AttentionType
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata


class OuroMLP(nn.Module):
    def __init__(
        self,
        config: Any,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if config.hidden_act != "silu":
            raise ValueError("Ouro vLLM plugin currently supports only silu")
        self.activation = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class OuroAttention(nn.Module):
    def __init__(
        self,
        config: Any,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        logical_prefix: str,
        cache_prefixes: list[str],
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{logical_prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{logical_prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        scaling = self.head_dim**-0.5
        self.cache_ops = nn.ModuleList(
            [
                Attention(
                    self.num_heads,
                    self.head_dim,
                    scaling,
                    num_kv_heads=self.num_kv_heads,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    attn_type=AttentionType.DECODER,
                    prefix=prefix,
                )
                for prefix in cache_prefixes
            ]
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, loop_step: int
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )
        query, key = self.rotary_emb(positions, query, key)
        output = self.cache_ops[loop_step](query, key, value)
        output, _ = self.o_proj(output)
        return output


class OuroDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Any,
        cache_config: CacheConfig | None,
        quant_config: QuantizationConfig | None,
        logical_layer: int,
        total_loop_steps: int,
        prefix: str,
    ) -> None:
        super().__init__()
        cache_prefixes = [
            (
                f"model.layers."
                f"{step * config.ouro_num_hidden_layers + logical_layer}.self_attn.attn"
            )
            for step in range(total_loop_steps)
        ]
        self.self_attn = OuroAttention(
            config,
            cache_config,
            quant_config,
            f"{prefix}.self_attn",
            cache_prefixes,
        )
        self.mlp = OuroMLP(config, quant_config, f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_2 = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor, loop_step: int
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, loop_step)
        hidden_states = self.input_layernorm_2(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm_2(hidden_states)
        return residual + hidden_states


class OuroModel(nn.Module):
    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.logical_layers = int(config.ouro_num_hidden_layers)
        self.total_loop_steps = int(config.total_ut_steps)
        if config.num_hidden_layers != self.logical_layers * self.total_loop_steps:
            raise ValueError("Prepared Ouro vLLM config has an invalid effective layer count")
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix="model.embed_tokens",
        )
        self.layers = nn.ModuleList(
            [
                OuroDecoderLayer(
                    config,
                    vllm_config.cache_config,
                    vllm_config.quant_config,
                    logical_layer=index,
                    total_loop_steps=self.total_loop_steps,
                    prefix=f"model.layers.{index}",
                )
                for index in range(self.logical_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.early_exit_gate = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for loop_step in range(self.total_loop_steps):
            for layer in self.layers:
                hidden_states = layer(positions, hidden_states, loop_step)
            hidden_states = self.norm(hidden_states)
        return hidden_states


class OuroForCausalLM(nn.Module):
    """vLLM R3 implementation with one KV cache slot per UT-step/layer pair."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        del prefix
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = OuroModel(vllm_config)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix="lm_head",
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None or inputs_embeds is not None:
            raise NotImplementedError("Ouro R3 plugin does not support pipeline parallelism")
        return self.model(input_ids, positions)

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        parameters = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        for source_name, weight in weights:
            name = source_name
            if "rotary_emb.inv_freq" in name:
                continue
            for packed_name, unpacked_name, shard_id in stacked:
                if unpacked_name not in name:
                    continue
                name = name.replace(unpacked_name, packed_name)
                parameter = parameters[name]
                parameter.weight_loader(parameter, weight, shard_id)
                break
            else:
                parameter = parameters[name]
                loader = getattr(parameter, "weight_loader", default_weight_loader)
                loader(parameter, weight)
            loaded.add(name)
        return loaded
