from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

HIDDEN_LATEST_DEPTH_KV = "latest_depth_kv"


@dataclass(slots=True)
class HiddenLoopTokenRecord:
    token_id: int
    token_text: str
    depth: int
    r1_top1_token_id: int
    r1_top1_text: str
    r2_top1_token_id: int | None
    r2_top1_text: str | None
    final_output_token_id: int
    final_output_text: str
    logical_position: int
    output_position: int
    position_ids_by_depth: list[int]
    logits_max_abs_difference: float | None
    reference_rms: float | None
    hidden_rms: float | None
    latency_seconds: float


@dataclass(slots=True)
class HiddenLoopGeneration:
    prompt: str | None
    prompt_token_ids: list[int]
    token_ids: list[int]
    text: str
    alpha: float
    cache_mode: str = HIDDEN_LATEST_DEPTH_KV
    records: list[HiddenLoopTokenRecord] = field(default_factory=list)
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)


@dataclass(slots=True)
class HiddenLoopState:
    """Single-sequence state whose cache ends before ``logical_position``."""

    past_key_values: DynamicCache
    current_input_ids: Tensor
    logical_position: int
    prompt_token_ids: list[int]
    generated_token_ids: list[int] = field(default_factory=list)


class QwenHiddenLoopEngine:
    """Training-free hidden-state residual depth loop for Hugging Face Qwen2.5."""

    cache_mode = HIDDEN_LATEST_DEPTH_KV
    max_supported_depth = 2

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        alpha: float = 0.5,
        rms_epsilon: float = 1e-6,
    ) -> None:
        if getattr(model.config, "model_type", None) != "qwen2":
            raise ValueError("QwenHiddenLoopEngine requires a Qwen2/Qwen2.5 causal LM")
        if not math.isfinite(alpha):
            raise ValueError("alpha must be finite")
        if not math.isfinite(rms_epsilon) or rms_epsilon <= 0:
            raise ValueError("rms_epsilon must be finite and positive")

        decoder = getattr(model, "model", None)
        embed_tokens = getattr(decoder, "embed_tokens", None)
        lm_head = getattr(model, "lm_head", None)
        if not isinstance(decoder, nn.Module):
            raise RuntimeError("Qwen decoder backbone was not found at model.model")
        if not isinstance(embed_tokens, nn.Embedding):
            raise RuntimeError("Qwen input embedding was not found at model.embed_tokens")
        if not isinstance(lm_head, nn.Module):
            raise RuntimeError("Qwen LM head was not found at model.lm_head")
        if model.get_input_embeddings() is not embed_tokens:
            raise RuntimeError("model.embed_tokens is not the model input embedding module")

        self.model = model.eval()
        self.decoder = decoder
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.alpha = float(alpha)
        self.rms_epsilon = float(rms_epsilon)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        revision: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
        local_files_only: bool = True,
        alpha: float = 0.5,
        rms_epsilon: float = 1e-6,
    ) -> QwenHiddenLoopEngine:
        source = str(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            source,
            revision=revision,
            local_files_only=local_files_only,
            trust_remote_code=False,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        attention_backend = getattr(model.config, "_attn_implementation", None)
        if attention_backend != "sdpa":
            raise RuntimeError(
                f"Qwen hidden-loop reference backend requires SDPA, got {attention_backend!r}"
            )
        target_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model.to(target_device)
        return cls(
            model,
            tokenizer,
            alpha=alpha,
            rms_epsilon=rms_epsilon,
        )

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    def _decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _validate_depth(self, depth: int) -> None:
        if not isinstance(depth, int) or isinstance(depth, bool):
            raise TypeError("depth must be an integer")
        if not 1 <= depth <= self.max_supported_depth:
            raise ValueError(
                f"depth must be between 1 and {self.max_supported_depth} for this backend"
            )

    @staticmethod
    def _validate_sampling(temperature: float, top_p: float) -> None:
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

    @staticmethod
    def _select_token(
        logits: Tensor,
        *,
        temperature: float,
        top_p: float,
        generator: torch.Generator | None,
    ) -> Tensor:
        if temperature == 0:
            return torch.argmax(logits, dim=-1)
        scaled_logits = logits.float() / temperature
        sorted_logits, sorted_indices = torch.sort(scaled_logits, dim=-1, descending=True)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative_probabilities - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
        sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
        sampled_rank = torch.multinomial(
            sorted_probabilities,
            num_samples=1,
            generator=generator,
        )
        return sorted_indices.gather(-1, sampled_rank).squeeze(-1)

    def rms_scale(self, hidden_state: Tensor, *, reference: Tensor) -> Tensor:
        """Match hidden-state RMS to the reference embedding, with FP32 arithmetic."""
        if hidden_state.shape != reference.shape:
            raise ValueError("hidden_state and reference must have identical shapes")
        hidden_fp32 = hidden_state.float()
        reference_fp32 = reference.float()
        hidden_rms = torch.sqrt(torch.mean(hidden_fp32.square(), dim=-1, keepdim=True))
        reference_rms = torch.sqrt(
            torch.mean(reference_fp32.square(), dim=-1, keepdim=True)
        )
        scaled = hidden_fp32 * reference_rms / (hidden_rms + self.rms_epsilon)
        return scaled.to(dtype=reference.dtype)

    def residual_input(self, hidden_state: Tensor, *, reference: Tensor) -> Tensor:
        scaled_hidden = self.rms_scale(hidden_state, reference=reference)
        return reference + self.alpha * scaled_hidden

    @torch.inference_mode()
    def prefill(self, input_ids: Tensor) -> HiddenLoopState:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("the reference backend currently supports batch size 1")
        if input_ids.shape[1] < 1:
            raise ValueError("input_ids cannot be empty")

        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        prompt_length = input_ids.shape[1]
        cache = DynamicCache()
        if prompt_length > 1:
            prefix_length = prompt_length - 1
            positions = torch.arange(prefix_length, device=self.device)
            outputs = self.decoder(
                input_ids=input_ids[:, :-1],
                attention_mask=torch.ones_like(input_ids[:, :-1]),
                position_ids=positions.unsqueeze(0),
                cache_position=positions,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values

        state = HiddenLoopState(
            past_key_values=cache,
            current_input_ids=input_ids[:, -1:],
            logical_position=prompt_length - 1,
            prompt_token_ids=input_ids[0].tolist(),
        )
        self._assert_state_invariant(state)
        return state

    def _assert_state_invariant(self, state: HiddenLoopState) -> None:
        cached_positions = state.past_key_values.get_seq_length()
        if cached_positions != state.logical_position:
            raise RuntimeError(
                "latest_depth_kv invariant failed: "
                f"cache length {cached_positions} != logical position {state.logical_position}"
            )
        if state.current_input_ids.shape != (1, 1):
            raise RuntimeError("current_input_ids must have shape [1, 1]")

    @torch.inference_mode()
    def append_visible_tokens(self, state: HiddenLoopState, token_ids: Tensor) -> None:
        self._assert_state_invariant(state)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] < 1:
            raise ValueError("token_ids must have shape [1, sequence] with sequence >= 1")
        token_ids = token_ids.to(device=self.device, dtype=torch.long)

        appended_length = token_ids.shape[1]
        tokens_to_cache = torch.cat((state.current_input_ids, token_ids[:, :-1]), dim=1)
        start = state.logical_position
        end = start + appended_length
        positions = torch.arange(start, end, device=self.device)
        outputs = self.decoder(
            input_ids=tokens_to_cache,
            attention_mask=torch.ones((1, end), dtype=torch.long, device=self.device),
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=state.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        state.past_key_values = outputs.past_key_values
        state.current_input_ids = token_ids[:, -1:]
        state.logical_position = end
        self._assert_state_invariant(state)

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def step(
        self,
        state: HiddenLoopState,
        *,
        depth: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> HiddenLoopTokenRecord:
        self._validate_depth(depth)
        self._validate_sampling(temperature, top_p)
        self._assert_state_invariant(state)
        self._synchronize()
        started = time.perf_counter()

        position = state.logical_position
        position_ids = torch.tensor([[position]], dtype=torch.long, device=self.device)
        cache_position = torch.tensor([position], dtype=torch.long, device=self.device)
        attention_mask = torch.ones((1, position + 1), dtype=torch.long, device=self.device)
        reference_embedding = self.embed_tokens(state.current_input_ids)
        depth_logits: list[Tensor] = []
        depth_hidden_states: list[Tensor] = []
        depth_input = reference_embedding

        for depth_index in range(depth):
            if depth_index > 0:
                state.past_key_values.crop(-1)

            outputs = self.decoder(
                inputs_embeds=depth_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=state.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            state.past_key_values = outputs.past_key_values
            hidden_state = outputs.last_hidden_state[:, -1:, :]
            logits = self.lm_head(hidden_state[:, -1, :])
            depth_hidden_states.append(hidden_state)
            depth_logits.append(logits)
            if depth_index + 1 < depth:
                depth_input = self.residual_input(
                    hidden_state,
                    reference=reference_embedding,
                )

        final_token = self._select_token(
            depth_logits[-1],
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        r1_top1 = int(torch.argmax(depth_logits[0], dim=-1).item())
        r2_top1 = (
            int(torch.argmax(depth_logits[1], dim=-1).item()) if len(depth_logits) >= 2 else None
        )
        final_token_id = int(final_token.item())
        logits_delta = (
            float((depth_logits[1].float() - depth_logits[0].float()).abs().max().item())
            if len(depth_logits) >= 2
            else None
        )
        reference_rms = None
        hidden_rms = None
        if len(depth_hidden_states) >= 2:
            reference_rms = float(
                torch.sqrt(torch.mean(reference_embedding.float().square())).item()
            )
            hidden_rms = float(
                torch.sqrt(torch.mean(depth_hidden_states[0].float().square())).item()
            )

        state.generated_token_ids.append(final_token_id)
        state.current_input_ids = final_token.view(1, 1)
        state.logical_position += 1
        self._assert_state_invariant(state)
        self._synchronize()

        return HiddenLoopTokenRecord(
            token_id=final_token_id,
            token_text=self._decode_token(final_token_id),
            depth=depth,
            r1_top1_token_id=r1_top1,
            r1_top1_text=self._decode_token(r1_top1),
            r2_top1_token_id=r2_top1,
            r2_top1_text=self._decode_token(r2_top1) if r2_top1 is not None else None,
            final_output_token_id=final_token_id,
            final_output_text=self._decode_token(final_token_id),
            logical_position=position,
            output_position=position + 1,
            position_ids_by_depth=[position] * depth,
            logits_max_abs_difference=logits_delta,
            reference_rms=reference_rms,
            hidden_rms=hidden_rms,
            latency_seconds=time.perf_counter() - started,
        )

    def _depth_for_token(
        self,
        token_index: int,
        *,
        depth: int,
        depth_by_token: Mapping[int, int] | None,
    ) -> int:
        selected = depth_by_token.get(token_index, depth) if depth_by_token else depth
        self._validate_depth(selected)
        return selected

    @torch.inference_mode()
    def generate_ids(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        depth: int = 1,
        depth_by_token: Mapping[int, int] | None = None,
        eos_token_id: int | None = None,
        prompt: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        stop_sequences: tuple[str, ...] = (),
    ) -> HiddenLoopGeneration:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self._validate_depth(depth)
        self._validate_sampling(temperature, top_p)
        if depth_by_token:
            invalid_indices = [index for index in depth_by_token if index < 0]
            if invalid_indices:
                raise ValueError("depth_by_token keys must be non-negative 0-based indices")
            for selected_depth in depth_by_token.values():
                self._validate_depth(selected_depth)

        state = self.prefill(input_ids)
        generator = None
        if temperature > 0 and seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        records: list[HiddenLoopTokenRecord] = []
        stop_token_id = self.tokenizer.eos_token_id if eos_token_id is None else eos_token_id
        stop_reason: str | None = None
        for token_index in range(max_new_tokens):
            selected_depth = self._depth_for_token(
                token_index,
                depth=depth,
                depth_by_token=depth_by_token,
            )
            record = self.step(
                state,
                depth=selected_depth,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            records.append(record)
            if stop_token_id is not None and record.token_id == stop_token_id:
                stop_reason = "eos"
                break
            if stop_sequences:
                current_text = self.tokenizer.decode(
                    state.generated_token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                matched_stop = next(
                    (value for value in stop_sequences if value in current_text),
                    None,
                )
                if matched_stop is not None:
                    stop_reason = matched_stop
                    break

        generated_ids = list(state.generated_token_ids)
        text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return HiddenLoopGeneration(
            prompt=prompt,
            prompt_token_ids=list(state.prompt_token_ids),
            token_ids=generated_ids,
            text=text,
            alpha=self.alpha,
            records=records,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop_reason=stop_reason,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        depth: int = 1,
        depth_by_token: Mapping[int, int] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        stop_sequences: tuple[str, ...] = (),
    ) -> HiddenLoopGeneration:
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return self.generate_ids(
            encoded["input_ids"],
            max_new_tokens=max_new_tokens,
            depth=depth,
            depth_by_token=depth_by_token,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            stop_sequences=stop_sequences,
        )
