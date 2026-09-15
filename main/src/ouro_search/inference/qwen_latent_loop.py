from __future__ import annotations

import json
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

LATEST_DEPTH_KV = "latest_depth_kv"


@dataclass(slots=True)
class SoftFeedback:
    """Top-k distribution and its projection through Qwen's input embedding table."""

    embedding: Tensor
    token_ids: Tensor
    probabilities: Tensor


@dataclass(slots=True)
class LatentLoopTokenRecord:
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
    latency_seconds: float


@dataclass(slots=True)
class LatentLoopGeneration:
    prompt: str | None
    prompt_token_ids: list[int]
    token_ids: list[int]
    text: str
    cache_mode: str = LATEST_DEPTH_KV
    records: list[LatentLoopTokenRecord] = field(default_factory=list)
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
class LatentLoopState:
    """Mutable single-sequence generation state.

    The cache contains positions strictly before ``logical_position``. ``current_input_ids``
    is processed at that position and predicts the next visible token.
    """

    past_key_values: DynamicCache
    current_input_ids: Tensor
    logical_position: int
    prompt_token_ids: list[int]
    generated_token_ids: list[int] = field(default_factory=list)


class QwenLatentLoopEngine:
    """Training-free latent-depth greedy decoding for Hugging Face Qwen2.5 models."""

    cache_mode = LATEST_DEPTH_KV
    feedback_top_k = 100
    max_supported_depth = 2

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        *,
        feedback_top_k: int = 100,
    ) -> None:
        if getattr(model.config, "model_type", None) != "qwen2":
            raise ValueError("QwenLatentLoopEngine requires a Qwen2/Qwen2.5 causal LM")
        if feedback_top_k != self.feedback_top_k:
            raise ValueError("The reference backend fixes feedback_top_k at 100")

        decoder = getattr(model, "model", None)
        embed_tokens = getattr(decoder, "embed_tokens", None)
        if not isinstance(embed_tokens, nn.Embedding):
            raise RuntimeError("Qwen input embedding was not found at model.embed_tokens")
        if model.get_input_embeddings() is not embed_tokens:
            raise RuntimeError("model.embed_tokens is not the model input embedding module")

        self.model = model.eval()
        self.tokenizer = tokenizer
        self.embed_tokens = embed_tokens
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
    ) -> QwenLatentLoopEngine:
        """Load an unmodified Qwen causal LM with Hugging Face SDPA."""
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
                f"Qwen latent-loop reference backend requires SDPA, got {attention_backend!r}"
            )
        target_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model.to(target_device)
        return cls(model, tokenizer)

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

    def build_soft_feedback(self, logits: Tensor) -> SoftFeedback:
        """Project the FP32 top-100 distribution through ``embed_tokens.weight``."""
        if logits.ndim != 2:
            raise ValueError("logits must have shape [batch, vocabulary]")
        vocabulary_size = logits.shape[-1]
        if vocabulary_size != self.embed_tokens.num_embeddings:
            raise ValueError("logit vocabulary does not match model.embed_tokens")

        k = min(self.feedback_top_k, vocabulary_size)
        top_logits, top_token_ids = torch.topk(logits, k=k, dim=-1)
        probabilities = torch.softmax(top_logits.float(), dim=-1)
        embedding_rows = self.embed_tokens.weight[top_token_ids].float()
        soft_embedding = torch.sum(probabilities.unsqueeze(-1) * embedding_rows, dim=-2)
        return SoftFeedback(
            embedding=soft_embedding.to(dtype=self.dtype),
            token_ids=top_token_ids,
            probabilities=probabilities,
        )

    @torch.inference_mode()
    def prefill(self, input_ids: Tensor) -> LatentLoopState:
        """Cache the prompt except for its final token, which starts generation depth R1."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("the reference backend currently supports batch size 1")
        if input_ids.shape[1] < 1:
            raise ValueError("input_ids cannot be empty")

        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        prompt_length = input_ids.shape[1]
        cache = DynamicCache()
        if prompt_length > 1:
            prefix_length = prompt_length - 1
            position_ids = torch.arange(prefix_length, device=self.device).unsqueeze(0)
            cache_position = torch.arange(prefix_length, device=self.device)
            outputs = self.model(
                input_ids=input_ids[:, :-1],
                attention_mask=torch.ones_like(input_ids[:, :-1]),
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            cache = outputs.past_key_values

        state = LatentLoopState(
            past_key_values=cache,
            current_input_ids=input_ids[:, -1:],
            logical_position=prompt_length - 1,
            prompt_token_ids=input_ids[0].tolist(),
        )
        self._assert_state_invariant(state)
        return state

    def _assert_state_invariant(self, state: LatentLoopState) -> None:
        cached_positions = state.past_key_values.get_seq_length()
        if cached_positions != state.logical_position:
            raise RuntimeError(
                "latest_depth_kv invariant failed: "
                f"cache length {cached_positions} != logical position {state.logical_position}"
            )
        if state.current_input_ids.shape != (1, 1):
            raise RuntimeError("current_input_ids must have shape [1, 1]")

    @torch.inference_mode()
    def append_visible_tokens(self, state: LatentLoopState, token_ids: Tensor) -> None:
        """Append external visible tokens without rebuilding or discarding latent-loop KV.

        ``state.current_input_ids`` is the last model-generated token and has not yet been
        cached. It is cached together with all appended tokens except the final one. The final
        appended token becomes the next generation input. Existing highest-depth cache entries
        are preserved unchanged.
        """
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
        outputs = self.model(
            input_ids=tokens_to_cache,
            attention_mask=torch.ones((1, end), dtype=torch.long, device=self.device),
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=state.past_key_values,
            use_cache=True,
            logits_to_keep=1,
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
        state: LatentLoopState,
        *,
        depth: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> LatentLoopTokenRecord:
        """Generate one token and retain only its producer position's highest-depth KV."""
        self._validate_depth(depth)
        self._validate_sampling(temperature, top_p)
        self._assert_state_invariant(state)
        self._synchronize()
        started = time.perf_counter()

        position = state.logical_position
        position_ids = torch.tensor([[position]], dtype=torch.long, device=self.device)
        cache_position = torch.tensor([position], dtype=torch.long, device=self.device)
        attention_mask = torch.ones((1, position + 1), dtype=torch.long, device=self.device)
        depth_logits: list[Tensor] = []
        next_inputs_embeds: Tensor | None = None

        for depth_index in range(depth):
            if depth_index > 0:
                # R1/R(previous depth) appended this position; remove it before replacement.
                state.past_key_values.crop(-1)

            model_inputs: dict[str, Tensor] = {}
            if depth_index == 0:
                model_inputs["input_ids"] = state.current_input_ids
            else:
                assert next_inputs_embeds is not None
                model_inputs["inputs_embeds"] = next_inputs_embeds

            outputs = self.model(
                **model_inputs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=state.past_key_values,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            state.past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            depth_logits.append(logits)
            if depth_index + 1 < depth:
                feedback = self.build_soft_feedback(logits)
                next_inputs_embeds = feedback.embedding.unsqueeze(1)

        # There is deliberately no z1 + z2 residual: the deepest logits win directly.
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

        state.generated_token_ids.append(final_token_id)
        state.current_input_ids = final_token.view(1, 1)
        state.logical_position += 1
        self._assert_state_invariant(state)
        self._synchronize()
        latency_seconds = time.perf_counter() - started

        return LatentLoopTokenRecord(
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
            latency_seconds=latency_seconds,
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
    ) -> LatentLoopGeneration:
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
        records: list[LatentLoopTokenRecord] = []
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
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return LatentLoopGeneration(
            prompt=prompt,
            prompt_token_ids=list(state.prompt_token_ids),
            token_ids=generated_ids,
            text=text,
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
    ) -> LatentLoopGeneration:
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
