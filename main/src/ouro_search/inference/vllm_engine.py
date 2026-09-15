from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ouro_search.agent.profiles import AgentPrompt
from ouro_search.agent.rendering import render_agent_prompt
from ouro_search.inference.types import GenerationResult


class VllmEngine:
    """Synchronous vLLM backend implementing the shared AgentRunner interface."""

    def __init__(
        self,
        model: str | Path,
        *,
        dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        max_model_len: int = 7096,
        gpu_memory_utilization: float = 0.4,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
        use_chat_template: bool = True,
        fixed_loop_steps: int | None = None,
    ) -> None:
        from vllm import LLM

        self.use_chat_template = use_chat_template
        self.fixed_loop_steps = fixed_loop_steps
        self.max_model_len = max_model_len
        self.llm = LLM(
            model=str(model),
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def encode_text(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _format_prompt(self, prompt: str | AgentPrompt) -> str:
        return render_agent_prompt(
            self.tokenizer,
            prompt,
            use_chat_template=self.use_chat_template,
        )

    def render_prompt(self, prompt: str | AgentPrompt) -> str:
        return self._format_prompt(prompt)

    def available_generation_tokens(self, prompt: str | AgentPrompt) -> int:
        rendered = self._format_prompt(prompt)
        return self.max_model_len - len(self.encode_text(rendered))

    def char_span_to_token_span(self, text: str, span: tuple[int, int]) -> list[int] | None:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            return None
        overlapping = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > span[0] and start < span[1]
        ]
        if not overlapping:
            return None
        return [overlapping[0], overlapping[-1] + 1]

    def generate_detailed(
        self,
        prompt: str | AgentPrompt,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        loop_steps: int = 1,
        seed: int | None = None,
    ) -> GenerationResult:
        if self.fixed_loop_steps is not None and loop_steps != self.fixed_loop_steps:
            raise ValueError(
                f"This Ouro engine is fixed at R{self.fixed_loop_steps}, got R{loop_steps}"
            )
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        rendered_prompt = self._format_prompt(prompt)
        prompt_token_ids = self.encode_text(rendered_prompt)
        if len(prompt_token_ids) + max_new_tokens > self.max_model_len:
            raise ValueError(
                "Requested generation exceeds model context: "
                f"prompt={len(prompt_token_ids)} generation={max_new_tokens} "
                f"max_model_len={self.max_model_len}"
            )
        stop_sequences = list(prompt.stop_sequences) if isinstance(prompt, AgentPrompt) else []
        sampling_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "stop": stop_sequences,
            "include_stop_str_in_output": True,
            "logprobs": 5,
        }
        if seed is not None:
            sampling_kwargs["seed"] = seed
        params = SamplingParams(**sampling_kwargs)
        started = time.perf_counter()
        tokenized_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
        request_output = self.llm.generate([tokenized_prompt], params, use_tqdm=False)[0]
        latency_seconds = time.perf_counter() - started
        output = request_output.outputs[0]
        completion_token_ids = list(output.token_ids)
        chosen_logprobs: list[float | None] = []
        top_logprobs: list[list[dict[str, Any]]] = []
        for token_id, candidates in zip(
            completion_token_ids, output.logprobs or [], strict=False
        ):
            chosen = candidates.get(token_id)
            chosen_logprobs.append(float(chosen.logprob) if chosen is not None else None)
            ranked = sorted(
                candidates.items(),
                key=lambda item: (
                    item[1].rank if item[1].rank is not None else 1_000_000,
                    -item[1].logprob,
                ),
            )[:5]
            top_logprobs.append(
                [
                    {
                        "token_id": int(candidate_id),
                        "token": candidate.decoded_token,
                        "logprob": float(candidate.logprob),
                        "rank": candidate.rank,
                    }
                    for candidate_id, candidate in ranked
                ]
            )
        if len(chosen_logprobs) < len(completion_token_ids):
            missing = len(completion_token_ids) - len(chosen_logprobs)
            chosen_logprobs.extend([None] * missing)
            top_logprobs.extend([[] for _ in range(missing)])
        stop_reason = output.stop_reason
        if not isinstance(stop_reason, (str, int, type(None))):
            stop_reason = str(stop_reason)
        return GenerationResult(
            text=output.text,
            rendered_prompt=rendered_prompt,
            prompt_token_ids=list(request_output.prompt_token_ids or prompt_token_ids),
            completion_token_ids=completion_token_ids,
            finish_reason=output.finish_reason,
            stop_reason=stop_reason,
            latency_seconds=latency_seconds,
            chosen_token_logprobs=chosen_logprobs,
            token_top_logprobs=top_logprobs,
        )

    def generate(
        self,
        prompt: str | AgentPrompt,
        *,
        max_new_tokens: int = 500,
        temperature: float = 0.0,
        top_p: float = 1.0,
        loop_steps: int = 1,
    ) -> str:
        return self.generate_detailed(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            loop_steps=loop_steps,
        ).text
