from __future__ import annotations

from pathlib import Path
from typing import Any

from ouro_search.agent.profiles import AgentPrompt
from ouro_search.agent.rendering import render_agent_prompt


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

    def _format_prompt(self, prompt: str | AgentPrompt) -> str:
        return render_agent_prompt(
            self.tokenizer,
            prompt,
            use_chat_template=self.use_chat_template,
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

        stop_sequences = list(prompt.stop_sequences) if isinstance(prompt, AgentPrompt) else []
        sampling_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
            "stop": stop_sequences,
            "include_stop_str_in_output": True,
        }
        params = SamplingParams(**sampling_kwargs)
        result = self.llm.generate([self._format_prompt(prompt)], params, use_tqdm=False)[0]
        return result.outputs[0].text.strip()
