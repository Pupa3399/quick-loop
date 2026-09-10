from __future__ import annotations

from ouro_search.agent.protocol import split_agent_prompt
from ouro_search.models.ouro import OuroModel


class TransformersEngine:
    """Prompt-in, text-out generation interface for Ouro."""

    def __init__(
        self,
        model: OuroModel,
        *,
        use_chat_template: bool = True,
        use_cache: bool = False,
    ) -> None:
        self.model = model
        self.use_chat_template = use_chat_template
        self.use_cache = use_cache

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=False))

    def truncate_text(self, text: str, max_tokens: int) -> str:
        token_ids = self.model.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
        return self.model.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _format_prompt(self, prompt: str) -> str:
        initial_prompt, continuation = split_agent_prompt(prompt)
        tokenizer = self.model.tokenizer
        if not self.use_chat_template or not tokenizer.chat_template:
            return initial_prompt + continuation
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": initial_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered + continuation

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        greedy: bool | None = None,
        loop_steps: int | None = None,
    ) -> str:
        if loop_steps is not None:
            self.model.set_loop_steps(loop_steps)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")

        greedy = temperature == 0 if greedy is None else greedy
        if greedy and temperature > 0:
            raise ValueError("greedy=True is incompatible with temperature > 0")

        rendered_prompt = self._format_prompt(prompt)
        encoded = self.model.tokenizer(rendered_prompt, return_tensors="pt")
        encoded = {name: tensor.to(self.model.device) for name, tensor in encoded.items()}
        generation_kwargs: dict[str, object] = {
            **encoded,
            "max_new_tokens": max_new_tokens,
            "do_sample": not greedy,
            "use_cache": self.use_cache,
            "pad_token_id": self.model.tokenizer.pad_token_id,
            "eos_token_id": self.model.tokenizer.eos_token_id,
        }
        if not greedy:
            generation_kwargs.update(temperature=temperature, top_p=top_p)

        output_ids = self.model.generate(**generation_kwargs)
        prompt_length = encoded["input_ids"].shape[-1]
        completion_ids = output_ids[0, prompt_length:]
        return self.model.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
