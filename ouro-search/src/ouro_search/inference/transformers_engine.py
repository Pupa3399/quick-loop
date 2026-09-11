from __future__ import annotations

from ouro_search.agent.profiles import AgentPrompt
from ouro_search.agent.rendering import render_agent_prompt
from ouro_search.models.ouro import OuroModel


def _stopping_criteria(tokenizer: object, stop_sequences: tuple[str, ...]) -> object | None:
    if not stop_sequences:
        return None

    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    targets = [
        torch.tensor(tokenizer.encode(sequence, add_special_tokens=False))
        for sequence in stop_sequences
    ]

    class StopOnTokenSequences(StoppingCriteria):
        def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            **kwargs: object,
        ) -> torch.BoolTensor:
            del scores, kwargs
            matches = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            for target in targets:
                target = target.to(input_ids.device)
                if input_ids.shape[1] >= target.numel():
                    matches |= (input_ids[:, -target.numel() :] == target).all(dim=1)
            return matches

    return StoppingCriteriaList([StopOnTokenSequences()])


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

    def _format_prompt(self, prompt: str | AgentPrompt) -> str:
        return render_agent_prompt(
            self.model.tokenizer,
            prompt,
            use_chat_template=self.use_chat_template,
        )

    def generate(
        self,
        prompt: str | AgentPrompt,
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
        if isinstance(prompt, AgentPrompt):
            criteria = _stopping_criteria(self.model.tokenizer, prompt.stop_sequences)
            if criteria is not None:
                generation_kwargs["stopping_criteria"] = criteria

        output_ids = self.model.generate(**generation_kwargs)
        prompt_length = encoded["input_ids"].shape[-1]
        completion_ids = output_ids[0, prompt_length:]
        return self.model.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
