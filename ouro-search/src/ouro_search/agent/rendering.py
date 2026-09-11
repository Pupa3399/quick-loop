from __future__ import annotations

from typing import Any

from ouro_search.agent.profiles import AgentPrompt


def render_agent_prompt(
    tokenizer: Any,
    prompt: str | AgentPrompt,
    *,
    use_chat_template: bool,
) -> str:
    if isinstance(prompt, str):
        return prompt
    messages = [dict(message) for message in prompt.messages]
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered + prompt.continuation
    if len(messages) == 1 and messages[0].get("role") == "user":
        return str(messages[0]["content"]) + prompt.continuation
    rendered = "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages
    )
    return rendered + "<|im_start|>assistant\n" + prompt.continuation
