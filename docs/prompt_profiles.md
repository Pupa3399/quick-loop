# Agent Prompt Profiles

`AgentRunner` owns the retrieval state machine, token budgets, 0-based turn schedule,
and trajectory records. A `PromptProfile` owns only protocol-specific behavior:

- system and user prompt construction;
- the Search function schema;
- complete model-context construction;
- Retriever observation formatting;
- action and final-answer parsing.

`AgentPrompt` carries structured chat messages, any raw assistant-stream continuation,
and profile-specific stop sequences. This distinction is intentional: Search-R1 appends
retrieval results to one assistant generation stream, while Hermes reapplies a ChatML
template to a conversation containing `assistant` and `tool` messages.

Choose the Hydra config group with `prompt_profile=search_r1` (default) or
`prompt_profile=hermes`. New protocols can implement `PromptProfile` and add one entry
to the registry without copying the Agent loop.

## Search-R1

The implementation follows the official
[Search-R1 inference code](https://github.com/PeterGriffinJin/Search-R1/blob/main/infer.py):
the original user instruction is preserved, model actions use `<think>`, `<search>`,
and `<answer>`, and each result is appended as `<information>...documents...</information>`
in the same generation stream. The official `Doc N(Title: ...) ...` passage rendering
is also preserved. The configured four-search safety limit remains outside the prompt.

## Hermes

The implementation follows Nous Research's
[Hermes function-calling prompt](https://github.com/NousResearch/Hermes-Function-Calling/blob/main/prompt_assets/sys_prompt.yml),
[parser and recursive loop](https://github.com/NousResearch/Hermes-Function-Calling/blob/main/functioncall.py),
and the official
[Hermes 2 Pro tool-use message example](https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B#prompt-format-for-function-calling).
The system message exposes an OpenAI-style Search function schema inside `<tools>`.
An assistant action is strict JSON inside `<tool_call>`, and the Retriever result is
JSON inside `<tool_response>` in a `tool` role message. A final answer is ordinary
assistant text rather than an `<answer>` block. Invalid XML, invalid JSON, multiple
calls, unknown tools, and missing/empty `query` arguments are treated as malformed.

Run the offline protocol smoke with:

```bash
uv run python scripts/smoke_test_prompt_profiles.py
```
