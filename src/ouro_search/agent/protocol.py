from __future__ import annotations

from collections.abc import Sequence

from ouro_search.search.types import Document
from ouro_search.trajectory.schema import TurnRecord

SYSTEM_INSTRUCTION = """Answer the question by reasoning and, when necessary, searching.
Use exactly one of these action formats after your reasoning:

<think>
your reasoning
</think>
<search>
a concise search query
</search>

or, when enough evidence is available:

<think>
your reasoning
</think>
<answer>
your final answer
</answer>

Never emit both <search> and <answer> in the same response."""


def format_information(documents: Sequence[Document]) -> str:
    if not documents:
        return "<information>\nNo documents were found.\n</information>"
    blocks = [
        f"[{document.id}] {document.title}\n{document.text}\nscore={document.score}"
        for document in documents
    ]
    return "<information>\n" + "\n\n".join(blocks) + "\n</information>"


def build_agent_prompt(question: str, turns: Sequence[TurnRecord]) -> str:
    parts = [SYSTEM_INSTRUCTION, f"Question:\n{question}"]
    for turn in turns:
        parts.append(f"Assistant search action:\n{turn.model_output}")
        parts.append(turn.information)
    parts.append("Produce the next search action or the final answer now.")
    return "\n\n".join(parts)
