from __future__ import annotations

from collections.abc import Sequence

from ouro_search.search.types import Document
from ouro_search.trajectory.schema import TurnRecord

SEARCH_R1_PROMPT = """Answer the given question. You must conduct reasoning inside \
<think> and </think> first every time you get new information. After reasoning, if you \
find you lack some knowledge, you can call a search engine by <search> query </search> \
and it will return the top searched results between <information> and </information>. \
You can search at most four times. If you find no further external knowledge needed, \
you can directly provide the answer inside <answer> and </answer>, without detailed \
illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

# Inference engines remove this separator before tokenization. It lets them apply a
# tokenizer chat template once to the original user prompt, then append search history
# as the same assistant generation stream used by Search-R1 training.
PROMPT_CONTINUATION_SEPARATOR = "\n<|ouro_search_continuation|>\n"


def format_information(documents: Sequence[Document]) -> str:
    if not documents:
        return "<information>\nNo documents were found.\n</information>"
    blocks = [
        f"[{document.id}] {document.title}\n{document.text}\nscore={document.score}"
        for document in documents
    ]
    return "<information>\n" + "\n\n".join(blocks) + "\n</information>"


def build_agent_prompt(question: str, turns: Sequence[TurnRecord]) -> str:
    initial_prompt = SEARCH_R1_PROMPT.format(question=question)
    if not turns:
        return initial_prompt
    continuation = "".join(
        f"{turn.model_output}\n{turn.information}\n" for turn in turns
    )
    return initial_prompt + PROMPT_CONTINUATION_SEPARATOR + continuation


def split_agent_prompt(prompt: str) -> tuple[str, str]:
    initial_prompt, separator, continuation = prompt.partition(PROMPT_CONTINUATION_SEPARATOR)
    return initial_prompt, continuation if separator else ""
