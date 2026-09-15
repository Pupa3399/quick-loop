from __future__ import annotations

from ouro_search.agent.parser import ParsedOutput, parse_agent_output
from ouro_search.agent.profiles.base import AgentPrompt, PromptProfile, ToolSchema
from ouro_search.agent.profiles.common import search_tool_schema
from ouro_search.search.types import SearchResponse
from ouro_search.trajectory.schema import TurnRecord

# Kept byte-for-byte equivalent to Search-R1's official inference prompt, including
# its original wording. The runtime search limit remains an AgentRunner concern.
SEARCH_R1_PROMPT = """Answer the given question. You must conduct reasoning inside \
<think> and </think> first every time you get new information. After reasoning, if you \
find you lack some knowledge, you can call a search engine by <search> query </search> \
and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. If you find no further external knowledge \
needed, you can directly provide the answer inside <answer> and </answer>, without \
detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""


class SearchR1Profile(PromptProfile):
    name = "search_r1"
    stop_sequences = ("</search>", "</answer>")

    def build_system_prompt(self) -> str:
        return ""

    def build_user_prompt(self, question: str) -> str:
        return SEARCH_R1_PROMPT.format(question=question)

    def search_tool_schema(self) -> ToolSchema:
        return search_tool_schema()

    def build_prompt(self, question: str, turns: list[TurnRecord]) -> AgentPrompt:
        continuation = "".join(
            f"\n\n{turn.model_output}{turn.information}\n\n" for turn in turns
        )
        return AgentPrompt(
            messages=({"role": "user", "content": self.build_user_prompt(question)},),
            continuation=continuation,
            stop_sequences=self.stop_sequences,
        )

    def format_observation(self, response: SearchResponse) -> str:
        references = "".join(
            f"Doc {index}(Title: {document.title}) {document.text}\n"
            for index, document in enumerate(response.documents, start=1)
        )
        return f"<information>{references}</information>"

    def parse_action(self, text: str) -> ParsedOutput:
        return parse_agent_output(text)

    def parse_final_answer(self, text: str) -> str | None:
        return parse_agent_output(text).answer
