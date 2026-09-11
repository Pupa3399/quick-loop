from __future__ import annotations

from collections.abc import Sequence

from ouro_search.agent.profiles import SEARCH_R1_PROMPT, AgentPrompt, SearchR1Profile
from ouro_search.search.types import Document
from ouro_search.trajectory.schema import TurnRecord

__all__ = ["SEARCH_R1_PROMPT", "build_agent_prompt", "format_information"]


def format_information(documents: Sequence[Document]) -> str:
    """Compatibility wrapper for the Search-R1 observation format."""
    from ouro_search.search.types import SearchResponse

    return SearchR1Profile().format_observation(
        SearchResponse(query="", documents=list(documents))
    )


def build_agent_prompt(question: str, turns: Sequence[TurnRecord]) -> AgentPrompt:
    """Compatibility wrapper for constructing a Search-R1 prompt."""
    return SearchR1Profile().build_prompt(question, list(turns))
