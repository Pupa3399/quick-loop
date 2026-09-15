from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ouro_search.agent.parser import ParsedOutput
from ouro_search.search.types import SearchResponse
from ouro_search.trajectory.schema import TurnRecord

ChatMessage = dict[str, Any]
ToolSchema = dict[str, Any]


@dataclass(frozen=True, slots=True)
class AgentPrompt:
    """Structured model input plus an optional raw assistant continuation."""

    messages: tuple[ChatMessage, ...]
    continuation: str = ""
    stop_sequences: tuple[str, ...] = field(default_factory=tuple)


class PromptProfile(ABC):
    """Protocol boundary shared by all search-agent prompt formats."""

    name: str
    stop_sequences: tuple[str, ...]

    @abstractmethod
    def build_system_prompt(self) -> str:
        """Return the profile's system prompt, or an empty string when unused."""

    @abstractmethod
    def build_user_prompt(self, question: str) -> str:
        """Return the user-facing question prompt."""

    @abstractmethod
    def search_tool_schema(self) -> ToolSchema:
        """Return the Search tool schema exposed by this profile."""

    @abstractmethod
    def build_prompt(self, question: str, turns: list[TurnRecord]) -> AgentPrompt:
        """Build the complete context for the next model generation."""

    @abstractmethod
    def format_observation(self, response: SearchResponse) -> str:
        """Format a Retriever response for insertion into model context."""

    @abstractmethod
    def parse_action(self, text: str) -> ParsedOutput:
        """Parse one model generation as a search action or final answer."""

    @abstractmethod
    def parse_final_answer(self, text: str) -> str | None:
        """Extract a final answer, returning None for non-final or malformed output."""
