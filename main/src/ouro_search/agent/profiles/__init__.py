"""Configurable search-agent prompt and action protocols."""

from ouro_search.agent.profiles.base import AgentPrompt, PromptProfile, ToolSchema
from ouro_search.agent.profiles.hermes import HermesProfile
from ouro_search.agent.profiles.registry import available_prompt_profiles, get_prompt_profile
from ouro_search.agent.profiles.search_r1 import SEARCH_R1_PROMPT, SearchR1Profile

__all__ = [
    "AgentPrompt",
    "HermesProfile",
    "PromptProfile",
    "SEARCH_R1_PROMPT",
    "SearchR1Profile",
    "ToolSchema",
    "available_prompt_profiles",
    "get_prompt_profile",
]
