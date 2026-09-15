from __future__ import annotations

from copy import deepcopy

from ouro_search.agent.profiles.base import ToolSchema

_SEARCH_TOOL_SCHEMA: ToolSchema = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the Wikipedia-2018 passage corpus for relevant evidence.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query used to retrieve relevant passages.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def search_tool_schema() -> ToolSchema:
    return deepcopy(_SEARCH_TOOL_SCHEMA)
