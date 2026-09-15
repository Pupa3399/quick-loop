from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

from ouro_search.agent.parser import ActionType, ParsedOutput
from ouro_search.agent.profiles.base import AgentPrompt, PromptProfile, ToolSchema
from ouro_search.agent.profiles.common import search_tool_schema
from ouro_search.search.types import SearchResponse
from ouro_search.trajectory.schema import TurnRecord

_TOOL_CALL_TAG = re.compile(r"<\s*/?\s*tool_call\b", re.IGNORECASE)


class HermesProfile(PromptProfile):
    """Nous Hermes XML/JSON function calling over ChatML message roles."""

    name = "hermes"
    stop_sequences = ("</tool_call>",)

    def build_system_prompt(self) -> str:
        tools = json.dumps([self.search_tool_schema()], ensure_ascii=False)
        call_schema = {
            "properties": {
                "arguments": {"title": "Arguments", "type": "object"},
                "name": {"title": "Name", "type": "string"},
            },
            "required": ["arguments", "name"],
            "title": "FunctionCall",
            "type": "object",
        }
        return (
            "You are a function calling AI model. You are provided with function "
            "signatures within <tools></tools> XML tags. You can call only one function "
            "at a time and must wait for its result before continuing. Don't make "
            "assumptions about what values to plug into functions. Here are the available "
            f"tools: <tools> {tools} </tools> Use the following pydantic model json schema "
            "for each tool call you will make: "
            f"{json.dumps(call_schema, ensure_ascii=False)} For each function call return "
            "a valid json object with function name and arguments within "
            "<tool_call></tool_call> XML tags as follows:\n<tool_call>\n"
            '{"arguments": <args-dict>, "name": <function-name>}\n'
            "</tool_call> Tool results will be provided in the next turn within "
            "<tool_response></tool_response> XML tags. Your final response should directly "
            "answer the user query."
        )

    def build_user_prompt(self, question: str) -> str:
        return question

    def search_tool_schema(self) -> ToolSchema:
        return search_tool_schema()

    def build_prompt(self, question: str, turns: list[TurnRecord]) -> AgentPrompt:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": self.build_user_prompt(question)},
        ]
        for turn in turns:
            messages.append({"role": "assistant", "content": turn.model_output})
            messages.append({"role": "tool", "content": turn.information})
        return AgentPrompt(
            messages=tuple(messages),
            stop_sequences=self.stop_sequences,
        )

    def format_observation(self, response: SearchResponse) -> str:
        payload = {
            "name": "search",
            "content": response.to_dict(),
        }
        return (
            "<tool_response>\n"
            f"{json.dumps(payload, ensure_ascii=False)}\n"
            "</tool_response>"
        )

    def parse_action(self, text: str) -> ParsedOutput:
        raw = text.strip()
        if not raw:
            return ParsedOutput(action=ActionType.UNKNOWN, think="", raw_output=raw)
        if not _TOOL_CALL_TAG.search(raw):
            return ParsedOutput(
                action=ActionType.ANSWER,
                think="",
                answer=raw,
                raw_output=raw,
            )

        try:
            root = ET.fromstring(f"<root>{raw}</root>")
        except ET.ParseError:
            return ParsedOutput(action=ActionType.UNKNOWN, think="", raw_output=raw)
        calls = root.findall(".//tool_call")
        if len(calls) != 1 or calls[0].text is None:
            return ParsedOutput(action=ActionType.UNKNOWN, think="", raw_output=raw)
        try:
            call = json.loads(calls[0].text.strip())
        except (json.JSONDecodeError, TypeError):
            return ParsedOutput(action=ActionType.UNKNOWN, think="", raw_output=raw)
        if not isinstance(call, dict):
            return ParsedOutput(action=ActionType.UNKNOWN, think="", raw_output=raw)
        arguments = call.get("arguments")
        query = arguments.get("query") if isinstance(arguments, dict) else None
        if call.get("name") != "search" or not isinstance(query, str) or not query.strip():
            return ParsedOutput(action=ActionType.UNKNOWN, think="", raw_output=raw)
        opening = re.search(r"<\s*tool_call\s*>", raw, flags=re.IGNORECASE)
        think = raw[: opening.start()].strip() if opening else ""
        return ParsedOutput(
            action=ActionType.SEARCH,
            think=think,
            query=query.strip(),
            raw_output=raw,
        )

    def parse_final_answer(self, text: str) -> str | None:
        parsed = self.parse_action(text)
        return parsed.answer if parsed.action is ActionType.ANSWER else None
