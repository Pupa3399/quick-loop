from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from ouro_search.agent.parser import ActionType, ParsedOutput
from ouro_search.search.types import SearchResponse

SEARCH_R1_V0_2_PROMPT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <search> query </search> and it "
    "will return the top searched results between <information> and </information>. "
    "You can search at most four times. If you find no further external knowledge "
    "needed, you can directly provide the answer inside <answer> and </answer>, without "
    "detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
)

INVALID_ACTION_OBSERVATION = (
    "\nMy previous action is invalid. "
    "If I want to search, I should put the query between <search> and </search>. "
    "If I want to give the final answer, I should put the answer between <answer> and "
    "</answer>. Let me try again.\n"
)

_ACTION_PATTERN = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class TokenizerLike(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str: ...


@dataclass(frozen=True, slots=True)
class ProcessedGeneration:
    raw_text: str
    text: str
    token_ids: list[int]


def build_prompt(question: str) -> str:
    return SEARCH_R1_V0_2_PROMPT.format(question=question)


def strict_parse_action(text: str) -> ParsedOutput:
    """Match Search-R1's first, case-sensitive, fully closed action."""
    match = _ACTION_PATTERN.search(text)
    think_match = _THINK_PATTERN.search(text)
    think = think_match.group(1).strip() if think_match else ""
    if match is None:
        return ParsedOutput(action=ActionType.UNKNOWN, think=think, raw_output=text)
    content = match.group(2).strip()
    if match.group(1) == "search":
        return ParsedOutput(
            action=ActionType.SEARCH,
            think=think,
            query=content,
            raw_output=text,
        )
    return ParsedOutput(
        action=ActionType.ANSWER,
        think=think,
        answer=content,
        raw_output=text,
    )


def postprocess_generation(
    raw_token_ids: list[int], tokenizer: TokenizerLike
) -> ProcessedGeneration:
    """Decode, crop, then re-tokenize exactly in the Search-R1 v0.2 order."""
    raw_text = tokenizer.decode(raw_token_ids, skip_special_tokens=True)
    if "</search>" in raw_text:
        text = raw_text.split("</search>", 1)[0] + "</search>"
    elif "</answer>" in raw_text:
        text = raw_text.split("</answer>", 1)[0] + "</answer>"
    else:
        text = raw_text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return ProcessedGeneration(raw_text=raw_text, text=text, token_ids=token_ids)


def render_search_result(response: SearchResponse) -> str:
    return "".join(
        f"Doc {index}(Title: {document.title}) {document.text}\n"
        for index, document in enumerate(response.documents, start=1)
    )


def format_observation(search_result: str) -> str:
    return f"\n\n<information>{search_result.strip()}</information>\n\n"


def truncate_observation(token_ids: list[int], max_obs_length: int) -> list[int]:
    return token_ids[:max_obs_length]


def initial_context(token_ids: list[int], max_start_length: int) -> list[int]:
    return token_ids[-max_start_length:]


def update_rolling_context(
    current_ids: list[int],
    response_ids: list[int],
    observation_ids: list[int],
    max_prompt_length: int,
) -> list[int]:
    return (current_ids + response_ids + observation_ids)[-max_prompt_length:]


def official_sampling_params(
    sampling_params: dict[str, Any], *, max_response_length: int
) -> dict[str, Any]:
    """Use vLLM's ongoing RNG stream and post-generation action cropping."""
    result = dict(sampling_params)
    result.pop("stop", None)
    result.pop("include_stop_str_in_output", None)
    result.pop("seed", None)
    result["max_tokens"] = max_response_length
    return result
