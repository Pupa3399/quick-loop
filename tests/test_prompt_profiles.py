from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from ouro_search.agent.parser import ActionType
from ouro_search.agent.profiles import AgentPrompt, get_prompt_profile
from ouro_search.agent.runner import AgentRunner
from ouro_search.search.types import Document, SearchResponse
from ouro_search.trajectory.schema import TurnRecord


def _response() -> SearchResponse:
    return SearchResponse(
        query="Hamlet author",
        documents=[
            Document(
                id="wiki-1",
                title="Hamlet",
                text="Hamlet was written by William Shakespeare.",
                score=0.91,
            )
        ],
    )


def test_hydra_selects_prompt_profile() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="config", overrides=["prompt_profile=hermes"])

    assert config.prompt_profile.name == "hermes"


def test_search_r1_constructs_official_prompt_and_observation() -> None:
    profile = get_prompt_profile("search_r1")
    prompt = profile.build_prompt("Who wrote Hamlet?", [])

    assert profile.build_system_prompt() == ""
    assert prompt.messages[0]["role"] == "user"
    assert "You can search as many times as your want." in prompt.messages[0]["content"]
    assert prompt.messages[0]["content"].endswith("Question: Who wrote Hamlet?\n")
    assert prompt.stop_sequences == ("</search>", "</answer>")
    assert profile.format_observation(_response()) == (
        "<information>Doc 1(Title: Hamlet) "
        "Hamlet was written by William Shakespeare.\n</information>"
    )


def test_search_r1_reinserts_observation_in_single_generation_stream() -> None:
    profile = get_prompt_profile("search_r1")
    observation = profile.format_observation(_response())
    prompt = profile.build_prompt(
        "Who wrote Hamlet?",
        [
            TurnRecord(
                turn_id=0,
                loop_steps=3,
                think="Need evidence.",
                query="Hamlet author",
                information=observation,
                retrieved_documents=_response().documents,
                model_output="<think>Need evidence.</think><search>Hamlet author</search>",
            )
        ],
    )

    assert len(prompt.messages) == 1
    assert prompt.continuation == (
        "\n\n<think>Need evidence.</think><search>Hamlet author</search>"
        f"{observation}\n\n"
    )


def test_hermes_constructs_search_schema_and_chatml_messages() -> None:
    profile = get_prompt_profile("hermes")
    schema = profile.search_tool_schema()
    system_prompt = profile.build_system_prompt()

    assert schema["function"]["name"] == "search"
    assert schema["function"]["parameters"]["required"] == ["query"]
    assert "<tools>" in system_prompt
    assert '"name": "search"' in system_prompt
    assert "<tool_call>" in system_prompt

    observation = profile.format_observation(_response())
    turn = TurnRecord(
        turn_id=0,
        loop_steps=3,
        think="",
        query="Hamlet author",
        information=observation,
        retrieved_documents=_response().documents,
        model_output=(
            '<tool_call>\n{"arguments": {"query": "Hamlet author"}, '
            '"name": "search"}\n</tool_call>'
        ),
    )
    prompt = profile.build_prompt("Who wrote Hamlet?", [turn])

    assert [message["role"] for message in prompt.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert prompt.messages[-1]["content"] == observation
    payload = json.loads(observation.removeprefix("<tool_response>\n").removesuffix(
        "\n</tool_response>"
    ))
    assert payload["name"] == "search"
    assert payload["content"] == _response().to_dict()


def test_hermes_parses_tool_call_and_direct_final_answer() -> None:
    profile = get_prompt_profile("hermes")
    action = profile.parse_action(
        "I should search.\n<tool_call>\n"
        '{"arguments": {"query": "Hamlet author"}, "name": "search"}'
        "\n</tool_call>"
    )
    final = profile.parse_action("William Shakespeare wrote Hamlet.")

    assert action.action is ActionType.SEARCH
    assert action.query == "Hamlet author"
    assert action.think == "I should search."
    assert final.action is ActionType.ANSWER
    assert profile.parse_final_answer(final.raw_output) == "William Shakespeare wrote Hamlet."


@pytest.mark.parametrize(
    "output",
    [
        '<tool_call>{"arguments": {"query": "Hamlet"}, "name": "search"}',
        "<tool_call>{not json}</tool_call>",
        '<tool_call>{"arguments": {}, "name": "search"}</tool_call>',
        '<tool_call>{"arguments": {"query": "Hamlet"}, "name": "browse"}</tool_call>',
        (
            '<tool_call>{"arguments": {"query": "one"}, "name": "search"}</tool_call>'
            '<tool_call>{"arguments": {"query": "two"}, "name": "search"}</tool_call>'
        ),
    ],
)
def test_hermes_rejects_malformed_or_unsupported_actions(output: str) -> None:
    parsed = get_prompt_profile("hermes").parse_action(output)

    assert parsed.action is ActionType.UNKNOWN
    assert get_prompt_profile("hermes").parse_final_answer(output) is None


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown prompt profile"):
        get_prompt_profile("not-a-profile")


@dataclass
class RecordingEngine:
    outputs: list[str]
    prompts: list[AgentPrompt] = field(default_factory=list)

    def generate(self, prompt: AgentPrompt, **kwargs: object) -> str:
        del kwargs
        self.prompts.append(prompt)
        return self.outputs.pop(0)


class StubSearchClient:
    def search(self, query: str) -> SearchResponse:
        assert query == "Hamlet author"
        return _response()


def test_runner_uses_hermes_without_a_second_agent_flow() -> None:
    engine = RecordingEngine(
        outputs=[
            '<tool_call>{"arguments": {"query": "Hamlet author"}, '
            '"name": "search"}</tool_call>',
            "William Shakespeare",
        ]
    )
    runner = AgentRunner(engine, StubSearchClient(), prompt_profile="hermes")

    trajectory = runner.run(
        "Who wrote Hamlet?",
        answer_aliases=["William Shakespeare"],
    )

    assert trajectory.prediction == "William Shakespeare"
    assert trajectory.reward == 1.0
    assert trajectory.num_search_turns == 1
    assert len(engine.prompts) == 2
    assert [message["role"] for message in engine.prompts[1].messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
