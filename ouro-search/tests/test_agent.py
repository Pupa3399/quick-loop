from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ouro_search.agent.runner import AgentRunner
from ouro_search.search.types import Document, SearchResponse


@dataclass
class ScriptedEngine:
    outputs: list[str]
    loop_steps: list[int] = field(default_factory=list)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        loop_steps: int,
    ) -> str:
        del prompt, max_new_tokens, temperature, top_p
        self.loop_steps.append(loop_steps)
        return self.outputs.pop(0)


@dataclass
class StubSearchClient:
    queries: list[str] = field(default_factory=list)

    def search(self, query: str) -> SearchResponse:
        self.queries.append(query)
        return SearchResponse(
            query=query,
            documents=[
                Document(
                    id="doc-0",
                    title="Evidence",
                    text=f"Evidence for {query}",
                    score=1.0,
                )
            ],
        )


def test_agent_search_then_answer_with_zero_based_schedule() -> None:
    engine = ScriptedEngine(
        outputs=[
            "<think>first</think><search>query one</search>",
            "<think>second</think><search>query two</search>",
            "<think>done</think><answer>the answer</answer>",
        ]
    )
    search_client = StubSearchClient()
    runner = AgentRunner(engine, search_client, max_search_turns=4)

    trajectory = runner.run("question", loop_steps_by_turn={1: 4})

    assert trajectory.prediction == "the answer"
    assert trajectory.num_search_turns == 2
    assert [turn.turn_id for turn in trajectory.turns] == [0, 1]
    assert [turn.loop_steps for turn in trajectory.turns] == [3, 4]
    assert engine.loop_steps == [3, 4, 3]
    assert search_client.queries == ["query one", "query two"]
    assert trajectory.termination_reason == "answer"
    assert trajectory.reward == 0.0


def test_agent_records_exact_match_reward() -> None:
    engine = ScriptedEngine(outputs=["<think>done</think><answer>Shakespeare</answer>"])
    runner = AgentRunner(engine, StubSearchClient())

    trajectory = runner.run(
        "Who wrote Hamlet?",
        reference_answer="William Shakespeare",
        answer_aliases=["Shakespeare", "William Shakespeare"],
    )

    assert trajectory.reward == 1.0


def test_agent_stops_at_search_turn_limit() -> None:
    engine = ScriptedEngine(
        outputs=[
            "<search>one</search>",
            "<search>two</search>",
            "<search>three</search>",
        ]
    )
    runner = AgentRunner(engine, StubSearchClient(), max_search_turns=2)

    trajectory = runner.run("question")

    assert trajectory.num_search_turns == 2
    assert trajectory.termination_reason == "max_search_turns"
    assert len(engine.loop_steps) == 3


def test_agent_preserves_unparseable_real_output() -> None:
    engine = ScriptedEngine(outputs=["A raw model response without protocol tags."])
    runner = AgentRunner(engine, StubSearchClient())

    trajectory = runner.run("question")

    assert trajectory.prediction == "A raw model response without protocol tags."
    assert trajectory.termination_reason == "unparseable_output"
    assert trajectory.num_search_turns == 0


def test_agent_rejects_non_zero_based_schedule_keys() -> None:
    runner = AgentRunner(ScriptedEngine(outputs=[]), StubSearchClient())

    with pytest.raises(ValueError, match="non-negative"):
        runner.run("question", loop_steps_by_turn={-1: 4})
