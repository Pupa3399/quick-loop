from __future__ import annotations

from typing import Protocol

from ouro_search.agent.parser import ActionType, parse_agent_output
from ouro_search.agent.protocol import build_agent_prompt, format_information
from ouro_search.search.types import SearchResponse
from ouro_search.trajectory.schema import Trajectory, TurnRecord
from ouro_search.trajectory.writer import JsonlTrajectoryWriter


class InferenceEngine(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        loop_steps: int,
    ) -> str: ...


class SearchClient(Protocol):
    def search(self, query: str) -> SearchResponse: ...


class AgentRunner:
    def __init__(
        self,
        engine: InferenceEngine,
        search_client: SearchClient,
        *,
        max_search_turns: int = 4,
        default_loop_steps: int = 3,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        trajectory_writer: JsonlTrajectoryWriter | None = None,
    ) -> None:
        if max_search_turns < 0:
            raise ValueError("max_search_turns cannot be negative")
        if default_loop_steps < 1:
            raise ValueError("default_loop_steps must be positive")
        self.engine = engine
        self.search_client = search_client
        self.max_search_turns = max_search_turns
        self.default_loop_steps = default_loop_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.trajectory_writer = trajectory_writer

    def _validate_loop_schedule(self, schedule: dict[int, int]) -> None:
        for turn_id, steps in schedule.items():
            if isinstance(turn_id, bool) or not isinstance(turn_id, int) or turn_id < 0:
                raise ValueError("loop_steps_by_turn keys must be non-negative 0-based integers")
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
                raise ValueError("loop_steps_by_turn values must be positive integers")

    def run(
        self,
        question: str,
        *,
        sample_id: str = "sample-0",
        loop_steps_by_turn: dict[int, int] | None = None,
    ) -> Trajectory:
        if not question.strip():
            raise ValueError("question cannot be empty")
        schedule = dict(loop_steps_by_turn or {})
        self._validate_loop_schedule(schedule)

        turns: list[TurnRecord] = []
        prediction = ""
        termination_reason = ""
        final_model_output = ""

        # One final generation is allowed after the last retrieval.
        for _ in range(self.max_search_turns + 1):
            turn_id = len(turns)
            loop_steps = schedule.get(turn_id, self.default_loop_steps)
            prompt = build_agent_prompt(question.strip(), turns)
            final_model_output = self.engine.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                loop_steps=loop_steps,
            )
            parsed = parse_agent_output(final_model_output)

            if parsed.action is ActionType.ANSWER:
                prediction = parsed.answer or ""
                termination_reason = "answer"
                break
            if parsed.action is ActionType.UNKNOWN:
                prediction = final_model_output
                termination_reason = "unparseable_output"
                break
            if len(turns) >= self.max_search_turns:
                prediction = final_model_output
                termination_reason = "max_search_turns"
                break

            query = parsed.query or ""
            response = self.search_client.search(query)
            information = format_information(response.documents)
            turns.append(
                TurnRecord(
                    turn_id=turn_id,
                    loop_steps=loop_steps,
                    think=parsed.think,
                    query=query,
                    information=information,
                    retrieved_documents=response.documents,
                    model_output=final_model_output,
                )
            )

        trajectory = Trajectory(
            id=sample_id,
            question=question.strip(),
            prediction=prediction,
            num_search_turns=len(turns),
            turns=turns,
            termination_reason=termination_reason,
            final_model_output=final_model_output,
        )
        if self.trajectory_writer is not None:
            self.trajectory_writer.write(trajectory)
        return trajectory
