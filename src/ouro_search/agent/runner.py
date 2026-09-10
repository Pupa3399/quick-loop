from __future__ import annotations

from typing import Protocol

from ouro_search.agent.parser import ActionType
from ouro_search.agent.profiles import AgentPrompt, PromptProfile, get_prompt_profile
from ouro_search.rewards.answer_reward import answer_exact_match
from ouro_search.search.types import SearchResponse
from ouro_search.trajectory.schema import Trajectory, TurnRecord
from ouro_search.trajectory.writer import JsonlTrajectoryWriter


class InferenceEngine(Protocol):
    def generate(
        self,
        prompt: str | AgentPrompt,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        loop_steps: int,
    ) -> str: ...

    def count_tokens(self, text: str) -> int: ...

    def truncate_text(self, text: str, max_tokens: int) -> str: ...


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
        max_information_tokens: int = 500,
        max_response_tokens: int = 3000,
        temperature: float = 0.0,
        top_p: float = 1.0,
        trajectory_writer: JsonlTrajectoryWriter | None = None,
        prompt_profile: str | PromptProfile = "search_r1",
    ) -> None:
        if max_search_turns < 0:
            raise ValueError("max_search_turns cannot be negative")
        if default_loop_steps < 1:
            raise ValueError("default_loop_steps must be positive")
        if max_information_tokens < 1 or max_response_tokens < 1:
            raise ValueError("information and response token budgets must be positive")
        self.engine = engine
        self.search_client = search_client
        self.max_search_turns = max_search_turns
        self.default_loop_steps = default_loop_steps
        self.max_new_tokens = max_new_tokens
        self.max_information_tokens = max_information_tokens
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.trajectory_writer = trajectory_writer
        self.prompt_profile = get_prompt_profile(prompt_profile)

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
        reference_answer: str = "",
        answer_aliases: list[str] | None = None,
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
        response_token_count = 0

        # One final generation is allowed after the last retrieval.
        for _ in range(self.max_search_turns + 1):
            turn_id = len(turns)
            loop_steps = schedule.get(turn_id, self.default_loop_steps)
            prompt = self.prompt_profile.build_prompt(question.strip(), turns)
            remaining_tokens = self.max_response_tokens - response_token_count
            if remaining_tokens <= 0:
                termination_reason = "max_response_length"
                break
            final_model_output = self.engine.generate(
                prompt,
                max_new_tokens=min(self.max_new_tokens, remaining_tokens),
                temperature=self.temperature,
                top_p=self.top_p,
                loop_steps=loop_steps,
            )
            response_token_count += self._count_tokens(final_model_output)
            parsed = self.prompt_profile.parse_action(final_model_output)

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
            information = self.prompt_profile.format_observation(response)
            remaining_tokens = self.max_response_tokens - response_token_count
            information = self._truncate_text(
                information,
                min(self.max_information_tokens, max(remaining_tokens, 0)),
            )
            response_token_count += self._count_tokens(information)
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
            if response_token_count >= self.max_response_tokens:
                termination_reason = "max_response_length"
                break

        trajectory = Trajectory(
            id=sample_id,
            question=question.strip(),
            prediction=prediction,
            num_search_turns=len(turns),
            reference_answer=reference_answer,
            answer_aliases=list(answer_aliases or ([reference_answer] if reference_answer else [])),
            turns=turns,
            termination_reason=termination_reason,
            final_model_output=final_model_output,
            response_token_count=min(response_token_count, self.max_response_tokens),
        )
        final_answer = self.prompt_profile.parse_final_answer(final_model_output)
        trajectory.reward = float(
            final_answer is not None
            and answer_exact_match(final_answer, trajectory.answer_aliases)
        )
        if self.trajectory_writer is not None:
            self.trajectory_writer.write(trajectory)
        return trajectory

    def _count_tokens(self, text: str) -> int:
        counter = getattr(self.engine, "count_tokens", None)
        return counter(text) if callable(counter) else len(text.split())

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        truncator = getattr(self.engine, "truncate_text", None)
        if callable(truncator):
            return truncator(text, max_tokens)
        return " ".join(text.split()[:max_tokens])
