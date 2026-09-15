from __future__ import annotations

import json
import re
import time
from typing import Protocol

from ouro_search.agent.parser import ActionType
from ouro_search.agent.profiles import AgentPrompt, PromptProfile, get_prompt_profile
from ouro_search.inference.types import GenerationResult
from ouro_search.rewards.answer_reward import answer_exact_match
from ouro_search.search.types import SearchResponse
from ouro_search.trajectory.schema import GenerationRecord, Trajectory, TurnRecord
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
        max_new_tokens: int | None = 512,
        max_information_tokens: int | None = 500,
        max_response_tokens: int | None = 3000,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        trajectory_writer: JsonlTrajectoryWriter | None = None,
        prompt_profile: str | PromptProfile = "search_r1",
    ) -> None:
        if max_search_turns < 0:
            raise ValueError("max_search_turns cannot be negative")
        if default_loop_steps < 1:
            raise ValueError("default_loop_steps must be positive")
        for name, value in (
            ("max_new_tokens", max_new_tokens),
            ("max_information_tokens", max_information_tokens),
            ("max_response_tokens", max_response_tokens),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive or None")
        self.engine = engine
        self.search_client = search_client
        self.max_search_turns = max_search_turns
        self.default_loop_steps = default_loop_steps
        self.max_new_tokens = max_new_tokens
        self.max_information_tokens = max_information_tokens
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
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
        raw_generations: list[str] = []
        response_token_count = 0
        generations: list[GenerationRecord] = []
        initial_messages: list[dict[str, object]] = []
        initial_rendered_prompt = ""

        # One final generation is allowed after the last retrieval.
        for _ in range(self.max_search_turns + 1):
            turn_id = len(generations)
            loop_steps = schedule.get(turn_id, self.default_loop_steps)
            prompt = self.prompt_profile.build_prompt(question.strip(), turns)
            context_remaining = self._available_generation_tokens(prompt)
            if context_remaining is not None and context_remaining <= 0:
                termination_reason = "context_exhausted"
                break
            remaining_tokens = (
                None
                if self.max_response_tokens is None
                else self.max_response_tokens - response_token_count
            )
            if remaining_tokens is not None and remaining_tokens <= 0:
                termination_reason = "max_response_length"
                break
            limits = [
                value
                for value in (self.max_new_tokens, remaining_tokens, context_remaining)
                if value is not None
            ]
            if not limits:
                raise RuntimeError("Unbounded generation requires an engine context limit")
            requested_max_new_tokens = min(limits)
            result = self._generate(
                prompt,
                requested_max_new_tokens=requested_max_new_tokens,
                loop_steps=loop_steps,
            )
            final_model_output = result.text
            raw_generations.append(final_model_output)
            completion_count = len(result.completion_token_ids) or self._count_tokens(
                final_model_output
            )
            response_token_count += completion_count
            parsed = self.prompt_profile.parse_action(final_model_output)
            generation = self._generation_record(
                turn_id=turn_id,
                loop_steps=loop_steps,
                prompt=prompt,
                result=result,
                parsed=parsed,
                requested_max_new_tokens=requested_max_new_tokens,
                context_remaining=context_remaining,
            )
            generations.append(generation)
            if turn_id == 0:
                initial_messages = generation.messages
                initial_rendered_prompt = generation.rendered_prompt

            if parsed.action is ActionType.ANSWER:
                prediction = parsed.answer or ""
                termination_reason = "answer"
                break
            if parsed.action is ActionType.UNKNOWN:
                prediction = final_model_output
                termination_reason = (
                    "context_exhausted"
                    if result.finish_reason == "length"
                    and context_remaining == requested_max_new_tokens
                    else "length_exhausted"
                    if result.finish_reason == "length"
                    else "unparseable_output"
                )
                break
            if len(turns) >= self.max_search_turns:
                prediction = final_model_output
                termination_reason = "max_search_turns"
                break

            query = parsed.query or ""
            response = self.search_client.search(query)
            information = self.prompt_profile.format_observation(response)
            if self.max_information_tokens is not None:
                information_limit = self.max_information_tokens
                if self.max_response_tokens is not None:
                    information_limit = min(
                        information_limit,
                        max(self.max_response_tokens - response_token_count, 0),
                    )
                information = self._truncate_text(information, information_limit)
            information_token_ids = self._encode_text(information)
            information_count = len(information_token_ids) or self._count_tokens(information)
            response_token_count += information_count
            generation.retriever_called = True
            generation.retriever_query = query
            generation.retriever_latency_seconds = response.latency_seconds
            generation.retriever_raw_json = response.raw_json or response.to_dict()
            generation.retrieved_documents = response.documents
            generation.observation_text = information
            generation.observation_token_ids = information_token_ids
            turns.append(
                TurnRecord(
                    turn_id=turn_id,
                    loop_steps=loop_steps,
                    think=parsed.think,
                    query=query,
                    information=information,
                    retrieved_documents=response.documents,
                    model_output=final_model_output,
                    generation_id=turn_id,
                    retriever_latency_seconds=response.latency_seconds,
                    retriever_raw_json=response.raw_json or response.to_dict(),
                    observation_token_ids=information_token_ids,
                )
            )
            if (
                self.max_response_tokens is not None
                and response_token_count >= self.max_response_tokens
            ):
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
            raw_generations=raw_generations,
            response_token_count=(
                response_token_count
                if self.max_response_tokens is None
                else min(response_token_count, self.max_response_tokens)
            ),
            prompt_profile=self.prompt_profile.name,
            initial_messages=initial_messages,
            initial_rendered_prompt=initial_rendered_prompt,
            generations=generations,
            total_prompt_tokens=sum(item.prompt_token_count for item in generations),
            total_completion_tokens=sum(item.completion_token_count for item in generations),
            total_latency_seconds=sum(
                item.generation_latency_seconds
                + (item.retriever_latency_seconds or 0.0)
                for item in generations
            ),
            final_finish_reason=generations[-1].finish_reason if generations else None,
            final_stop_reason=generations[-1].stop_reason if generations else None,
        )
        final_answer = self.prompt_profile.parse_final_answer(final_model_output)
        trajectory.reward = float(
            final_answer is not None
            and answer_exact_match(final_answer, trajectory.answer_aliases)
        )
        if self.trajectory_writer is not None:
            self.trajectory_writer.write(trajectory)
        return trajectory

    def _available_generation_tokens(self, prompt: str | AgentPrompt) -> int | None:
        budgeter = getattr(self.engine, "available_generation_tokens", None)
        return int(budgeter(prompt)) if callable(budgeter) else None

    def _generate(
        self,
        prompt: str | AgentPrompt,
        *,
        requested_max_new_tokens: int,
        loop_steps: int,
    ) -> GenerationResult:
        detailed = getattr(self.engine, "generate_detailed", None)
        if callable(detailed):
            return detailed(
                prompt,
                max_new_tokens=requested_max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                loop_steps=loop_steps,
                seed=self.seed,
            )
        started = time.perf_counter()
        text = self.engine.generate(
            prompt,
            max_new_tokens=requested_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            loop_steps=loop_steps,
        )
        rendered = self._render_prompt(prompt)
        return GenerationResult(
            text=text,
            rendered_prompt=rendered,
            prompt_token_ids=self._encode_text(rendered),
            completion_token_ids=self._encode_text(text),
            finish_reason=None,
            stop_reason=None,
            latency_seconds=time.perf_counter() - started,
        )

    def _generation_record(
        self,
        *,
        turn_id: int,
        loop_steps: int,
        prompt: str | AgentPrompt,
        result: GenerationResult,
        parsed: object,
        requested_max_new_tokens: int,
        context_remaining: int | None,
    ) -> GenerationRecord:
        from ouro_search.agent.protocol_audit import audit_generation

        audit = audit_generation(self.prompt_profile.name, result.text)
        think = getattr(parsed, "think", "") or ""
        query = getattr(parsed, "query", None)
        think_span = self._think_span(result.text, think)
        query_span = self._query_span(result.text, query)
        think_token_span = self._token_span(result.text, think_span)
        query_token_span = self._token_span(result.text, query_span)
        token_indexes: set[int] = set()
        for span in re.finditer(
            r"</?(?:think|search|answer|tool_call)>|\"(?:name|arguments|query)\"",
            result.text,
            flags=re.IGNORECASE,
        ):
            token_span = self._token_span(result.text, (span.start(), span.end()))
            if token_span is not None:
                token_indexes.update(range(token_span[0], token_span[1]))
        if query_token_span is not None:
            token_indexes.update(range(query_token_span[0], query_token_span[1]))
        span_top_logprobs = [
            {
                "token_index": index,
                "token_id": result.completion_token_ids[index],
                "chosen_logprob": result.chosen_token_logprobs[index]
                if index < len(result.chosen_token_logprobs)
                else None,
                "top_logprobs": result.token_top_logprobs[index]
                if index < len(result.token_top_logprobs)
                else [],
            }
            for index in sorted(token_indexes)
            if index < len(result.completion_token_ids)
        ]
        messages = (
            [dict(message) for message in prompt.messages]
            if isinstance(prompt, AgentPrompt)
            else [{"role": "user", "content": prompt}]
        )
        action_type = getattr(getattr(parsed, "action", None), "value", "unknown")
        return GenerationRecord(
            turn_id=turn_id,
            loop_steps=loop_steps,
            messages=messages,
            prompt_continuation=(
                prompt.continuation if isinstance(prompt, AgentPrompt) else ""
            ),
            rendered_prompt=result.rendered_prompt,
            prompt_token_ids=result.prompt_token_ids,
            raw_generation=result.text,
            completion_token_ids=result.completion_token_ids,
            think=think,
            action_type=action_type,
            raw_action=self._raw_action(result.text, action_type),
            parsed_query=query,
            parser_valid=audit.valid_action,
            format_valid=audit.format_valid,
            explicit_format_valid=audit.explicit_protocol_format,
            malformed_reason=audit.reason,
            finish_reason=result.finish_reason,
            stop_reason=result.stop_reason,
            protocol_stop=bool(
                result.finish_reason == "stop"
                and isinstance(result.stop_reason, str)
                and result.stop_reason in self.prompt_profile.stop_sequences
            ),
            generation_latency_seconds=result.latency_seconds,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            requested_max_new_tokens=requested_max_new_tokens,
            context_tokens_remaining=context_remaining,
            think_character_span=list(think_span) if think_span is not None else None,
            think_token_span=think_token_span,
            query_character_span=list(query_span) if query_span is not None else None,
            query_token_span=query_token_span,
            query_prefix=result.text[: query_span[0]] if query_span is not None else "",
            context_snapshot=result.rendered_prompt,
            chosen_token_logprobs=result.chosen_token_logprobs,
            span_top_logprobs=span_top_logprobs,
        )

    def _render_prompt(self, prompt: str | AgentPrompt) -> str:
        renderer = getattr(self.engine, "render_prompt", None)
        if callable(renderer):
            return str(renderer(prompt))
        if isinstance(prompt, str):
            return prompt
        message_text = "\n".join(
            str(message.get("content", "")) for message in prompt.messages
        )
        return message_text + prompt.continuation

    def _encode_text(self, text: str) -> list[int]:
        encoder = getattr(self.engine, "encode_text", None)
        return list(encoder(text)) if callable(encoder) else []

    def _token_span(self, text: str, span: tuple[int, int] | None) -> list[int] | None:
        if span is None:
            return None
        converter = getattr(self.engine, "char_span_to_token_span", None)
        return converter(text, span) if callable(converter) else None

    @staticmethod
    def _text_span(text: str, value: str | None) -> tuple[int, int] | None:
        if not value:
            return None
        start = text.find(value)
        return (start, start + len(value)) if start >= 0 else None

    def _think_span(self, text: str, think: str) -> tuple[int, int] | None:
        if self.prompt_profile.name == "search_r1":
            span = self._tag_content_span(text, "think", think)
            if span is not None:
                return span
        return self._text_span(text, think)

    def _query_span(self, text: str, query: str | None) -> tuple[int, int] | None:
        if not query:
            return None
        if self.prompt_profile.name == "search_r1":
            span = self._tag_content_span(text, "search", query)
            if span is not None:
                return span
        if self.prompt_profile.name == "hermes":
            for match in re.finditer(
                r'"query"\s*:\s*("(?:\\.|[^"\\])*")', text, flags=re.DOTALL
            ):
                try:
                    parsed_value = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                if parsed_value == query:
                    start, end = match.span(1)
                    return start + 1, end - 1
        start = text.rfind(query)
        return (start, start + len(query)) if start >= 0 else None

    @staticmethod
    def _tag_content_span(
        text: str, tag: str, expected_value: str
    ) -> tuple[int, int] | None:
        matches = list(
            re.finditer(
                rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL
            )
        )
        for match in reversed(matches):
            raw_value = match.group(1)
            if raw_value.strip() != expected_value:
                continue
            leading = len(raw_value) - len(raw_value.lstrip())
            trailing = len(raw_value) - len(raw_value.rstrip())
            start, end = match.span(1)
            return start + leading, end - trailing
        return None

    @staticmethod
    def _raw_action(text: str, action_type: str) -> str:
        if action_type in {"search", "answer"}:
            tag = "search" if action_type == "search" else "answer"
            matches = list(
                re.finditer(rf"<{tag}>.*?</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
            )
            if matches:
                return matches[-1].group(0)
        if action_type == "search":
            match = re.search(
                r"<tool_call>.*?</tool_call>", text, flags=re.IGNORECASE | re.DOTALL
            )
            if match:
                return match.group(0)
        return text

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
