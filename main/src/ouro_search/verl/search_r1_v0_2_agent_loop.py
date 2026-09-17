from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

import httpx
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

from ouro_search.agent.parser import ActionType
from ouro_search.agent.search_r1_v0_2 import (
    INVALID_ACTION_OBSERVATION,
    build_prompt,
    format_observation,
    initial_context,
    official_sampling_params,
    postprocess_generation,
    render_search_result,
    strict_parse_action,
    truncate_observation,
    update_rolling_context,
)
from ouro_search.rewards.answer_reward import answer_exact_match
from ouro_search.search.types import SearchResponse
from ouro_search.trajectory.schema import Trajectory, TurnRecord
from ouro_search.trajectory.writer import JsonlTrajectoryWriter


class SearchR1V02AgentLoop(AgentLoopBase):
    """Search-R1 v0.2 rollout semantics adapted to veRL's AgentLoop API."""

    @classmethod
    def init_class(
        cls,
        config: Any,
        tokenizer: Any,
        processor: Any,
        *,
        retriever_url: str,
        top_k: int = 3,
        max_turns: int = 4,
        max_start_length: int = 2048,
        max_prompt_length: int = 4096,
        max_response_length: int = 500,
        max_obs_length: int = 500,
        max_accumulated_response_length: int = 4096,
        trajectory_dir: str = "main/data/trajectories/search_grpo",
        require_official_prompt: bool = True,
        **kwargs: Any,
    ) -> None:
        del config, processor, kwargs
        if cls._class_initialized:
            return
        cls._class_initialized = True
        cls.tokenizer = tokenizer
        cls.retriever_url = retriever_url
        cls.top_k = top_k
        cls.max_turns = max_turns
        cls.max_start_length = max_start_length
        cls.max_prompt_length = max_prompt_length
        cls.max_response_length = max_response_length
        cls.max_obs_length = max_obs_length
        cls.max_accumulated_response_length = max_accumulated_response_length
        cls.trajectory_writer = JsonlTrajectoryWriter(trajectory_dir)
        cls.require_official_prompt = require_official_prompt

    async def _search(self, query: str) -> SearchResponse:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.retriever_url,
                json={"query": query, "top_k": self.top_k},
            )
            response.raise_for_status()
            return SearchResponse.from_dict(response.json())

    def _prompt_ids(
        self, raw_prompt: list[dict[str, Any]], question: str
    ) -> tuple[list[int], list[int]]:
        if self.require_official_prompt:
            expected = build_prompt(question)
            actual = raw_prompt[-1].get("content", "") if raw_prompt else ""
            if actual != expected:
                raise ValueError(
                    "Search-R1 v0.2 baseline requires the official prompt text; "
                    "regenerate the dataset with ouro_search.data.searchr1."
                )
        ids = list(
            self.tokenizer.apply_chat_template(
                raw_prompt,
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        original_left_ids = initial_context(ids, self.max_start_length)
        rolling_ids = ids[-self.max_prompt_length :]
        return original_left_ids, rolling_ids

    def _encode_observation(self, text: str) -> tuple[str, list[int]]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        ids = truncate_observation(ids, self.max_obs_length)
        return self.tokenizer.decode(ids, skip_special_tokens=False), ids

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        question = str(kwargs.get("question", ""))
        raw_prompt = [dict(message) for message in kwargs["raw_prompt"]]
        original_prompt_ids, rolling_ids = self._prompt_ids(raw_prompt, question)
        response_ids: list[int] = []
        response_mask: list[int] = []
        raw_generations: list[str] = []
        processed_generations: list[str] = []
        queries: list[str] = []
        observation_blocks: list[str] = []
        turns: list[TurnRecord] = []
        valid_action_count = 0
        valid_search_count = 0
        request_id = uuid4().hex
        generation_seconds = 0.0
        search_seconds = 0.0
        termination_reason = "max_turns"

        params = official_sampling_params(
            sampling_params,
            max_response_length=self.max_response_length,
        )

        async def generate_once() -> tuple[str, Any]:
            nonlocal generation_seconds
            started = time.perf_counter()
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=rolling_ids,
                sampling_params=params,
            )
            generation_seconds += time.perf_counter() - started
            processed = postprocess_generation(list(output.token_ids), self.tokenizer)
            raw_generations.append(processed.raw_text)
            processed_generations.append(processed.text)
            response_ids.extend(processed.token_ids)
            response_mask.extend([1] * len(processed.token_ids))
            return processed.text, processed

        active = True
        for _ in range(self.max_turns):
            generated_text, processed = await generate_once()
            action = strict_parse_action(generated_text)
            if action.action is ActionType.ANSWER:
                valid_action_count += 1
                termination_reason = "answer"
                active = False
                break

            if action.action is ActionType.SEARCH:
                valid_action_count += 1
                valid_search_count += 1
                queries.append(action.query or "")
                started = time.perf_counter()
                search_response = await self._search(action.query or "")
                search_seconds += time.perf_counter() - started
                observation = format_observation(render_search_result(search_response))
                turns.append(
                    TurnRecord(
                        turn_id=len(turns),
                        loop_steps=3,
                        think=action.think,
                        query=action.query or "",
                        information=observation,
                        retrieved_documents=search_response.documents,
                        model_output=generated_text,
                    )
                )
            else:
                observation = INVALID_ACTION_OBSERVATION

            observation, observation_ids = self._encode_observation(observation)
            observation_blocks.append(observation)
            response_ids.extend(observation_ids)
            response_mask.extend([0] * len(observation_ids))
            rolling_ids = update_rolling_context(
                rolling_ids,
                processed.token_ids,
                observation_ids,
                self.max_prompt_length,
            )

        if active:
            generated_text, _ = await generate_once()
            final_action = strict_parse_action(generated_text)
            if final_action.action is not ActionType.UNKNOWN:
                valid_action_count += 1
            if final_action.action is ActionType.SEARCH:
                valid_search_count += 1
            if final_action.action is ActionType.ANSWER:
                termination_reason = "answer"

        response_ids = response_ids[: self.max_accumulated_response_length]
        response_mask = response_mask[: self.max_accumulated_response_length]
        final_output = processed_generations[-1] if processed_generations else ""
        final_action = strict_parse_action(final_output)
        prediction = final_action.answer or final_output
        reward_model = kwargs.get("reward_model", {})
        ground_truth = (
            reward_model.get("ground_truth", {})
            if isinstance(reward_model, dict)
            else {}
        )
        aliases = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
        if isinstance(aliases, str):
            aliases = [aliases]
        aliases = [str(alias) for alias in aliases]
        extra_info = kwargs.get("extra_info", {})
        if isinstance(extra_info, dict):
            extra_info["num_search_turns"] = len(turns)
        sample_id = str(
            kwargs.get("id")
            or (extra_info.get("id") if isinstance(extra_info, dict) else "")
            or request_id
        )
        trajectory = Trajectory(
            id=f"{sample_id}-{request_id}",
            question=question,
            prediction=prediction,
            reference_answer=str(kwargs.get("reference_answer", aliases[0] if aliases else "")),
            answer_aliases=aliases,
            reward=float(
                final_action.answer is not None
                and answer_exact_match(final_action.answer, aliases)
            ),
            num_search_turns=len(turns),
            turns=turns,
            termination_reason=termination_reason,
            final_model_output=final_output,
            raw_generations=raw_generations,
            response_token_count=len(response_ids),
        )
        self.trajectory_writer.write(trajectory)
        return AgentLoopOutput(
            prompt_ids=original_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            num_turns=len(processed_generations),
            metrics=AgentLoopMetrics(
                generate_sequences=generation_seconds,
                tool_calls=search_seconds,
            ),
            extra_fields={
                "model_outputs": processed_generations,
                "raw_generations": raw_generations,
                "queries": queries,
                "information": observation_blocks,
                "num_search_turns": len(queries),
                "tool_call_counts": len(queries),
                "valid_action_count": valid_action_count,
                "valid_search_count": valid_search_count,
                "format_valid": int(final_action.action is ActionType.ANSWER),
                "termination_reason": termination_reason,
                "rollout_semantics": "search_r1_v0_2_official",
            },
        )
