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
from ouro_search.agent.profiles import PromptProfile, get_prompt_profile
from ouro_search.rewards.answer_reward import answer_exact_match
from ouro_search.search.types import SearchResponse
from ouro_search.trajectory.schema import Trajectory, TurnRecord
from ouro_search.trajectory.writer import JsonlTrajectoryWriter


class OuroSearchAgentLoop(AgentLoopBase):
    """Tag-based Search-R1 loop backed by veRL's asynchronous vLLM servers."""

    @classmethod
    def init_class(
        cls,
        config: Any,
        tokenizer: Any,
        processor: Any,
        *,
        retriever_url: str,
        top_k: int = 3,
        max_search_actions: int = 4,
        max_generation_tokens: int = 500,
        max_information_tokens: int = 500,
        trajectory_dir: str = "outputs/trajectories/verl",
        prompt_profile: str = "search_r1",
        **kwargs: Any,
    ) -> None:
        del processor, kwargs
        if cls._class_initialized:
            return
        cls._class_initialized = True
        cls.tokenizer = tokenizer
        cls.retriever_url = retriever_url
        cls.top_k = top_k
        cls.max_search_actions = max_search_actions
        cls.max_generation_tokens = max_generation_tokens
        cls.max_information_tokens = max_information_tokens
        cls.trajectory_writer = JsonlTrajectoryWriter(trajectory_dir)
        cls.prompt_profile: PromptProfile = get_prompt_profile(prompt_profile)
        cls.response_length = config.actor_rollout_ref.rollout.response_length

    async def _search(self, query: str) -> SearchResponse:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.retriever_url,
                json={"query": query, "top_k": self.top_k},
            )
            response.raise_for_status()
            return SearchResponse.from_dict(response.json())

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        raw_prompt = list(kwargs["raw_prompt"])
        prompt_ids = self.tokenizer.apply_chat_template(
            raw_prompt, add_generation_prompt=True, tokenize=True
        )
        original_prompt_ids = list(prompt_ids)
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] | None = []
        model_outputs: list[str] = []
        queries: list[str] = []
        information_blocks: list[str] = []
        turns: list[TurnRecord] = []
        request_id = uuid4().hex
        generation_seconds = 0.0
        search_seconds = 0.0
        termination_reason = "max_search_actions"

        sampling_params = dict(sampling_params)
        sampling_params.update(
            max_tokens=min(
                int(sampling_params.get("max_tokens", self.response_length)),
                self.max_generation_tokens,
            ),
            stop=list(self.prompt_profile.stop_sequences),
            include_stop_str_in_output=True,
        )
        for turn_id in range(self.max_search_actions + 1):
            started = time.perf_counter()
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
            generation_seconds += time.perf_counter() - started
            generated_ids = output.token_ids
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            model_outputs.append(generated_text)
            prompt_ids.extend(generated_ids)
            response_ids.extend(generated_ids)
            response_mask.extend([1] * len(generated_ids))
            if response_logprobs is not None:
                if output.log_probs:
                    response_logprobs.extend(output.log_probs)
                else:
                    response_logprobs = None

            action = self.prompt_profile.parse_action(generated_text)
            if action.action is ActionType.ANSWER:
                termination_reason = "answer"
                break
            if action.action is not ActionType.SEARCH:
                termination_reason = "unparseable_output"
                break
            if turn_id == self.max_search_actions:
                break

            queries.append(action.query or "")
            started = time.perf_counter()
            search_result = await self._search(action.query or "")
            search_seconds += time.perf_counter() - started
            information = self.prompt_profile.format_observation(search_result)
            information_ids = self.tokenizer(
                information, add_special_tokens=False
            )["input_ids"][: self.max_information_tokens]
            information = self.tokenizer.decode(information_ids, skip_special_tokens=False)
            information_blocks.append(information)
            turns.append(
                TurnRecord(
                    turn_id=len(turns),
                    loop_steps=3,
                    think=action.think,
                    query=action.query or "",
                    information=information,
                    retrieved_documents=search_result.documents,
                    model_output=generated_text,
                )
            )
            prompt_ids.extend(information_ids)
            response_ids.extend(information_ids)
            response_mask.extend([0] * len(information_ids))
            if response_logprobs is not None:
                response_logprobs.extend([0.0] * len(information_ids))
            if len(response_ids) >= self.response_length:
                termination_reason = "max_response_length"
                break

        response_ids = response_ids[: self.response_length]
        response_mask = response_mask[: self.response_length]
        if response_logprobs is not None:
            response_logprobs = response_logprobs[: self.response_length]
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
        final_output = model_outputs[-1] if model_outputs else ""
        final_action = self.prompt_profile.parse_action(final_output)
        prediction = final_action.answer or final_output
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
            question=str(kwargs.get("question", "")),
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
            raw_generations=model_outputs,
            response_token_count=len(response_ids),
        )
        self.trajectory_writer.write(trajectory)
        return AgentLoopOutput(
            prompt_ids=original_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            num_turns=len(model_outputs),
            metrics=AgentLoopMetrics(
                generate_sequences=generation_seconds,
                tool_calls=search_seconds,
            ),
            extra_fields={
                "model_outputs": model_outputs,
                "queries": queries,
                "information": information_blocks,
                "num_search_turns": len(queries),
                "tool_call_counts": len(queries),
                "format_valid": int(final_action.action is ActionType.ANSWER),
                "termination_reason": termination_reason,
            },
        )
