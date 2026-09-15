from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ouro_search.search.types import Document


@dataclass(slots=True)
class TurnRecord:
    turn_id: int
    loop_steps: int
    think: str
    query: str
    information: str
    retrieved_documents: list[Document] = field(default_factory=list)
    model_output: str = ""
    generation_id: int | None = None
    retriever_latency_seconds: float | None = None
    retriever_raw_json: dict[str, Any] = field(default_factory=dict)
    observation_token_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["retrieved_documents"] = [
            {"rank": rank, **document.to_dict()}
            for rank, document in enumerate(self.retrieved_documents, start=1)
        ]
        return value


@dataclass(slots=True)
class GenerationRecord:
    turn_id: int
    loop_steps: int
    messages: list[dict[str, Any]]
    prompt_continuation: str
    rendered_prompt: str
    prompt_token_ids: list[int]
    raw_generation: str
    completion_token_ids: list[int]
    think: str
    action_type: str
    raw_action: str
    parsed_query: str | None
    parser_valid: bool
    format_valid: bool
    explicit_format_valid: bool
    malformed_reason: str
    finish_reason: str | None
    stop_reason: str | int | None
    protocol_stop: bool
    generation_latency_seconds: float
    temperature: float
    top_p: float
    seed: int | None
    requested_max_new_tokens: int
    context_tokens_remaining: int | None
    think_character_span: list[int] | None = None
    think_token_span: list[int] | None = None
    query_character_span: list[int] | None = None
    query_token_span: list[int] | None = None
    query_prefix: str = ""
    context_snapshot: str = ""
    chosen_token_logprobs: list[float | None] = field(default_factory=list)
    span_top_logprobs: list[dict[str, Any]] = field(default_factory=list)
    retriever_called: bool = False
    retriever_query: str | None = None
    retriever_latency_seconds: float | None = None
    retriever_raw_json: dict[str, Any] = field(default_factory=dict)
    retrieved_documents: list[Document] = field(default_factory=list)
    observation_text: str = ""
    observation_token_ids: list[int] = field(default_factory=list)

    @property
    def prompt_token_count(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def completion_token_count(self) -> int:
        return len(self.completion_token_ids)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["prompt_token_count"] = self.prompt_token_count
        value["completion_token_count"] = self.completion_token_count
        value["observation_token_count"] = len(self.observation_token_ids)
        value["retrieved_documents"] = [
            {"rank": rank, **document.to_dict()}
            for rank, document in enumerate(self.retrieved_documents, start=1)
        ]
        return value


@dataclass(slots=True)
class Trajectory:
    id: str
    question: str
    prediction: str
    num_search_turns: int
    reference_answer: str = ""
    answer_aliases: list[str] = field(default_factory=list)
    reward: float | None = None
    turns: list[TurnRecord] = field(default_factory=list)
    termination_reason: str = ""
    final_model_output: str = ""
    raw_generations: list[str] = field(default_factory=list)
    response_token_count: int = 0
    prompt_profile: str = ""
    initial_messages: list[dict[str, Any]] = field(default_factory=list)
    initial_rendered_prompt: str = ""
    generations: list[GenerationRecord] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_seconds: float = 0.0
    final_finish_reason: str | None = None
    final_stop_reason: str | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "sample_id": self.id,
            "question": self.question,
            "prediction": self.prediction,
            "reference_answer": self.reference_answer,
            "answer_aliases": self.answer_aliases,
            "reward": self.reward,
            "num_search_turns": self.num_search_turns,
            "turns": [turn.to_dict() for turn in self.turns],
            "termination_reason": self.termination_reason,
            "final_model_output": self.final_model_output,
            "raw_generations": self.raw_generations,
            "response_token_count": self.response_token_count,
            "prompt_profile": self.prompt_profile,
            "initial_messages": self.initial_messages,
            "initial_rendered_prompt": self.initial_rendered_prompt,
            "generations": [generation.to_dict() for generation in self.generations],
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_latency_seconds": self.total_latency_seconds,
            "final_finish_reason": self.final_finish_reason,
            "final_stop_reason": self.final_stop_reason,
        }
