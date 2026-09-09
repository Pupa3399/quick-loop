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

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["retrieved_documents"] = [document.to_dict() for document in self.retrieved_documents]
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "prediction": self.prediction,
            "reference_answer": self.reference_answer,
            "answer_aliases": self.answer_aliases,
            "reward": self.reward,
            "num_search_turns": self.num_search_turns,
            "turns": [turn.to_dict() for turn in self.turns],
            "termination_reason": self.termination_reason,
            "final_model_output": self.final_model_output,
        }
