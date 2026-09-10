from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    title: str
    text: str
    score: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Document:
        return cls(
            id=str(value["id"]),
            title=str(value["title"]),
            text=str(value["text"]),
            score=float(value["score"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    documents: list[Document]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SearchResponse:
        return cls(
            query=str(value["query"]),
            documents=[Document.from_dict(document) for document in value["documents"]],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "documents": [document.to_dict() for document in self.documents],
        }
