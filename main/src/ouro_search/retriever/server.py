from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from ouro_search.retriever.mock_backend import search_mock_documents


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query cannot be blank")
        return value


class DocumentResponse(BaseModel):
    id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    documents: list[DocumentResponse]


app = FastAPI(title="Ouro Search Mock Retriever", version="0.1.0")


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    query = request.query
    documents = search_mock_documents(query, request.top_k)
    return SearchResponse(query=query, documents=documents)
