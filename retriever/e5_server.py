from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator

from retriever.e5_backend import E5RetrieverConfig, E5Wiki18Retriever


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


class BatchSearchRequest(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=512)
    top_k: int = Field(default=3, ge=1, le=100)

    @field_validator("queries")
    @classmethod
    def queries_must_not_be_blank(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("queries cannot contain blank values")
        return normalized


@lru_cache(maxsize=1)
def backend() -> E5Wiki18Retriever:
    data_dir = Path(os.environ.get("OURO_WIKI18_DIR", "/data2/wuguanting/quick_loop/data/wiki18"))
    return E5Wiki18Retriever(
        E5RetrieverConfig(
            model_name=os.environ.get("OURO_RETRIEVER_MODEL", "intfloat/e5-base-v2"),
            model_revision=os.environ.get(
                "OURO_RETRIEVER_REVISION",
                "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
            ),
            index_path=Path(os.environ.get("OURO_RETRIEVER_INDEX", data_dir / "e5_Flat.index")),
            corpus_path=Path(
                os.environ.get("OURO_RETRIEVER_CORPUS", data_dir / "wiki-18.jsonl")
            ),
            device=os.environ.get("OURO_RETRIEVER_DEVICE", "cuda:0"),
            dtype=os.environ.get("OURO_RETRIEVER_DTYPE", "float16"),
            faiss_gpu=os.environ.get("OURO_FAISS_GPU", "0") == "1",
            cache_dir=os.environ.get("HF_HOME"),
        )
    )


app = FastAPI(title="Ouro E5 Wiki-2018 Retriever", version="0.2.0")


@app.post("/search")
def search(request: SearchRequest) -> dict[str, object]:
    return {"query": request.query, "documents": backend().search(request.query, request.top_k)}


@app.post("/search/batch")
def search_batch(request: BatchSearchRequest) -> dict[str, object]:
    documents = backend().search_batch(request.queries, request.top_k)
    return {
        "results": [
            {"query": query, "documents": single_documents}
            for query, single_documents in zip(request.queries, documents, strict=True)
        ]
    }


@app.get("/health")
def health() -> dict[str, object]:
    instance = backend()
    return {"status": "ok", "index_size": int(instance.index.ntotal)}
