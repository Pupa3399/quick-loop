from __future__ import annotations

import time

import httpx

from ouro_search.search.types import SearchResponse


class HttpSearchClient:
    """HTTP-only retriever client; it has no dependency on a backend implementation."""

    def __init__(
        self,
        base_url: str,
        *,
        top_k: int = 3,
        timeout_seconds: float = 30.0,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.base_url = base_url.rstrip("/")
        self.top_k = top_k
        self.timeout_seconds = timeout_seconds

    def search(self, query: str) -> SearchResponse:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query cannot be empty")
        started = time.perf_counter()
        response = httpx.post(
            f"{self.base_url}/search",
            json={"query": normalized_query, "top_k": self.top_k},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        parsed = SearchResponse.from_dict(payload)
        return SearchResponse(
            query=parsed.query,
            documents=parsed.documents,
            latency_seconds=time.perf_counter() - started,
            raw_json=payload,
        )
