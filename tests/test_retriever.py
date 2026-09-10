import httpx
import pytest

from retriever import e5_server
from retriever.server import app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_mock_search_contract() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/search", json={"query": "Ouro model", "top_k": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "Ouro model"
    assert len(payload["documents"]) == 2
    assert payload["documents"][0] == {
        "id": "mock-0",
        "title": "Mock Document",
        "text": "Mock evidence for query: Ouro model",
        "score": 1.0,
    }


@pytest.mark.anyio
async def test_mock_search_rejects_blank_query() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/search", json={"query": "   "})

    assert response.status_code == 422


def test_e5_search_contract_without_loading_artifacts(monkeypatch) -> None:
    class FakeBackend:
        config = type("Config", (), {"faiss_gpu": True})()
        index = type("GpuIndexFlat", (), {"ntotal": 21_015_324})()
        faiss_index_device = "cuda:0"
        faiss_index_precision = "float32"
        faiss_gpu_memory_bytes = 64_000_000_000

        def search(self, query: str, top_k: int) -> list[dict[str, object]]:
            return [
                {
                    "id": "wiki-1",
                    "title": "Hamlet",
                    "text": "Hamlet was written by William Shakespeare.",
                    "score": 0.9,
                    "rank": 0,
                }
            ][:top_k]

        def search_batch(
            self, queries: list[str], top_k: int
        ) -> list[list[dict[str, object]]]:
            return [self.search(query, top_k) for query in queries]

    monkeypatch.setattr(e5_server, "backend", FakeBackend)
    payload = e5_server.search(
        e5_server.SearchRequest(query="Who wrote Hamlet?", top_k=3)
    )
    batch_payload = e5_server.search_batch(
        e5_server.BatchSearchRequest(
            queries=["Who wrote Hamlet?", "Hamlet author"], top_k=3
        )
    )

    assert payload["query"] == "Who wrote Hamlet?"
    assert payload["documents"][0]["title"] == "Hamlet"
    assert len(batch_payload["results"]) == 2

    health = e5_server.health()
    assert health["faiss_gpu"] is True
    assert health["faiss_index_device"] == "cuda:0"
    assert health["index_size"] == 21_015_324
