import httpx
import pytest

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
