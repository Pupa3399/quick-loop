from __future__ import annotations


def search_mock_documents(query: str, top_k: int) -> list[dict[str, object]]:
    return [
        {
            "id": f"mock-{index}",
            "title": "Mock Document",
            "text": f"Mock evidence for query: {query}",
            "score": 1.0 if index == 0 else round(1.0 / (index + 1), 6),
        }
        for index in range(top_k)
    ]
