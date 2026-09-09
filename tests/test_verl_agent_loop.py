from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouro_search.search.types import Document, SearchResponse

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("verl") is None, reason="veRL extra not installed"
)


class FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs) -> list[int]:
        del args, kwargs
        return [1, 2]

    def decode(self, token_ids: list[int], **kwargs) -> str:
        del kwargs
        if token_ids == [10]:
            return "<think>need evidence</think><search>Hamlet author</search>"
        if token_ids == [11]:
            return "<think>use evidence</think><answer>William Shakespeare</answer>"
        if token_ids == [20]:
            return "<information>Hamlet was written by William Shakespeare.</information>"
        raise AssertionError(f"Unexpected token IDs: {token_ids}")

    def __call__(self, text: str, **kwargs) -> dict[str, list[int]]:
        del text, kwargs
        return {"input_ids": [20]}


class FakeServerManager:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, **kwargs) -> SimpleNamespace:
        del kwargs
        self.calls += 1
        token_id = 10 if self.calls == 1 else 11
        return SimpleNamespace(token_ids=[token_id], log_probs=[-0.1])


@pytest.mark.anyio
async def test_verl_agent_loop_masks_information_and_writes_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ouro_search.trajectory.writer import JsonlTrajectoryWriter
    from ouro_search.verl.agent_loop import OuroSearchAgentLoop

    loop = object.__new__(OuroSearchAgentLoop)
    loop.tokenizer = FakeTokenizer()
    loop.server_manager = FakeServerManager()
    loop.retriever_url = "http://unused/search"
    loop.top_k = 3
    loop.max_search_actions = 4
    loop.max_generation_tokens = 500
    loop.max_information_tokens = 500
    loop.response_length = 32
    loop.trajectory_writer = JsonlTrajectoryWriter(tmp_path)

    async def fake_search(query: str) -> SearchResponse:
        assert query == "Hamlet author"
        return SearchResponse(
            query=query,
            documents=[
                Document(
                    id="wiki-1",
                    title="Hamlet",
                    text="Hamlet was written by William Shakespeare.",
                    score=0.9,
                )
            ],
        )

    monkeypatch.setattr(loop, "_search", fake_search)
    extra_info = {"id": "nq-test-0"}
    output = await loop.run(
        {"max_tokens": 3000},
        raw_prompt=[{"role": "user", "content": "Who wrote Hamlet?"}],
        question="Who wrote Hamlet?",
        reference_answer="William Shakespeare",
        reward_model={"ground_truth": {"target": ["Shakespeare", "William Shakespeare"]}},
        extra_info=extra_info,
    )

    assert output.response_mask == [1, 0, 1]
    assert output.extra_fields["num_search_turns"] == 1
    assert output.extra_fields["format_valid"] == 1
    assert extra_info["num_search_turns"] == 1
    path = next(tmp_path.glob("*.jsonl"))
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    assert trajectory["reward"] == 1.0
    assert trajectory["turns"][0]["turn_id"] == 0
    assert trajectory["turns"][0]["loop_steps"] == 3
