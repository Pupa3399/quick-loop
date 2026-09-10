from __future__ import annotations

from ouro_search.agent.protocol import build_agent_prompt
from ouro_search.inference.vllm_engine import VllmEngine
from ouro_search.search.types import Document
from ouro_search.trajectory.schema import TurnRecord


class FakeTokenizer:
    chat_template = "present"

    def apply_chat_template(self, messages, **kwargs) -> str:
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        assert len(messages) == 1
        return f"CHAT[{messages[0]['content']}]ASSISTANT:"


def test_search_history_continues_after_single_chat_template() -> None:
    turn = TurnRecord(
        turn_id=0,
        loop_steps=3,
        think="need evidence",
        query="Hamlet author",
        information="<information>William Shakespeare</information>",
        retrieved_documents=[Document(id="1", title="Hamlet", text="text", score=1.0)],
        model_output="<think>need evidence</think><search>Hamlet author</search>",
    )
    engine = object.__new__(VllmEngine)
    engine.tokenizer = FakeTokenizer()
    engine.use_chat_template = True

    rendered = engine._format_prompt(build_agent_prompt("Who wrote Hamlet?", [turn]))

    assert rendered.count("CHAT[") == 1
    assert "Question: Who wrote Hamlet?" in rendered
    assert rendered.index("ASSISTANT:") < rendered.index("<search>Hamlet author</search>")
    assert rendered.endswith("<information>William Shakespeare</information>\n")


def test_vllm_engine_rejects_wrong_ouro_depth() -> None:
    engine = object.__new__(VllmEngine)
    engine.fixed_loop_steps = 3

    try:
        engine.generate("prompt", loop_steps=4)
    except ValueError as exc:
        assert "fixed at R3" in str(exc)
    else:
        raise AssertionError("wrong recurrent depth was accepted")
