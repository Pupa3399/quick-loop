from __future__ import annotations

from ouro_search.agent.parser import ActionType
from ouro_search.agent.profiles import available_prompt_profiles, get_prompt_profile
from ouro_search.search.types import Document, SearchResponse
from ouro_search.trajectory.schema import TurnRecord


def main() -> None:
    response = SearchResponse(
        query="Hamlet author",
        documents=[
            Document(
                id="wiki-1",
                title="Hamlet",
                text="Hamlet was written by William Shakespeare.",
                score=1.0,
            )
        ],
    )
    action_text = {
        "search_r1": "<think>Need evidence.</think><search>Hamlet author</search>",
        "hermes": (
            '<tool_call>{"arguments": {"query": "Hamlet author"}, '
            '"name": "search"}</tool_call>'
        ),
    }
    final_text = {
        "search_r1": "<think>Done.</think><answer>William Shakespeare</answer>",
        "hermes": "William Shakespeare",
    }

    for name in available_prompt_profiles():
        profile = get_prompt_profile(name)
        action = profile.parse_action(action_text[name])
        assert action.action is ActionType.SEARCH
        observation = profile.format_observation(response)
        turn = TurnRecord(
            turn_id=0,
            loop_steps=3,
            think=action.think,
            query=action.query or "",
            information=observation,
            retrieved_documents=response.documents,
            model_output=action_text[name],
        )
        prompt = profile.build_prompt("Who wrote Hamlet?", [turn])
        assert observation in prompt.continuation or any(
            message.get("content") == observation for message in prompt.messages
        )
        assert profile.parse_final_answer(final_text[name]) == "William Shakespeare"
        print(
            f"{name}: messages={len(prompt.messages)}, "
            f"continuation={bool(prompt.continuation)}, action=search, final=ok"
        )


if __name__ == "__main__":
    main()
