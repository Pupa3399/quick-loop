from ouro_search.agent.parser import ActionType, parse_agent_output


def test_parse_complete_search() -> None:
    parsed = parse_agent_output("<think>Need evidence.</think>\n<search>capital of France</search>")

    assert parsed.action is ActionType.SEARCH
    assert parsed.think == "Need evidence."
    assert parsed.query == "capital of France"


def test_parse_complete_answer() -> None:
    parsed = parse_agent_output("<think>I know it.</think><answer>Paris</answer>")

    assert parsed.action is ActionType.ANSWER
    assert parsed.answer == "Paris"


def test_missing_closing_tag_falls_back_to_end() -> None:
    parsed = parse_agent_output("<think>Need evidence.\n<search>capital of France")

    assert parsed.action is ActionType.SEARCH
    assert parsed.think == "Need evidence."
    assert parsed.query == "capital of France"


def test_latest_action_tag_wins() -> None:
    parsed = parse_agent_output("<search>old query</search><answer>final answer</answer>")

    assert parsed.action is ActionType.ANSWER
    assert parsed.answer == "final answer"


def test_plain_text_is_unknown() -> None:
    parsed = parse_agent_output("I cannot follow the requested protocol.")

    assert parsed.action is ActionType.UNKNOWN
    assert parsed.raw_output == "I cannot follow the requested protocol."
