from ouro_search.rewards.answer_reward import (
    answer_exact_match,
    compute_answer_reward,
    extract_final_answer,
    normalize_answer,
)


def test_normalization_matches_search_r1() -> None:
    assert normalize_answer(" The, Eiffel Tower! ") == "eiffel tower"


def test_multiple_aliases_and_final_answer_only() -> None:
    aliases = ["Paris", "City of Paris"]
    assert answer_exact_match("the Paris", aliases)
    assert compute_answer_reward("<think>x</think><answer>The Paris</answer>", aliases) == 1.0
    assert compute_answer_reward("Paris", aliases) == 0.0
    assert compute_answer_reward("<search>Paris</search><answer>London</answer>", aliases) == 0.0


def test_last_answer_wins_and_unclosed_answer_falls_back() -> None:
    assert extract_final_answer("<answer>wrong</answer><answer>right</answer>") == "right"
    assert extract_final_answer("<answer>fallback") == "fallback"
