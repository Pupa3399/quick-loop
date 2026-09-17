from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
from ouro_search.agent.parser import ActionType
from ouro_search.agent.search_r1_v0_2 import (
    INVALID_ACTION_OBSERVATION,
    SEARCH_R1_V0_2_PROMPT,
    format_observation,
    initial_context,
    official_sampling_params,
    postprocess_generation,
    strict_parse_action,
    update_rolling_context,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)


def test_official_prompt_uses_v0_2_search_limit() -> None:
    assert "You can search at most four times." in SEARCH_R1_V0_2_PROMPT
    assert "You can search as many times as your want." not in SEARCH_R1_V0_2_PROMPT
    assert SEARCH_R1_V0_2_PROMPT.endswith("Question: {question}\n")


def test_observation_has_official_text_and_token_boundaries() -> None:
    tokenizer = CharacterTokenizer()
    search_result = "Doc 1(Title: Hamlet) William Shakespeare\n"
    observation = format_observation(search_result)

    assert observation == (
        "\n\n<information>Doc 1(Title: Hamlet) William Shakespeare"
        "</information>\n\n"
    )
    assert tokenizer.decode(tokenizer.encode(observation)) == observation
    assert "\n</information>" not in observation


def test_strict_parser_matches_first_fully_closed_case_sensitive_action() -> None:
    parsed = strict_parse_action(
        "<think>x</think><answer> first </answer><search>later</search>"
    )
    assert parsed.action is ActionType.ANSWER
    assert parsed.answer == "first"

    assert strict_parse_action("<search>missing close").action is ActionType.UNKNOWN
    assert strict_parse_action("<SEARCH>wrong case</SEARCH>").action is ActionType.UNKNOWN


def test_postprocess_runs_after_generation_and_prioritizes_search_close() -> None:
    tokenizer = CharacterTokenizer()
    raw = "<answer>early</answer>tail<search>query</search>ignored"
    processed = postprocess_generation(tokenizer.encode(raw), tokenizer)

    assert processed.raw_text == raw
    assert processed.text == "<answer>early</answer>tail<search>query</search>"
    assert processed.token_ids == tokenizer.encode(processed.text)


def test_invalid_action_recovery_matches_official_text() -> None:
    assert INVALID_ACTION_OBSERVATION == (
        "\nMy previous action is invalid. If I want to search, I should put the query "
        "between <search> and </search>. If I want to give the final answer, I should "
        "put the answer between <answer> and </answer>. Let me try again.\n"
    )
    assert strict_parse_action("plain text").action is ActionType.UNKNOWN


def test_rolling_context_keeps_the_latest_official_window() -> None:
    assert initial_context(list(range(8)), max_start_length=3) == [5, 6, 7]
    result = update_rolling_context(
        list(range(6)),
        [6, 7, 8],
        [9, 10],
        max_prompt_length=5,
    )
    assert result == [6, 7, 8, 9, 10]


def test_sampling_params_have_no_closing_stop_or_per_turn_seed() -> None:
    original = {
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 123,
        "stop": ["</search>", "</answer>"],
        "include_stop_str_in_output": True,
    }
    trajectory_params = official_sampling_params(original, max_response_length=500)

    assert trajectory_params == {"temperature": 1.0, "top_p": 1.0, "max_tokens": 500}
    assert "seed" not in trajectory_params
    assert "stop" not in trajectory_params
    # The AgentLoop constructs this once before its turn loop and reuses it unchanged.
    assert official_sampling_params(original, max_response_length=500) == trajectory_params


def test_official_config_and_agent_loop_config_do_not_drift() -> None:
    baseline = OmegaConf.load(PROJECT_ROOT / "configs/search_r1/v0_2.yaml")
    agent_loop = OmegaConf.load(
        PROJECT_ROOT / "main/eval/search_grpo/agent_loop_search_r1_v0_2.yaml"
    )[0]

    assert baseline.prompt.template == SEARCH_R1_V0_2_PROMPT
    assert baseline.rollout.stop_strings == []
    assert agent_loop.max_turns == baseline.rollout.max_turns
    assert agent_loop.max_start_length == baseline.rollout.max_start_length
    assert agent_loop.max_prompt_length == baseline.rollout.max_prompt_length
    assert agent_loop.max_response_length == baseline.rollout.max_response_length
    assert agent_loop.max_obs_length == baseline.rollout.max_obs_length


def test_search_r1_models_are_default_and_isolated_from_ouro() -> None:
    experiment_config = OmegaConf.load(PROJECT_ROOT / "main/eval/search_grpo/config.yaml")
    defaults = OmegaConf.to_container(experiment_config.defaults)
    assert {"experiment_model": "searchr1_3b"} in defaults

    model_dir = PROJECT_ROOT / "main/eval/search_grpo/experiment_model"
    searchr1_3b = OmegaConf.load(model_dir / "searchr1_3b.yaml")
    searchr1_7b = OmegaConf.load(model_dir / "searchr1_7b.yaml")
    ouro = OmegaConf.load(model_dir / "ouro_r3.yaml")
    local_override = OmegaConf.load(PROJECT_ROOT / "configs/search_r1/local_override.yaml")

    for model in (searchr1_3b, searchr1_7b):
        assert model.family == "search_r1"
        assert model.trust_remote_code is False
        assert OmegaConf.to_container(model.override_config) == {}
        assert model.external_lib is None
        assert model.requires_ouro_plugin is False
        assert Path(PROJECT_ROOT / model.path).is_dir()

    assert searchr1_3b.path != searchr1_7b.path
    assert searchr1_3b.checkpoint_dir != searchr1_7b.checkpoint_dir
    assert searchr1_3b.trajectory_dir != searchr1_7b.trajectory_dir
    assert ouro.family == "ouro"
    assert ouro.override_config.total_ut_steps == 3
    assert ouro.requires_ouro_plugin is True
    assert "model" not in local_override
