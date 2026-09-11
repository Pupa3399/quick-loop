from verl.workers.rollout.vllm_rollout.vllm_async_server import _make_sampling_params


def test_sampling_params_honor_the_smaller_per_turn_limit() -> None:
    params = _make_sampling_params(3000, {"max_tokens": 500, "temperature": 1.0})

    assert params.max_tokens == 500


def test_sampling_params_cap_requested_tokens_to_context() -> None:
    params = _make_sampling_params(400, {"max_tokens": 500})

    assert params.max_tokens == 400
