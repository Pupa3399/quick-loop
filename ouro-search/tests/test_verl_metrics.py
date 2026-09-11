from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("verl") is None, reason="veRL extra not installed"
)


def test_verl_metrics_report_response_distribution_and_information_mask() -> None:
    from verl import DataProto
    from verl.trainer.ppo.metric_utils import compute_data_metrics

    batch = DataProto.from_dict(
        tensors={
            "responses": torch.ones(2, 4, dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.long
            ),
            "response_mask": torch.tensor(
                [[1, 0, 0, 0], [1, 0, 1, 1]], dtype=torch.long
            ),
            "token_level_scores": torch.tensor(
                [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            ),
            "token_level_rewards": torch.tensor(
                [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            ),
            "advantages": torch.ones(2, 4),
            "returns": torch.ones(2, 4),
        },
        non_tensors={
            "num_search_turns": np.array([1, 0]),
            "format_valid": np.array([1, 0]),
        },
    )

    metrics = compute_data_metrics(batch, use_critic=False)

    assert metrics["response_length/mean"] == pytest.approx(3.5)
    assert metrics["response_length/p50"] == pytest.approx(3.5)
    assert metrics["response_length/p95"] == pytest.approx(3.95)
    assert metrics["information_length/mean"] == pytest.approx(1.5)
    assert metrics["information_length/max"] == pytest.approx(2.0)
    assert metrics["search/action_rate"] == pytest.approx(0.5)
    assert metrics["search/turns_mean"] == pytest.approx(0.5)
    assert metrics["outcome/format_valid_rate"] == pytest.approx(0.5)
