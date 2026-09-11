import importlib.util
from pathlib import Path

import pytest

_MODULE_SPEC = importlib.util.spec_from_file_location(
    "benchmark_retriever",
    Path(__file__).parents[1] / "scripts" / "benchmark_retriever.py",
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_MODULE)
percentile = _MODULE.percentile
compare_results = _MODULE.compare_results


def test_percentile_interpolates_sorted_samples() -> None:
    samples = [0.4, 0.1, 0.3, 0.2]

    assert percentile(samples, 0.5) == pytest.approx(0.25)
    assert percentile(samples, 0.95) == pytest.approx(0.385)


def test_percentile_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        percentile([], 0.95)


def test_compare_results_reports_ordered_and_set_parity() -> None:
    reference = [{"query": "q", "documents": [{"id": "a"}, {"id": "b"}]}]
    reordered = [{"query": "q", "documents": [{"id": "b"}, {"id": "a"}]}]

    metrics = compare_results(reordered, reference)

    assert metrics == {
        "exact_top_k_match_rate": 0.0,
        "top_k_set_match_rate": 1.0,
        "mean_top_k_overlap_rate": 1.0,
    }
