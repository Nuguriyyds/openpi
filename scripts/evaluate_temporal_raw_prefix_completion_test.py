from __future__ import annotations

import dataclasses

import numpy as np

from scripts import evaluate_temporal_raw_prefix_completion as _evaluate


@dataclasses.dataclass(frozen=True)
class _Row:
    trajectory_id: str
    task_index: int
    boundary_tick: int
    label: int
    sample_kind: str
    window_variant: str


def test_variant_metrics_and_endpoint_hard_pair_ignore_terminal_positive_overwrite():
    rows = (
        _Row("trajectory-0", 0, 100, 1, "positive", "endpoint_positive"),
        _Row("trajectory-0", 0, 100, 1, "positive", "terminal_one_hold_positive"),
        _Row("trajectory-0", 0, 100, 1, "positive", "terminal_full_hold_positive"),
        _Row("trajectory-0", 0, 100, 0, "hard_negative", "hard_negative"),
        _Row("trajectory-0", 0, 100, 0, "ordinary_negative", "ordinary_negative"),
        _Row("trajectory-0", 0, 100, 0, "ordinary_negative", "start_0_negative"),
        _Row("trajectory-0", 0, 100, 0, "ordinary_negative", "start_15_negative"),
    )
    logits = np.asarray([2.0, 8.0, 9.0, 0.0, -1.0, -2.0, -3.0], dtype=np.float64)

    metrics = _evaluate._metrics(rows, logits)  # noqa: SLF001
    variants = _evaluate._variant_metrics(rows, _evaluate.temporal_metrics.stable_sigmoid(logits))  # noqa: SLF001

    expected_margin = float(_evaluate.temporal_metrics.stable_sigmoid(np.asarray([2.0]))[0]) - 0.5
    assert metrics["positive_hard_paired_ordering_accuracy"] == 1.0
    assert np.isclose(metrics["positive_hard_margin_mean"], expected_margin)
    assert variants["endpoint_positive"]["count"] == 1.0
    assert np.isclose(variants["endpoint_positive"]["score_mean"], 1.0 / (1.0 + np.exp(-2.0)))
    assert variants["terminal_one_hold_positive"]["count"] == 1.0
    assert variants["terminal_full_hold_positive"]["count"] == 1.0
    assert variants["start_0_negative"]["count"] == 1.0
    assert variants["start_15_negative"]["count"] == 1.0
