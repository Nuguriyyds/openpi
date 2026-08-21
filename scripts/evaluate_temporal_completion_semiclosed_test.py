from __future__ import annotations

import json

import numpy as np
import pytest

from openpi.training.temporal_completion_semiclosed import SemiClosedCompletionController
from openpi.training.temporal_completion_semiclosed import classify_boundary
from openpi.training.temporal_completion_semiclosed import gt_end_frames
from openpi.training.temporal_completion_semiclosed import reference_tick
from scripts import evaluate_temporal_completion_semiclosed as evaluator


def test_global_ends_and_reference_ticks_use_inclusive_endpoints() -> None:
    ends = gt_end_frames((90, 100, 110, 120))
    assert ends == (89, 189, 299, 419)
    assert reference_tick(89) == 90
    assert reference_tick(90) == 90
    assert reference_tick(91) == 105


def test_history_warmup_switch_and_prompt_local_reset() -> None:
    seen: list[np.ndarray] = []

    def score(history: np.ndarray) -> float:
        seen.append(np.array(history, copy=True))
        return 1.0

    controller = SemiClosedCompletionController(("p0", "p1", "p2", "p3"), threshold=0.5, mode="history")
    first = controller.step(0, np.asarray([0.0]), score)
    second = controller.step(15, np.asarray([1.0]), score)
    third = controller.step(30, np.asarray([2.0]), score)
    assert first.score is None
    assert second.score is None
    assert not first.history_ready
    assert not second.history_ready
    assert third.triggered
    assert third.task_before == 0
    assert third.task_after == 1
    assert third.active_prompt == "p0"
    assert seen[0].reshape(-1).tolist() == [0.0, 1.0, 2.0]

    after_switch = controller.step(45, np.asarray([3.0]), score)
    assert after_switch.score is None
    assert not after_switch.history_ready
    assert after_switch.active_prompt == "p1"
    controller.step(60, np.asarray([4.0]), score)
    controller.step(75, np.asarray([5.0]), score)
    assert seen[1].reshape(-1).tolist() == [3.0, 4.0, 5.0]


def test_current_only_scores_immediately_with_zero_history() -> None:
    inputs: list[np.ndarray] = []
    controller = SemiClosedCompletionController(("p0", "p1", "p2", "p3"), threshold=0.5, mode="current_only")

    def score(history: np.ndarray) -> float:
        inputs.append(history)
        return 0.0

    decision = controller.step(0, np.asarray([2.0, 3.0]), score)
    assert decision.history_ready
    assert decision.score == 0.0
    np.testing.assert_array_equal(inputs[0], np.asarray([[0.0, 0.0], [0.0, 0.0], [2.0, 3.0]], dtype=np.float32))


def test_cascade_does_not_ground_truth_correct_and_unavailable_is_explicit() -> None:
    controller = SemiClosedCompletionController(("p0", "p1", "p2", "p3"), threshold=0.5, mode="current_only")
    decision = controller.step(15, np.asarray([1.0]), lambda _: 1.0)
    assert decision.triggered
    assert decision.task_before == 0
    assert decision.task_after == 1
    result = classify_boundary(3, 120, None, full_length=120)
    assert result.classification == "missed"
    assert not result.reference_tick_available
    result = classify_boundary(3, 120, 105, full_length=120)
    assert result.classification == "unavailable"


def test_threshold_requires_one_source_and_never_searches_test(tmp_path) -> None:
    path = tmp_path / "validation.json"
    path.write_text(json.dumps({"threshold_selection": {"threshold": 0.37}}), encoding="utf-8")
    threshold, source, stored_path = evaluator.load_threshold(validation_report=path, explicit_threshold=None)
    assert threshold == pytest.approx(0.37)
    assert source == "validation_report"
    assert stored_path == str(path.resolve())
    with pytest.raises(ValueError, match="exactly one"):
        evaluator.load_threshold(validation_report=path, explicit_threshold=0.5)
    with pytest.raises(ValueError, match="exactly one"):
        evaluator.load_threshold(validation_report=None, explicit_threshold=None)
