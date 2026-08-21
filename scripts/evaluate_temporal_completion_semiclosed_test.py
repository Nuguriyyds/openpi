from __future__ import annotations

import json

import numpy as np
import pytest

from openpi.training.temporal_completion_semiclosed import GatedCompletionController
from openpi.training.temporal_completion_semiclosed import SemiClosedCompletionController
from openpi.training.temporal_completion_semiclosed import classify_boundary
from openpi.training.temporal_completion_semiclosed import gt_end_frames
from openpi.training.temporal_completion_semiclosed import reference_tick
from scripts import evaluate_temporal_completion_semiclosed as evaluator


def _gated_controller(*, mode: str, threshold: float = 0.5, timeout_seconds: float = 2.0):
    return GatedCompletionController(
        ("p0", "p1", "p2", "p3"),
        playback_start_frames=(0, 100, 200, 300),
        gt_end_frames=(30, 130, 230, 330),
        threshold=threshold,
        mode=mode,  # type: ignore[arg-type]
        timeout_seconds=timeout_seconds,
    )


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


def test_transition_warms_up_with_three_prefixes_at_episode_start() -> None:
    inputs: list[np.ndarray] = []
    controller = SemiClosedCompletionController(("p0", "p1", "p2", "p3"), threshold=0.5, mode="transition")

    def score(history: np.ndarray) -> float:
        inputs.append(np.array(history, copy=True))
        return 0.0

    first = controller.step(0, np.asarray([0.0]), score)
    second = controller.step(15, np.asarray([1.0]), score)
    third = controller.step(30, np.asarray([2.0]), score)
    assert first.score is None
    assert second.score is None
    assert not first.history_ready
    assert not second.history_ready
    assert third.history_ready
    np.testing.assert_array_equal(inputs, np.asarray([[[0.0], [1.0], [2.0]]], dtype=np.float32))


def test_transition_switch_keeps_deque_and_scores_first_new_prompt_tick() -> None:
    inputs: list[np.ndarray] = []
    controller = SemiClosedCompletionController(("p0", "p1", "p2", "p3"), threshold=0.5, mode="transition")

    def score(history: np.ndarray) -> float:
        inputs.append(np.array(history, copy=True))
        return 1.0

    controller.step(0, np.asarray([0.0]), score)
    controller.step(15, np.asarray([1.0]), score)
    switched = controller.step(30, np.asarray([2.0]), score)
    first_new_tick = controller.step(45, np.asarray([3.0]), score)
    assert switched.triggered
    assert switched.active_prompt == "p0"
    assert first_new_tick.active_prompt == "p1"
    assert first_new_tick.history_ready
    assert first_new_tick.score == 1.0
    assert controller.history_size == 3
    np.testing.assert_array_equal(inputs[1], np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32))


def test_transition_slots_evolve_old_old_new_to_new_new_new() -> None:
    inputs: list[np.ndarray] = []
    controller = SemiClosedCompletionController(("p0", "p1", "p2", "p3"), threshold=0.5, mode="transition")

    for frame, value in ((0, 0.0), (15, 1.0), (30, 2.0), (45, 3.0), (60, 4.0), (75, 5.0)):
        controller.step(frame, np.asarray([value]), lambda history: inputs.append(np.array(history, copy=True)) or 1.0)

    assert len(inputs) == 4
    np.testing.assert_array_equal(inputs[1], np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32))
    np.testing.assert_array_equal(inputs[2], np.asarray([[2.0], [3.0], [4.0]], dtype=np.float32))
    np.testing.assert_array_equal(inputs[3], np.asarray([[3.0], [4.0], [5.0]], dtype=np.float32))


def test_gated_replay_caps_source_frame_and_repeats_terminal_until_timeout() -> None:
    controller = _gated_controller(mode="current_only")
    decisions = []
    while not controller.done:
        decision = controller.step(len(decisions), np.asarray([0.0]), lambda _: 0.0)
        decisions.append(decision)
    assert max(decision.source_frame_index for decision in decisions if decision.active_task_index == 0) == 30
    task0 = [decision for decision in decisions if decision.active_task_index == 0]
    assert [decision.source_frame_index for decision in task0[-5:]] == [30, 30, 30, 30, 30]
    assert [decision.terminal_hold_tick for decision in task0[-5:]] == [0, 1, 2, 3, 4]
    assert task0[-1].score == 0.0
    assert task0[-1].timeout_forced
    assert task0[-1].switch_reason == "timeout"


def test_gated_replay_on_time_scores_at_first_terminal_tick_and_switches_next_tick() -> None:
    controller = _gated_controller(mode="current_only")
    decisions = []
    while not controller.done:
        tick = len(decisions)
        source_frame = controller.current_source_frame
        decision = controller.step(
            tick,
            np.asarray([float(source_frame)]),
            lambda history: float(history[-1, 0]) == float(controller.gt_end_frames[controller.current_task_index]),
        )
        decisions.append(decision)
    task0 = [decision for decision in decisions if decision.active_task_index == 0]
    assert task0[-1].source_frame_index == 30
    assert task0[-1].terminal_hold_tick == 0
    assert task0[-1].switch_reason == "head_on_time"
    assert task0[-1].triggered
    assert decisions[len(task0)].active_task_index == 1
    assert decisions[len(task0)].source_frame_index == 100


def test_gated_timeout_deadline_is_scored_before_forcing_and_task3_finishes() -> None:
    calls: list[int] = []
    controller = _gated_controller(mode="current_only", timeout_seconds=2.0)
    decisions = []

    def score(_: np.ndarray) -> float:
        calls.append(1)
        return 0.0

    while not controller.done:
        decision = controller.step(len(decisions), np.asarray([0.0]), score)
        decisions.append(decision)
    assert len(calls) == len(decisions)
    task3 = [decision for decision in decisions if decision.active_task_index == 3]
    assert task3[-1].terminal_hold_tick == 4
    assert task3[-1].timeout_forced
    assert task3[-1].done
    assert controller.done_rollout_tick == task3[-1].rollout_tick


def test_gated_history_clears_after_switch_and_current_only_scores_first_tick() -> None:
    history = _gated_controller(mode="history")
    history_decisions = []
    while history.current_task_index == 0:
        tick = len(history_decisions)
        source_frame = history.current_source_frame
        history_decisions.append(
            history.step(tick, np.asarray([float(source_frame)]), lambda values: float(values[-1, 0]) == 30.0)
        )
    # The task-0 end is reached after three warmup prefixes; after the
    # on-time switch, task 1 starts with an empty history.
    first_task1 = history.step(len(history_decisions), np.asarray([0.0]), lambda _: 1.0)
    second_task1 = history.step(len(history_decisions) + 1, np.asarray([0.0]), lambda _: 1.0)
    assert not first_task1.history_ready
    assert first_task1.score is None
    assert not second_task1.history_ready
    assert second_task1.score is None

    current_only = _gated_controller(mode="current_only")
    first = current_only.step(0, np.asarray([0.0]), lambda _: 0.0)
    assert first.history_ready
    assert first.score == 0.0


def test_gated_transition_keeps_history_and_scores_first_new_task_tick() -> None:
    controller = _gated_controller(mode="transition")
    inputs: list[np.ndarray] = []

    def score(history: np.ndarray) -> float:
        inputs.append(np.array(history, copy=True))
        return float(history[-1, 0]) in (2.0, 99.0)

    decisions = []
    while controller.current_task_index == 0:
        decisions.append(controller.step(len(decisions), np.asarray([float(len(decisions))]), score))
    first_new = controller.step(len(decisions), np.asarray([99.0]), score)
    assert first_new.active_task_index == first_new.source_task_index == 1
    assert first_new.history_ready
    assert first_new.score == 1.0
    np.testing.assert_array_equal(inputs[-1], np.asarray([[1.0], [2.0], [99.0]], dtype=np.float32))


@pytest.mark.parametrize("timeout", [0.0, -0.5, 0.25, 1.1])
def test_gated_timeout_requires_positive_half_second_multiple(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        _gated_controller(mode="current_only", timeout_seconds=timeout)


def test_gated_timeout_default_is_two_seconds_and_four_ticks() -> None:
    assert evaluator.validate_timeout_seconds(2.0) == (2.0, 4)


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
