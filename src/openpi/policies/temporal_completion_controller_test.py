import numpy as np
import pytest

from openpi.policies.temporal_completion_controller import TemporalCompletionController
from openpi.policies.temporal_completion_controller import TemporalCompletionPolicy

PROMPTS = ("task zero", "task one", "task two", "task three")


def _feature(value: float) -> np.ndarray:
    return np.full((4,), value, dtype=np.float32)


def test_controller_waits_for_three_exact_2hz_features_and_preserves_order():
    seen = []
    controller = TemporalCompletionController(PROMPTS, threshold=0.7)

    assert controller.step(0, _feature(0), lambda _: 1.0).score is None
    assert controller.step(15, _feature(1), lambda _: 1.0).score is None

    def score(history):
        seen.append(history.copy())
        return 0.2

    decision = controller.step(30, _feature(2), score)
    assert decision.history_ready
    assert not decision.triggered
    np.testing.assert_array_equal(seen[0][:, 0], [0, 1, 2])


def test_trigger_switches_prompt_for_next_forward_and_clears_history():
    controller = TemporalCompletionController(PROMPTS, threshold=0.5)
    for frame in (0, 15):
        controller.step(frame, _feature(frame), lambda _: 0.0)
    decision = controller.step(30, _feature(30), lambda _: 0.5)

    assert decision.triggered
    assert decision.task_before == 0
    assert decision.task_after == 1
    assert decision.invalidate_action_plan
    assert controller.current_prompt == "task one"
    assert controller.history_size == 0
    assert controller.override_prompt({"prompt": "whole breakfast", "x": 1}) == {
        "prompt": "task one",
        "x": 1,
    }

    assert controller.step(45, _feature(45), lambda _: 1.0).score is None
    assert controller.step(60, _feature(60), lambda _: 1.0).score is None
    assert controller.step(75, _feature(75), lambda _: 0.0).history_ready


def test_task_three_trigger_latches_done():
    controller = TemporalCompletionController(PROMPTS, threshold=0.5)
    frame = 0
    for task in range(4):
        for _ in range(2):
            controller.step(frame, _feature(frame), lambda _: 0.0)
            frame += 15
        decision = controller.step(frame, _feature(frame), lambda _: 1.0)
        frame += 15
        assert decision.triggered
        assert decision.done is (task == 3)

    assert controller.done
    assert controller.triggered_boundaries == (30, 75, 120, 165)
    with pytest.raises(RuntimeError, match="already done"):
        controller.step(frame, _feature(frame), lambda _: 0.0)


@pytest.mark.parametrize("bad_frame", [1, 14, 16])
def test_controller_rejects_off_grid_frames(bad_frame):
    controller = TemporalCompletionController(PROMPTS, threshold=0.5)
    with pytest.raises(ValueError, match="tick grid"):
        controller.step(bad_frame, _feature(0), lambda _: 0.0)


@pytest.mark.parametrize("bad_frame", [0.0, 15.5, True, "30"])
def test_controller_rejects_non_integer_trajectory_frame_index(bad_frame):
    controller = TemporalCompletionController(PROMPTS, threshold=0.5)
    with pytest.raises(ValueError, match="must be an integer"):
        controller.step(bad_frame, _feature(0), lambda _: 0.0)


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.1, np.nan])
def test_controller_rejects_non_probability_threshold(bad_threshold):
    with pytest.raises(ValueError, match=r"\[0, nextafter\(1,\+inf\)\]"):
        TemporalCompletionController(PROMPTS, threshold=bad_threshold)


def test_controller_accepts_validation_abstention_sentinel_without_triggering():
    controller = TemporalCompletionController(PROMPTS, threshold=np.nextafter(1.0, np.inf))
    controller.step(0, _feature(0), lambda _: 1.0)
    controller.step(15, _feature(0), lambda _: 1.0)
    decision = controller.step(30, _feature(0), lambda _: 1.0)

    assert decision.score == 1.0
    assert not decision.triggered


def test_controller_rejects_stale_or_skipped_feature_tick():
    controller = TemporalCompletionController(PROMPTS, threshold=0.5)
    controller.step(0, _feature(0), lambda _: 0.0)
    with pytest.raises(ValueError, match="exact-gap"):
        controller.step(30, _feature(1), lambda _: 0.0)


def test_controller_rejects_invalid_score_and_feature_shape():
    controller = TemporalCompletionController(PROMPTS, threshold=0.5)
    with pytest.raises(ValueError, match="shape"):
        controller.step(0, np.zeros((1, 4), dtype=np.float32), lambda _: 0.0)

    controller.reset()
    controller.step(0, _feature(0), lambda _: 0.0)
    controller.step(15, _feature(0), lambda _: 0.0)
    with pytest.raises(ValueError, match="one scalar"):
        controller.step(30, _feature(0), lambda _: np.zeros((2,), dtype=np.float32))

    controller.reset()
    controller.step(0, _feature(0), lambda _: 0.0)
    controller.step(15, _feature(0), lambda _: 0.0)
    with pytest.raises(ValueError, match=r"probability in \[0, 1\]"):
        controller.step(30, _feature(0), lambda _: 1.1)


class _FakeFeaturePolicy:
    def __init__(self):
        self.prompts = []

    def infer(self, obs, **kwargs):
        self.prompts.append(obs["prompt"])
        result = {"actions": np.zeros((2, 3), dtype=np.float32)}
        if kwargs["return_prefix_feature"]:
            result["prefix_feature"] = np.asarray([len(self.prompts)], dtype=np.float32)
        return result

    def score_temporal_completion(self, prefix_history, *, return_logit=False):
        del return_logit
        return 0.9 if prefix_history[-1, 0] >= 3 else 0.0


def test_policy_wrapper_overrides_prompt_reuses_tick_feature_and_invalidates_old_plan():
    base = _FakeFeaturePolicy()
    policy = TemporalCompletionPolicy(base, ["task 0", "task 1"], threshold=0.8)

    first = policy.infer({"prompt": "full breakfast"}, trajectory_frame_index=0)
    middle = policy.infer({"prompt": "full breakfast"}, trajectory_frame_index=7)
    second = policy.infer({"prompt": "full breakfast"}, trajectory_frame_index=15)
    third = policy.infer({"prompt": "full breakfast"}, trajectory_frame_index=30)
    after_switch = policy.infer({"prompt": "full breakfast"}, trajectory_frame_index=31)

    assert first["completion_decision"]["history_ready"] is False
    assert middle["completion_decision"] is None
    assert second["completion_decision"]["history_ready"] is False
    assert third["completion_decision"]["triggered"] is True
    assert third["invalidate_action_plan"] is True
    assert third["active_prompt"] == "task 0"
    assert third["active_task_index"] == 0
    assert third["next_prompt"] == "task 1"
    assert third["next_task_index"] == 1
    assert after_switch["active_prompt"] == "task 1"
    assert base.prompts == ["task 0", "task 0", "task 0", "task 0", "task 1"]
