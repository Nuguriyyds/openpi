import numpy as np

from openpi.training.temporal_completion_semiclosed import GatedCompletionController


def _raw_prefix(value: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "prefix_out": np.full((5, 4), value, dtype=np.float32),
        "prefix_mask": np.ones((5,), dtype=np.bool_),
        "prefix_segment_ids": np.asarray([0, 1, 2, 3, 3], dtype=np.int32),
        "prefix_position_ids": np.asarray([0, 0, 0, 0, 1], dtype=np.int32),
    }


def test_raw_prefix_gated_controller_is_current_only_and_records_logit():
    controller = GatedCompletionController(
        ("task 0", "task 1", "task 2", "task 3"),
        playback_start_frames=(0, 30, 60, 90),
        gt_end_frames=(30, 60, 90, 120),
        threshold=0.5,
        mode="raw_prefix_current",
    )
    seen = []

    def score(raw_input):
        seen.append(raw_input)
        return 0.75, 1.1

    decision = controller.step(0, _raw_prefix(), score)

    assert decision.history_ready
    assert decision.triggered
    assert decision.logit == 1.1
    assert controller.history_size == 0
    assert seen[0]["prefix_out"].shape == (5, 4)
    assert controller.task_results[0].classification == "early"
    assert controller.task_results[0].trigger_logit == 1.1

