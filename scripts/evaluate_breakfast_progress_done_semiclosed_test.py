import pytest

from openpi.training.breakfast_done_data import DoneSample
from scripts import evaluate_breakfast_progress_done_semiclosed as evaluate


def test_deterministic_xla_flags_are_enabled_by_default():
    flags = evaluate._with_default_deterministic_xla_flags("--existing_flag=true")  # noqa: SLF001

    assert "--existing_flag=true" in flags
    assert "--xla_gpu_deterministic_ops=true" in flags
    assert "--xla_gpu_exclude_nondeterministic_ops=true" in flags


def test_progress_summary_reports_error_monotonicity_and_trigger_calibration():
    episodes = [
        {
            "ticks": [
                {
                    "progress_score": 0.1,
                    "progress_target": 0.0,
                    "source_task_index": 0,
                    "triggered": False,
                },
                {
                    "progress_score": 0.4,
                    "progress_target": 0.5,
                    "source_task_index": 0,
                    "triggered": False,
                },
                {
                    "progress_score": 0.3,
                    "progress_target": 1.0,
                    "source_task_index": 0,
                    "triggered": True,
                },
            ]
        }
    ]

    metrics = evaluate.progress_summary(episodes)

    assert metrics["count"] == 3
    assert metrics["mae"] == pytest.approx(0.3)
    assert metrics["spearman"] == pytest.approx(0.5)
    assert metrics["early_mae"] == pytest.approx(0.1)
    assert metrics["middle_mae"] == pytest.approx(0.1)
    assert metrics["late_mae"] == pytest.approx(0.7)
    assert metrics["backward_step_rate"] == 0.5
    assert metrics["trigger_progress_mean"] == 0.3


def _sample(task: str, frame: int) -> DoneSample:
    return DoneSample(
        sample_id=f"{task}-{frame}",
        split="val",
        episode_index=184,
        query_frame=frame,
        history_frames=(frame - 30, frame - 15, frame),
        current_sub_task=task,
        prompt=task,
        label=0,
        sample_type="continue",
        sampling_source="base",
    )


def test_query_playback_starts_use_existing_query_frames():
    samples = [
        _sample("load_bread_into_toaster", 45),
        _sample("load_bread_into_toaster", 60),
        _sample("activate_toaster", 405),
        _sample("pour_drink_into_cup", 555),
        _sample("place_toasted_bread_on_plate", 1050),
    ]

    starts = evaluate._query_playback_starts(samples, episode_id=184)  # noqa: SLF001

    assert starts == (45, 405, 555, 1050)


def test_attach_progress_uses_annotation_start_not_playback_start():
    result = {
        "playback_start_frames": [45, 405, 555, 1050],
        "gt_end_frames": [390, 540, 1035, 1410],
        "ticks": [
            {
                "history_ready": True,
                "source_task_index": 0,
                "source_frame_index": 75,
            }
        ],
    }

    evaluate._attach_progress(result, [0.2], (31, 388, 543, 1038))  # noqa: SLF001

    assert result["annotation_start_frames"] == [31, 388, 543, 1038]
    assert result["ticks"][0]["progress_target"] == pytest.approx((75 - 31) / (390 - 31))
