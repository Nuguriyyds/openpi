"""Tests for the single-frame completion-head evaluator's split / train-fit features.

These tests exercise the pure-Python + numpy pieces of
``scripts/evaluate_completion_head.py``: the ``--split`` CLI, the boundary
train-sample mask, the train-fit metric set (AUPRC / percentiles / separation
gap), the resume split-mismatch check, the HTML episode subselection, and the
report header / legend wording. They deliberately avoid JAX / LeRobot / the
training config so they run anywhere numpy + pytest are installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pytest

_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import evaluate_completion_head as ech  # noqa: E402

# --------------------------------------------------------------------------- #
#  CLI: --split                                                                 #
# --------------------------------------------------------------------------- #


def _parse(argv: list[str]) -> argparse.Namespace:
    original = sys.argv
    sys.argv = ["completion-head-eval", *argv]
    try:
        return ech._parse_args()  # noqa: SLF001
    finally:
        sys.argv = original


def test_default_split_is_test():
    args = _parse(["--worker-checkpoint", "x", "--worker-output", "y"])
    assert args.split == "test"


def test_split_rejects_invalid_choice():
    with pytest.raises(SystemExit):
        _parse(["--split", "bogus", "--worker-checkpoint", "x", "--worker-output", "y"])


def test_train_split_defaults_to_no_copy_videos():
    train_args = _parse(["--split", "train", "--worker-checkpoint", "x", "--worker-output", "y"])
    test_args = _parse(["--split", "test", "--worker-checkpoint", "x", "--worker-output", "y"])
    assert train_args.copy_videos is False
    assert test_args.copy_videos is True


def test_worker_command_forwards_split():
    args = _parse(["--split", "train", "--worker-checkpoint", "x", "--worker-output", "y"])
    command = ech._checkpoint_worker_command(  # noqa: SLF001
        args,
        checkpoint_dir=Path("/ckpt/200"),
        prediction_file=Path("/out/predictions.npz"),
    )
    assert "--split" in command
    assert command[command.index("--split") + 1] == "train"


# --------------------------------------------------------------------------- #
#  Boundary train-sample mask                                                   #
# --------------------------------------------------------------------------- #


def test_train_sample_mask_selects_sampled_frames():
    episode_indices = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    frame_indices = np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int32)
    sample_sets = {0: np.array([0, 2, 3]), 1: np.array([1])}
    mask = ech._train_sample_mask(episode_indices, frame_indices, sample_sets)  # noqa: SLF001
    assert mask.tolist() == [True, False, True, True, False, True, False]


def test_train_sample_mask_empty_for_val_and_test():
    episode_indices = np.array([0, 0, 1, 1], dtype=np.int32)
    frame_indices = np.array([0, 1, 0, 1], dtype=np.int32)
    # val/test never build sample sets, so the mask must be all False.
    mask = ech._train_sample_mask(episode_indices, frame_indices, {})  # noqa: SLF001
    assert mask.dtype == bool
    assert not mask.any()


# --------------------------------------------------------------------------- #
#  Train-fit metrics: AUPRC, percentiles, separation gap                        #
# --------------------------------------------------------------------------- #


def test_average_precision_single_class_returns_none():
    assert ech._average_precision(np.array([0.1, 0.9]), np.array([0, 0])) is None  # noqa: SLF001
    assert ech._average_precision(np.array([0.1, 0.9]), np.array([1, 1])) is None  # noqa: SLF001


def test_average_precision_perfect_separation_is_one():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    targets = np.array([1, 1, 0, 0])
    assert ech._average_precision(scores, targets) == pytest.approx(1.0)  # noqa: SLF001


def test_average_precision_matches_step_definition():
    # Sorted descending: target sequence 1, 0, 1, 0 -> AP = 0.5 + (0.5 * 2/3).
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    targets = np.array([1, 0, 1, 0])
    assert ech._average_precision(scores, targets) == pytest.approx(5.0 / 6.0)  # noqa: SLF001


def test_safe_percentile_empty_returns_none():
    assert ech._safe_percentile(np.array([]), 50) is None  # noqa: SLF001
    assert ech._safe_percentile(np.array([1.0, 2.0]), 50) == pytest.approx(1.5)  # noqa: SLF001


_REQUIRED_TRAIN_FIT_KEYS = {
    "sample_count",
    "positive_count",
    "negative_count",
    "positive_fraction",
    "bce",
    "auprc",
    "auc",
    "precision_at_0.5",
    "recall_at_0.5",
    "f1_at_0.5",
    "positive_score_mean",
    "positive_score_p10",
    "positive_score_p50",
    "positive_score_p90",
    "negative_score_mean",
    "negative_score_p90",
    "negative_score_p95",
    "negative_score_p99",
    "negative_score_max",
    "positive_median_minus_negative_p95",
    "best_f1",
    "best_threshold",
    "confusion_matrix",
}


def test_train_fit_metrics_has_required_keys():
    logits = np.array([2.0, -2.0, 2.0, -2.0])
    targets = np.array([1, 0, 1, 0])
    metrics = ech._train_fit_metrics(logits, targets)  # noqa: SLF001
    assert _REQUIRED_TRAIN_FIT_KEYS.issubset(metrics.keys())
    assert metrics["sample_count"] == 4
    assert metrics["positive_count"] == 2
    assert metrics["negative_count"] == 2
    assert metrics["confusion_matrix"] == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}


def test_train_fit_metrics_single_class_returns_null_auprc_auc():
    metrics = ech._train_fit_metrics(np.array([-1.0, 1.0]), np.array([0, 0]))  # noqa: SLF001
    assert metrics["auprc"] is None
    assert metrics["auc"] is None
    assert metrics["positive_count"] == 0


def test_train_fit_metrics_perfect_separation():
    logits = np.array([5.0, 5.0, -5.0, -5.0])
    targets = np.array([1, 1, 0, 0])
    metrics = ech._train_fit_metrics(logits, targets)  # noqa: SLF001
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["f1_at_0.5"] == pytest.approx(1.0)
    assert metrics["precision_at_0.5"] == pytest.approx(1.0)
    assert metrics["recall_at_0.5"] == pytest.approx(1.0)


def test_train_fit_metrics_separation_gap_uses_median_minus_negative_p95():
    logits = np.array([5.0, 5.0, -5.0, -5.0, -5.0, -5.0])
    targets = np.array([1, 1, 0, 0, 0, 0])
    metrics = ech._train_fit_metrics(logits, targets)  # noqa: SLF001
    expected = metrics["positive_score_p50"] - metrics["negative_score_p95"]
    assert metrics["positive_median_minus_negative_p95"] == pytest.approx(expected)
    assert expected > 0.5


# --------------------------------------------------------------------------- #
#  _compute_train_fit_metrics: two sets from one npz                            #
# --------------------------------------------------------------------------- #


def _write_predictions(path: Path, *, split: str = "train") -> None:
    # Two episodes, 4 frames each. Episode 0: last frame positive. Episode 1:
    # last two frames positive. The training sampler "sampled" the positives
    # plus frame 0 of each episode (a forced-first negative).
    episode_index = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
    task_index = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16)
    frame_index = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
    logit = np.array([-2.0, -2.0, -2.0, 2.0, -2.0, -2.0, 2.0, 2.0], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    is_train_sample = np.array([1, 0, 0, 1, 1, 0, 1, 1], dtype=bool)
    np.savez_compressed(
        path,
        episode_index=episode_index,
        task_index=task_index,
        frame_index=frame_index,
        logit=logit,
        target=target,
        infer_ms=np.zeros(8, dtype=np.float32),
        is_train_sample=is_train_sample,
        split=np.asarray(split),
    )


def test_compute_train_fit_metrics_produces_two_sets(tmp_path):
    path = tmp_path / "predictions.npz"
    _write_predictions(path)
    result = ech._compute_train_fit_metrics(path)  # noqa: SLF001
    assert set(result.keys()) == {"metrics_all_frames", "metrics_train_sampled"}
    all_frames = result["metrics_all_frames"]["overall"]
    sampled = result["metrics_train_sampled"]["overall"]
    assert all_frames["sample_count"] == 8
    assert sampled["sample_count"] == 5  # 5 is_train_sample==True rows
    assert sampled["positive_count"] == 3  # all three positives were sampled
    assert sampled["negative_count"] == 2  # the two forced-first negatives


def test_compute_train_fit_metrics_sampled_set_excludes_non_sampled(tmp_path):
    path = tmp_path / "predictions.npz"
    _write_predictions(path)
    result = ech._compute_train_fit_metrics(path)  # noqa: SLF001
    episodes = result["metrics_train_sampled"]["episodes"]
    # Episode 0 sampled frames: {0, 3}; episode 1 sampled frames: {0, 2, 3}.
    by_episode = {row["episode_index"]: row for row in episodes}
    assert by_episode[0]["sample_count"] == 2
    assert by_episode[1]["sample_count"] == 3


# --------------------------------------------------------------------------- #
#  Resume split checks                                                          #
# --------------------------------------------------------------------------- #


def test_read_prediction_split_returns_stored_split(tmp_path):
    path = tmp_path / "predictions.npz"
    _write_predictions(path, split="train")
    assert ech._read_prediction_split(path) == "train"  # noqa: SLF001


def test_read_prediction_split_none_for_legacy_npz(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        episode_index=np.zeros(1, dtype=np.int32),
        task_index=np.zeros(1, dtype=np.int16),
        frame_index=np.zeros(1, dtype=np.int32),
        logit=np.zeros(1, dtype=np.float32),
        target=np.zeros(1, dtype=np.float32),
        infer_ms=np.zeros(1, dtype=np.float32),
    )
    assert ech._read_prediction_split(path) is None  # noqa: SLF001


def test_assert_split_match_raises_on_mismatch(tmp_path):
    path = tmp_path / "predictions.npz"
    _write_predictions(path, split="train")
    with pytest.raises(ValueError, match="split 'train'"):
        ech._assert_prediction_split_matches(path, "test")  # noqa: SLF001


def test_assert_split_match_refuses_legacy_npz_in_train_mode(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, logit=np.zeros(1, dtype=np.float32))
    with pytest.raises(ValueError, match="predates the split field"):
        ech._assert_prediction_split_matches(path, "train")  # noqa: SLF001
    # A legacy test-only file may still resume a test run.
    ech._assert_prediction_split_matches(path, "test")  # noqa: SLF001


# --------------------------------------------------------------------------- #
#  HTML episode subselection                                                    #
# --------------------------------------------------------------------------- #


def _fake_series() -> list[dict]:
    # 8 episodes spread across 4 tasks (2 each), with distinct BCE values.
    series = []
    bce = 0.10
    for episode_index in range(8):
        task_index = episode_index // 2
        series.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "metrics": {"bce": bce},
            }
        )
        bce += 0.10
    return series


def test_select_report_episodes_explicit_ids():
    series = _fake_series()
    selected = ech._select_report_episodes(series, max_episodes=0, episode_ids=(1, 5))  # noqa: SLF001
    assert [ep["episode_index"] for ep in selected] == [1, 5]


def test_select_report_episodes_explicit_missing_id_raises():
    series = _fake_series()
    with pytest.raises(ValueError, match="not found in this split"):
        ech._select_report_episodes(series, max_episodes=0, episode_ids=(99,))  # noqa: SLF001


def test_auto_select_episodes_deterministic_capped_and_balanced():
    series = _fake_series()
    first = ech._select_report_episodes(series, max_episodes=4, episode_ids=())  # noqa: SLF001
    second = ech._select_report_episodes(series, max_episodes=4, episode_ids=())  # noqa: SLF001
    assert [ep["episode_index"] for ep in first] == [ep["episode_index"] for ep in second]
    assert len(first) == 4
    # Best (lowest BCE = episode 0) and worst (highest BCE = episode 7) included.
    indices = {ep["episode_index"] for ep in first}
    assert 0 in indices
    assert 7 in indices
    # Balanced across the four tasks: at least one episode per task.
    tasks = {ep["task_index"] for ep in first}
    assert tasks == {0, 1, 2, 3}


def test_select_report_episodes_zero_keeps_all():
    series = _fake_series()
    selected = ech._select_report_episodes(series, max_episodes=0, episode_ids=())  # noqa: SLF001
    assert len(selected) == len(series)


# --------------------------------------------------------------------------- #
#  Comparison CSV + HTML wording                                                #
# --------------------------------------------------------------------------- #


def _fake_overall() -> dict:
    keys = [
        "frame_count",
        "positive_count",
        "negative_count",
        "auc",
        "best_f1",
        "best_threshold",
        "best_precision",
        "best_recall",
        "f1_at_0.5",
        "precision_at_0.5",
        "recall_at_0.5",
        "bce",
        "positive_score_mean",
        "negative_score_mean",
        "early_trigger_rate",
        "never_trigger_rate",
        "pre_threshold_false_positive_rate",
        "mean_detection_delay_frames",
        "median_detection_delay_frames",
        "mean_detection_delay_seconds",
        "median_detection_delay_seconds",
        "mean_infer_ms_per_frame",
    ]
    return dict.fromkeys(keys, 0.0)


def test_comparison_row_includes_split_column():
    metrics = {"overall": _fake_overall(), "per_task": {}}
    row = ech._comparison_row(200, metrics, split="train")  # noqa: SLF001
    assert row["checkpoint_step"] == 200
    assert row["split"] == "train"


def _fake_episode(episode_index: int, task_index: int) -> dict:
    return {
        "episode_index": episode_index,
        "task_index": task_index,
        "prompt": "make breakfast",
        "fps": 10.0,
        "frame_count": 5,
        "video": "videos/episode.mp4",
        "source_video": "/data/episode.mp4",
        "score": [0.1, 0.2, 0.3, 0.8, 0.9],
        "target": [0.0, 0.0, 0.0, 1.0, 1.0],
        "sampled_positive_frames": [3, 4],
        "sampled_negative_frames": [0],
        "metrics": {
            "auc": 1.0,
            "best_f1": 1.0,
            "best_threshold": 0.5,
            "best_precision": 1.0,
            "best_recall": 1.0,
            "f1_at_0.5": 1.0,
            "precision_at_0.5": 1.0,
            "recall_at_0.5": 1.0,
            "bce": 0.1,
            "positive_score_mean": 0.85,
            "negative_score_mean": 0.2,
            "early_triggered": False,
            "never_triggered": False,
            "detection_delay_seconds": 0.0,
        },
    }


def _render_html(split: str = "train") -> str:
    manifest = {
        "checkpoint_step": 200,
        "split": split,
        "created_at_utc": "2026-08-18T00:00:00Z",
        "dataset_root": "/data",
        "threshold": 0.5,
        "top_camera_key": ech.TOP_VIDEO_KEY,
        "episodes": [_fake_episode(0, 0)],
    }
    return ech._html_document(manifest)  # noqa: SLF001


def test_html_header_has_split_model_input_and_sample_rule():
    html = _render_html(split="train")
    assert "Split:" in html
    assert "current-frame prefix only" in html
    assert "all positives + stride-sampled negatives" in html
    assert "Checkpoint" in html


def test_html_has_no_temporal_wording():
    html = _render_html(split="train")
    for forbidden in ("temporal prefix", "history prefix", "three-frame", "t-2/t-1/t", "temporal MLP"):
        assert forbidden not in html, f"report must not mention {forbidden!r}"


def test_html_legend_has_sampled_positive_and_negative():
    html = _render_html(split="train")
    assert "Sampled positive" in html
    assert "Sampled negative" in html
    assert "Threshold" in html


# --------------------------------------------------------------------------- #
#  _load_report_series attaches sampled frame lists                             #
# --------------------------------------------------------------------------- #


def test_load_report_series_attaches_sampled_frames(tmp_path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "videos" / "chunk-0" / "episode_000000").mkdir(parents=True)
    info = {
        "fps": 10.0,
        "chunks_size": 1000,
        "video_path": "videos/chunk-{episode_chunk}/episode_{episode_index:06d}/{video_key}.mp4",
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "make breakfast"}) + "\n", encoding="utf-8"
    )
    (root / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 4, "task_index": 0}) + "\n", encoding="utf-8"
    )
    video = root / "videos" / "chunk-0" / "episode_000000" / f"{ech.TOP_VIDEO_KEY}.mp4"
    video.write_bytes(b"")  # empty file satisfies the is_file() check

    prediction_file = tmp_path / "predictions.npz"
    # 4 frames: targets [0,0,0,1]; frames 0 and 3 were sampled during training.
    np.savez_compressed(
        prediction_file,
        episode_index=np.array([0, 0, 0, 0], dtype=np.int32),
        task_index=np.array([0, 0, 0, 0], dtype=np.int16),
        frame_index=np.array([0, 1, 2, 3], dtype=np.int32),
        logit=np.array([-2.0, -2.0, -2.0, 2.0], dtype=np.float32),
        target=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        is_train_sample=np.array([1, 0, 0, 1], dtype=bool),
        split=np.asarray("train"),
    )
    episode_metrics = [
        {"episode_index": 0, "task_index": 0, "bce": 0.1, "auc": 1.0, "best_f1": 1.0}
    ]
    report_dir = tmp_path / "report"
    series, _ = ech._load_report_series(  # noqa: SLF001
        root,
        prediction_file,
        episode_metrics,
        output_dir=report_dir,
        copy_videos=False,
    )
    assert len(series) == 1
    episode = series[0]
    # Frame 3 is a sampled positive (target 1, sampled); frame 0 a sampled negative.
    assert episode["sampled_positive_frames"] == [3]
    assert episode["sampled_negative_frames"] == [0]
