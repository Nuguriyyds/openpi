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
import random
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
        ech._assert_prediction_scope_matches(  # noqa: SLF001
            path,
            expected_split="test",
            train_group_count=0,
            train_group_seed=42,
            expected_scope=None,
        )


def test_assert_split_match_refuses_legacy_npz_in_train_mode(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez_compressed(path, logit=np.zeros(1, dtype=np.float32))
    with pytest.raises(ValueError, match="predates the split field"):
        ech._assert_prediction_scope_matches(  # noqa: SLF001
            path,
            expected_split="train",
            train_group_count=0,
            train_group_seed=42,
            expected_scope=None,
        )
    # A legacy test-only file may still resume a test run.
    ech._assert_prediction_scope_matches(  # noqa: SLF001
        path,
        expected_split="test",
        train_group_count=0,
        train_group_seed=42,
        expected_scope=None,
    )


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
    episode_metrics = [{"episode_index": 0, "task_index": 0, "bce": 0.1, "auc": 1.0, "best_f1": 1.0}]
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


def _make_video_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Minimal dataset (1 episode, 4 frames) with a top-camera video file."""

    root = tmp_path / "dataset"
    (root / "videos" / "chunk-0" / "episode_000000").mkdir(parents=True)
    (root / "meta").mkdir(parents=True)
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
    (root / "videos" / "chunk-0" / "episode_000000" / f"{ech.TOP_VIDEO_KEY}.mp4").write_bytes(b"video-bytes")
    prediction_file = tmp_path / "predictions.npz"
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
    return root, prediction_file


def test_load_report_series_copy_videos_is_idempotent(tmp_path):
    root, prediction_file = _make_video_dataset(tmp_path)
    report_dir = tmp_path / "report"
    episode_metrics = [{"episode_index": 0, "task_index": 0, "bce": 0.1, "auc": 1.0, "best_f1": 1.0}]

    first, _ = ech._load_report_series(  # noqa: SLF001
        root, prediction_file, episode_metrics, output_dir=report_dir, copy_videos=True
    )
    copied_video = report_dir / "videos" / "episode_000000.mp4"
    assert copied_video.is_file()
    # HTML references the video by a relative path, not a file:// URI.
    assert first[0]["video"] == "videos/episode_000000.mp4"
    assert not first[0]["video"].startswith("file://")

    # Mark the copied video; a re-copy would overwrite this marker back to the
    # source bytes (shutil.copy2 preserves the source's content, not ours).
    with copied_video.open("ab") as handle:
        handle.write(b"-marker")
    marked = copied_video.read_bytes()

    # Second call (simulating --resume) must NOT re-copy the already-present video.
    second, _ = ech._load_report_series(  # noqa: SLF001
        root, prediction_file, episode_metrics, output_dir=report_dir, copy_videos=True
    )
    assert copied_video.read_bytes() == marked  # marker survived -> not re-copied
    assert second[0]["video"] == "videos/episode_000000.mp4"


# --------------------------------------------------------------------------- #
#  Evaluation-scope group selection (--train-group-count)                       #
# --------------------------------------------------------------------------- #


class _FakeGroup:
    """Stand-in for completion_data.TaskGroup (group_id + episode_ids)."""

    def __init__(self, group_id: int, episode_ids: tuple[int, ...]):
        self.group_id = group_id
        self.episode_ids = tuple(episode_ids)


class _FakeManifest:
    """Stand-in for completion_data.SplitManifest (splits mapping)."""

    def __init__(self, splits: dict[str, tuple[_FakeGroup, ...]]):
        self.splits = splits


def _fake_manifest(n_train_groups: int, *, n_test_groups: int = 3) -> _FakeManifest:
    """Build a fake manifest with 4-episode groups (BOUNDARY_GROUP == 4)."""

    def build(count: int) -> tuple[_FakeGroup, ...]:
        return tuple(_FakeGroup(group_id=g, episode_ids=tuple(range(g * 4, g * 4 + 4))) for g in range(count))

    return _FakeManifest({"train": build(n_train_groups), "val": (), "test": build(n_test_groups)})


def _scope(group_ids, episode_ids, *, count, seed, split="train") -> ech.EvaluationScope:
    return ech.EvaluationScope(
        split=split,
        groups=(),
        episode_ids=tuple(episode_ids),
        group_ids=tuple(group_ids),
        requested_group_count=count,
        group_seed=seed,
        is_full_split=count == 0,
    )


def _write_scoped_npz(
    path: Path,
    *,
    group_ids,
    count: int,
    seed: int,
    split: str = "train",
) -> ech.EvaluationScope:
    """Write a minimal npz carrying group-sampling scope arrays.

    Returns the matching ``EvaluationScope`` so resume tests can pass it as the
    expected scope. Only the scope arrays matter here; the frame arrays are
    placeholders because ``_assert_prediction_scope_matches`` never reads them.
    """

    episode_ids = [eid for g in group_ids for eid in range(g * 4, g * 4 + 4)]
    np.savez_compressed(
        path,
        episode_index=np.array(episode_ids[:1], dtype=np.int32),
        task_index=np.array([0], dtype=np.int16),
        frame_index=np.array([0], dtype=np.int32),
        logit=np.array([0.0], dtype=np.float32),
        target=np.array([0.0], dtype=np.float32),
        infer_ms=np.array([0.0], dtype=np.float32),
        is_train_sample=np.array([0], dtype=bool),
        split=np.asarray(split),
        selected_group_ids=np.asarray(group_ids, dtype=np.int64),
        selected_episode_ids=np.asarray(episode_ids, dtype=np.int64),
        requested_train_group_count=np.asarray(count, dtype=np.int64),
        train_group_seed=np.asarray(seed, dtype=np.int64),
    )
    return _scope(group_ids, episode_ids, count=count, seed=seed, split=split)


# --- CLI defaults / validation ---------------------------------------------- #


def test_train_group_count_defaults_to_zero():
    args = _parse(["--worker-checkpoint", "x", "--worker-output", "y"])
    assert args.train_group_count == 0


def test_train_group_seed_defaults_to_42_and_is_independent_of_model_seed():
    args = _parse(["--seed", "7", "--worker-checkpoint", "x", "--worker-output", "y"])
    assert args.seed == 7
    assert args.train_group_seed == 42


def test_parse_train_group_count_requires_train_split():
    with pytest.raises(SystemExit):
        _parse(["--split", "test", "--train-group-count", "5", "--worker-checkpoint", "x", "--worker-output", "y"])


def test_parse_train_group_count_negative_rejected():
    with pytest.raises(SystemExit):
        _parse(["--split", "train", "--train-group-count", "-1", "--worker-checkpoint", "x", "--worker-output", "y"])


def test_parse_train_group_count_zero_allowed_on_non_train_split():
    # count == 0 means "no sampling" and is allowed for any split.
    args = _parse(["--split", "test", "--train-group-count", "0", "--worker-checkpoint", "x", "--worker-output", "y"])
    assert args.train_group_count == 0


def test_report_max_episodes_is_independent_of_train_group_count():
    args = _parse(
        [
            "--split",
            "train",
            "--train-group-count",
            "20",
            "--report-max-episodes",
            "5",
            "--worker-checkpoint",
            "x",
            "--worker-output",
            "y",
        ]
    )
    assert args.train_group_count == 20
    assert args.report_max_episodes == 5


# --- _select_evaluation_groups (pure) --------------------------------------- #


def test_select_groups_zero_returns_all_in_manifest_order():
    groups = _fake_manifest(5).splits["train"]
    selected = ech._select_evaluation_groups(groups, group_count=0, seed=42)  # noqa: SLF001
    assert selected == tuple(groups)
    assert [g.group_id for g in selected] == [0, 1, 2, 3, 4]


def test_select_groups_deterministic_for_same_seed_and_count():
    groups = _fake_manifest(30).splits["train"]
    first = ech._select_evaluation_groups(groups, group_count=10, seed=42)  # noqa: SLF001
    second = ech._select_evaluation_groups(groups, group_count=10, seed=42)  # noqa: SLF001
    assert [g.group_id for g in first] == [g.group_id for g in second]


def test_select_groups_different_seed_yields_different_groups():
    groups = _fake_manifest(30).splits["train"]
    first = ech._select_evaluation_groups(groups, group_count=10, seed=42)  # noqa: SLF001
    second = ech._select_evaluation_groups(groups, group_count=10, seed=7)  # noqa: SLF001
    assert [g.group_id for g in first] != [g.group_id for g in second]


def test_select_groups_twenty_groups_yield_exactly_eighty_episodes():
    groups = _fake_manifest(40).splits["train"]
    selected = ech._select_evaluation_groups(groups, group_count=20, seed=42)  # noqa: SLF001
    episodes = [eid for g in selected for eid in g.episode_ids]
    assert len(selected) == 20
    assert len(episodes) == 80


def test_select_groups_keeps_all_four_episodes_of_each_group():
    groups = _fake_manifest(40).splits["train"]
    selected = ech._select_evaluation_groups(groups, group_count=20, seed=42)  # noqa: SLF001
    for group in selected:
        assert len(group.episode_ids) == 4


def test_select_groups_has_no_duplicate_group_or_episode():
    groups = _fake_manifest(40).splits["train"]
    selected = ech._select_evaluation_groups(groups, group_count=20, seed=42)  # noqa: SLF001
    group_ids = [g.group_id for g in selected]
    episodes = [eid for g in selected for eid in g.episode_ids]
    assert len(group_ids) == len(set(group_ids))
    assert len(episodes) == len(set(episodes))


def test_select_groups_result_is_sorted_by_group_id():
    groups = _fake_manifest(40).splits["train"]
    selected = ech._select_evaluation_groups(groups, group_count=20, seed=42)  # noqa: SLF001
    group_ids = [g.group_id for g in selected]
    assert group_ids == sorted(group_ids)


def test_select_groups_count_exceeding_available_raises_without_truncation():
    groups = _fake_manifest(5).splits["train"]
    with pytest.raises(ValueError, match="only has 5"):
        ech._select_evaluation_groups(groups, group_count=6, seed=42)  # noqa: SLF001


def test_select_groups_negative_count_raises():
    groups = _fake_manifest(5).splits["train"]
    with pytest.raises(ValueError, match="non-negative"):
        ech._select_evaluation_groups(groups, group_count=-1, seed=42)  # noqa: SLF001


def test_select_groups_does_not_touch_global_random_state():
    random.seed(123)
    before = random.getstate()
    groups = _fake_manifest(40).splits["train"]
    ech._select_evaluation_groups(groups, group_count=20, seed=42)  # noqa: SLF001
    after = random.getstate()
    assert before == after


# --- _resolve_evaluation_scope (worker + parent share this) ----------------- #


def test_resolve_scope_test_split_ignores_group_count():
    manifest = _fake_manifest(40, n_test_groups=3)
    scope = ech._resolve_evaluation_scope(  # noqa: SLF001
        manifest, "test", train_group_count=20, train_group_seed=42
    )
    assert scope.is_full_split is True
    assert scope.requested_group_count == 0
    assert len(scope.group_ids) == 3
    assert len(scope.episode_ids) == 12


def test_resolve_scope_train_zero_is_full_split():
    manifest = _fake_manifest(40)
    scope = ech._resolve_evaluation_scope(  # noqa: SLF001
        manifest, "train", train_group_count=0, train_group_seed=42
    )
    assert scope.is_full_split is True
    assert scope.requested_group_count == 0
    assert len(scope.group_ids) == 40
    assert len(scope.episode_ids) == 160


def test_resolve_scope_train_sampled_is_partial_and_unsorted_input_safe():
    manifest = _fake_manifest(40)
    scope = ech._resolve_evaluation_scope(  # noqa: SLF001
        manifest, "train", train_group_count=20, train_group_seed=42
    )
    assert scope.is_full_split is False
    assert scope.requested_group_count == 20
    assert scope.group_ids == tuple(sorted(scope.group_ids))
    assert len(scope.episode_ids) == 80


def test_resolve_scope_episode_ids_come_only_from_selected_groups():
    manifest = _fake_manifest(40)
    scope = ech._resolve_evaluation_scope(  # noqa: SLF001
        manifest, "train", train_group_count=20, train_group_seed=42
    )
    selected = set(scope.group_ids)
    for group in manifest.splits["train"]:
        if group.group_id in selected:
            assert set(group.episode_ids).issubset(scope.episode_ids)
        else:
            assert not set(group.episode_ids).intersection(scope.episode_ids)


# --- worker command forwarding ---------------------------------------------- #


def test_worker_command_forwards_train_group_count_and_seed():
    args = _parse(
        [
            "--split",
            "train",
            "--train-group-count",
            "20",
            "--train-group-seed",
            "7",
            "--worker-checkpoint",
            "x",
            "--worker-output",
            "y",
        ]
    )
    command = ech._checkpoint_worker_command(  # noqa: SLF001
        args, checkpoint_dir=Path("/ckpt/200"), prediction_file=Path("/out/predictions.npz")
    )
    assert command[command.index("--train-group-count") + 1] == "20"
    assert command[command.index("--train-group-seed") + 1] == "7"


# --- npz scope round-trip + summary ----------------------------------------- #


def test_read_prediction_scope_arrays_roundtrip(tmp_path):
    path = tmp_path / "predictions.npz"
    _write_scoped_npz(path, group_ids=[3, 7], count=2, seed=9)
    scope = ech._read_prediction_scope_arrays(path)  # noqa: SLF001
    assert scope is not None
    assert scope["group_ids"] == (3, 7)
    assert scope["requested_count"] == 2
    assert scope["seed"] == 9
    assert len(scope["episode_ids"]) == 8


def test_read_prediction_scope_arrays_none_for_legacy_npz(tmp_path):
    path = tmp_path / "legacy.npz"
    _write_predictions(path, split="train")  # no scope arrays
    assert ech._read_prediction_scope_arrays(path) is None  # noqa: SLF001


def test_build_evaluation_scope_summary_from_sampled_npz(tmp_path):
    path = tmp_path / "predictions.npz"
    _write_scoped_npz(path, group_ids=[3, 7], count=2, seed=9)
    summary = ech._build_evaluation_scope_summary(path, split="train", train_group_seed=9)  # noqa: SLF001
    assert summary["split"] == "train"
    assert summary["requested_group_count"] == 2
    assert summary["selected_group_count"] == 2
    assert summary["selected_episode_count"] == 8
    assert summary["group_sample_seed"] == 9
    assert summary["selected_group_ids"] == [3, 7]
    assert summary["is_full_split"] is False


def test_build_evaluation_scope_summary_legacy_npz_is_full_split(tmp_path):
    path = tmp_path / "legacy.npz"
    _write_predictions(path, split="test")  # no scope arrays
    summary = ech._build_evaluation_scope_summary(path, split="test", train_group_seed=42)  # noqa: SLF001
    assert summary["is_full_split"] is True
    assert summary["selected_group_count"] is None
    assert summary["requested_group_count"] == 0


# --- resume scope validation ------------------------------------------------ #


def test_assert_scope_match_accepts_matching_scope(tmp_path):
    path = tmp_path / "predictions.npz"
    expected = _write_scoped_npz(path, group_ids=[0, 1, 2], count=3, seed=42)
    # No raise when npz scope matches the current request exactly.
    ech._assert_prediction_scope_matches(  # noqa: SLF001
        path,
        expected_split="train",
        train_group_count=3,
        train_group_seed=42,
        expected_scope=expected,
    )


def test_assert_scope_match_refuses_count_mismatch(tmp_path):
    path = tmp_path / "predictions.npz"
    _write_scoped_npz(path, group_ids=[0, 1], count=2, seed=42)
    expected = _scope([0, 1], list(range(8)), count=3, seed=42)  # request count=3
    with pytest.raises(ValueError, match="train_group_count=2"):
        ech._assert_prediction_scope_matches(  # noqa: SLF001
            path,
            expected_split="train",
            train_group_count=3,
            train_group_seed=42,
            expected_scope=expected,
        )


def test_assert_scope_match_refuses_seed_mismatch(tmp_path):
    path = tmp_path / "predictions.npz"
    _write_scoped_npz(path, group_ids=[0, 1], count=2, seed=42)
    expected = _scope([0, 1], list(range(8)), count=2, seed=7)  # request seed=7
    with pytest.raises(ValueError, match="train_group_seed=42"):
        ech._assert_prediction_scope_matches(  # noqa: SLF001
            path,
            expected_split="train",
            train_group_count=2,
            train_group_seed=7,
            expected_scope=expected,
        )


def test_assert_scope_match_refuses_group_id_mismatch(tmp_path):
    path = tmp_path / "predictions.npz"
    _write_scoped_npz(path, group_ids=[0, 1], count=2, seed=42)
    # Same count + seed, but the expected selection picked different groups.
    expected = _scope([2, 3], list(range(8, 16)), count=2, seed=42)
    with pytest.raises(ValueError, match="group IDs"):
        ech._assert_prediction_scope_matches(  # noqa: SLF001
            path,
            expected_split="train",
            train_group_count=2,
            train_group_seed=42,
            expected_scope=expected,
        )


def test_assert_scope_match_refuses_sampled_npz_resumed_as_full_split(tmp_path):
    path = tmp_path / "predictions.npz"
    _write_scoped_npz(path, group_ids=[0, 1], count=2, seed=42)
    with pytest.raises(ValueError, match="group-sampled run"):
        ech._assert_prediction_scope_matches(  # noqa: SLF001
            path,
            expected_split="train",
            train_group_count=0,
            train_group_seed=42,
            expected_scope=None,
        )


def test_assert_scope_match_refuses_sampled_request_on_scopeless_npz(tmp_path):
    path = tmp_path / "predictions.npz"
    # Has a split field but no scope arrays (older --split-only npz).
    _write_predictions(path, split="train")
    with pytest.raises(ValueError, match="lacks group-sampling scope"):
        ech._assert_prediction_scope_matches(  # noqa: SLF001
            path,
            expected_split="train",
            train_group_count=2,
            train_group_seed=42,
            expected_scope=_scope([0, 1], list(range(8)), count=2, seed=42),
        )


# --- metrics_scope label + no-hash guarantee -------------------------------- #


def test_metrics_scope_labels_present_in_source():
    source = Path(__file__).resolve().parent.joinpath("evaluate_completion_head.py").read_text(encoding="utf-8")
    assert '"selected_train_groups"' in source
    # The non-sampled label is an f-string template: f"full_{args.split}_split".
    assert "full_{args.split}_split" in source


def test_no_sha256_or_hashlib_in_source():
    source = Path(__file__).resolve().parent.joinpath("evaluate_completion_head.py").read_text(encoding="utf-8")
    assert "sha256" not in source.lower()
    assert "hashlib" not in source


# --- HTML train-subset rendering -------------------------------------------- #


def test_html_renders_train_subset_template_from_scope():
    manifest = {
        "checkpoint_step": 200,
        "split": "train",
        "created_at_utc": "2026-08-18T00:00:00Z",
        "dataset_root": "/data",
        "threshold": 0.5,
        "top_camera_key": ech.TOP_VIDEO_KEY,
        "evaluation_scope": {
            "split": "train",
            "requested_group_count": 20,
            "selected_group_count": 20,
            "selected_episode_count": 80,
            "group_sample_seed": 42,
            "selected_group_ids": list(range(20)),
            "selected_episode_ids": list(range(80)),
            "is_full_split": False,
        },
        "metrics_scope": "selected_train_groups",
        "episodes": [_fake_episode(0, 0)],
    }
    html = ech._html_document(manifest)  # noqa: SLF001
    # The JS template and the embedded scope values are both present.
    assert "Train subset:" in html
    assert "groups /" in html
    assert '"selected_group_count":20' in html
    assert '"is_full_split":false' in html


def test_html_has_train_subset_span_and_is_full_split_guard():
    html = _render_html(split="train")
    assert 'id="train-subset"' in html
    assert "is_full_split" in html
