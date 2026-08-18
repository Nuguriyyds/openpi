"""Offline evaluation and visualization for the single-frame completion head.

Evaluates one or more S2 completion-head checkpoints on a split selected from
the persisted leak-free manifest (``--split {train,val,test}``; default
``test``) and writes complete, independent reports (HTML + JSON + JSONL + CSV).
Pass ``--export-mp4`` to additionally render one annotated MP4 per episode.

The head outputs a single binary "is the episode about to end?" logit per
frame from the current frame's image and prompt alone (pi0.5 frozen prefix →
completion head). This evaluator never defines or regenerates completion
labels: it only reads the existing ``completion`` target that the labeled
dataset already carries. The positive/negative rule for each episode is a
property of the dataset (see ``openpi.training.completion_data``), not of this
script.

For ``--split train`` the evaluator answers two separate questions from one
inference pass over every train frame: (1) how the model scores the full
training trajectories (``metrics_all_frames``), and (2) whether it has fit the
frames actually sampled during training (``metrics_train_sampled``). The
training-sample membership is computed by reusing
``build_boundary_train_sample_set`` with the stride / forced-first-N values
taken from the training config, never by hand-rewriting the sampler rules.

For large train splits, ``--train-group-count`` samples whole ``TaskGroup``s
(four contiguous episodes each) directly from ``manifest.splits["train"]`` with
an independent ``--train-group-seed``. Selection happens before any video decode
or model forward, so inference, metrics, npz, and HTML all cover exactly the
selected episodes. This is distinct from ``--report-max-episodes``, which only
caps the HTML/MP4 visualization and never trims inference. ``metrics_all_frames``
then means "all frames in the selected groups", labelled ``metrics_scope =
"selected_train_groups"``.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
from datetime import UTC
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any, NamedTuple

import numpy as np

LOGGER = logging.getLogger("completion_head_evaluation")

DEFAULT_CONFIG_NAME = "pi05_agilex_breakfast_frozen_head_s2_completion_head"
DEFAULT_EXP_NAME = "s2_completion_head"
DEFAULT_HF_LEROBOT_HOME = Path("/mnt/data/models/wyt/data")
DEFAULT_CHECKPOINT_BASE = Path("/mnt/data/models/wyt/checkpoints")
DEFAULT_EVALUATION_BASE = Path("/mnt/data/models/wyt/evaluations")
DEFAULT_DATASET_ROOT = Path("/mnt/data/models/wyt/data/agilex_make_breakfast_subtask_730_frozen_head")
DEFAULT_CHECKPOINT_STEPS = ("200", "1000", "latest")
DEFAULT_SPLIT = "test"
SPLIT_CHOICES = ("train", "val", "test")
TOP_VIDEO_KEY = "observation.image.top"
LABEL_KEY = "completion"
# Single-frame model description shown in every report header. Hard-coded so
# the report can never accidentally advertise the temporal/three-frame scheme.
MODEL_INPUT_DESCRIPTION = "current-frame prefix only"
TRAIN_SAMPLE_RULE_DESCRIPTION = "all positives + stride-sampled negatives"


def _list_collate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keeps raw LeRobot samples separate for policy transforms in the parent."""

    return samples


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _setup_logging(output_dir: Path | None = None) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)
    if output_dir is not None:
        file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scalar(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _resolve_checkpoint_steps(checkpoint_root: Path, requested: list[str]) -> list[int]:
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint experiment directory not found: {checkpoint_root}")
    available = sorted(int(p.name) for p in checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit())
    if not available:
        raise FileNotFoundError(f"No numeric checkpoint directories found in: {checkpoint_root}")
    resolved: list[int] = []
    for value in requested:
        step = available[-1] if value.lower() == "latest" else int(value)
        if step not in available:
            raise FileNotFoundError(f"Checkpoint step {step} is unavailable; found {available}")
        if step not in resolved:
            resolved.append(step)
    return resolved


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    scores = np.empty_like(logits)
    pos = logits >= 0
    scores[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
    exp = np.exp(logits[~pos])
    scores[~pos] = exp / (1.0 + exp)
    return scores


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sorter = np.argsort(values, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.int64)
    inv[sorter] = np.arange(sorter.size, dtype=np.int64)
    sorted_values = values[sorter]
    is_tie_start = np.concatenate(([True], sorted_values[1:] != sorted_values[:-1]))
    sorted_group_ids = np.cumsum(is_tie_start) - 1
    group_ids = sorted_group_ids[inv]
    counts = np.bincount(sorted_group_ids)
    rank_sums = np.zeros(counts.size, dtype=np.float64)
    np.add.at(rank_sums, sorted_group_ids, np.arange(1, sorter.size + 1, dtype=np.float64))
    avg_ranks = rank_sums / np.maximum(counts, 1)
    return avg_ranks[group_ids]


def _best_threshold_f1(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float, float, float]:
    """Returns (best_f1, best_threshold, best_precision, best_recall)."""
    positives = targets == 1
    negatives = ~positives
    best_f1 = 0.0
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    for threshold in np.unique(scores):
        preds = scores >= threshold
        tp = int(np.sum(np.logical_and(preds, positives)))
        fp = int(np.sum(np.logical_and(preds, negatives)))
        fn = int(np.sum(np.logical_and(~preds, positives)))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2.0 * p * r / max(p + r, np.finfo(np.float64).eps)
        if f > best_f1:
            best_f1 = f
            best_threshold = float(threshold)
            best_precision = p
            best_recall = r
    return best_f1, best_threshold, best_precision, best_recall


def _roc_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    positives = targets == 1
    positive_count = int(np.sum(positives))
    negative_count = int(np.sum(~positives))
    if positive_count == 0 or negative_count == 0:
        return 0.5
    ranks = _rankdata(scores)
    return float(
        (np.sum(ranks[positives]) - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)
    )


def _frame_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    scores = _sigmoid(logits)
    bce = float(np.mean(np.logaddexp(0.0, logits) - targets * logits))
    positives = targets == 1
    negative = ~positives
    positive_count = int(np.sum(positives))
    negative_count = int(np.sum(negative))
    predictions_05 = scores >= 0.5
    tp_05 = int(np.sum(np.logical_and(predictions_05, positives)))
    fp_05 = int(np.sum(np.logical_and(predictions_05, negative)))
    fn_05 = int(np.sum(np.logical_and(~predictions_05, positives)))
    precision_05 = tp_05 / max(tp_05 + fp_05, 1)
    recall_05 = tp_05 / max(tp_05 + fn_05, 1)
    f1_05 = 2.0 * precision_05 * recall_05 / max(precision_05 + recall_05, np.finfo(np.float64).eps)
    best_f1, best_threshold, best_precision, best_recall = _best_threshold_f1(scores, targets)
    auc = _roc_auc(scores, targets)
    return {
        "frame_count": len(targets),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "bce": bce,
        "positive_score_mean": float(np.mean(scores[positives])) if positive_count else 0.0,
        "negative_score_mean": float(np.mean(scores[negative])) if negative_count else 0.0,
        "precision_at_0.5": precision_05,
        "recall_at_0.5": recall_05,
        "f1_at_0.5": f1_05,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "auc": auc,
    }


def _safe_percentile(values: np.ndarray, percentile: float) -> float | None:
    if len(values) == 0:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _average_precision(scores: np.ndarray, targets: np.ndarray) -> float | None:
    """Area under the precision-recall curve (average precision).

    Returns ``None`` when the slice contains only one class, so a degenerate
    train subset can never crash the report. Computed by the standard
    descending-score cumulative sum: AP = sum_k (R_k - R_{k-1}) * P_k.
    """

    positives = targets == 1
    negative_count = int(np.sum(~positives))
    positive_count = int(np.sum(positives))
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_targets = targets[order]
    cumulative_tp = np.cumsum(sorted_targets == 1)
    cumulative_fp = np.cumsum(sorted_targets == 0)
    recall = cumulative_tp / positive_count
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    # Average precision: sum over recall deltas of precision at each new
    # positive. Vectorized by only counting steps where recall increases.
    recall_delta = np.concatenate(([recall[0]], np.diff(recall)))
    return float(np.sum(recall_delta * precision))


def _train_fit_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Frame-level train-fit metrics with score-separation diagnostics.

    Unlike ``_frame_metrics`` (the fixed test/episode metric set reused by the
    boundary evaluator), this adds AUPRC, score percentiles, and the
    ``positive_median - negative_p95`` separation gap requested for judging
    whether the single-frame head has fit the training data. Single-class
    slices return ``None`` for AUPRC/ROC-AUC instead of a placeholder.
    """

    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    scores = _sigmoid(logits)
    positives = targets == 1
    negatives = ~positives
    positive_count = int(np.sum(positives))
    negative_count = int(np.sum(negatives))
    sample_count = len(targets)
    bce = float(np.mean(np.logaddexp(0.0, logits) - targets * logits)) if sample_count else 0.0

    positive_scores = scores[positives]
    negative_scores = scores[negatives]

    auprc = _average_precision(scores, targets)
    if positive_count == 0 or negative_count == 0:
        auc: float | None = None
    else:
        ranks = _rankdata(scores)
        auc = float(
            (np.sum(ranks[positives]) - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)
        )

    predictions_05 = scores >= 0.5
    tp_05 = int(np.sum(np.logical_and(predictions_05, positives)))
    fp_05 = int(np.sum(np.logical_and(predictions_05, negatives)))
    fn_05 = int(np.sum(np.logical_and(~predictions_05, positives)))
    tn_05 = int(np.sum(np.logical_and(~predictions_05, negatives)))
    precision_05 = tp_05 / max(tp_05 + fp_05, 1)
    recall_05 = tp_05 / max(tp_05 + fn_05, 1)
    f1_05 = 2.0 * precision_05 * recall_05 / max(precision_05 + recall_05, np.finfo(np.float64).eps)
    best_f1, best_threshold, best_precision, best_recall = _best_threshold_f1(scores, targets)

    positive_p50 = _safe_percentile(positive_scores, 50)
    negative_p95 = _safe_percentile(negative_scores, 95)
    positive_median_minus_negative_p95: float | None
    if positive_p50 is not None and negative_p95 is not None:
        positive_median_minus_negative_p95 = float(positive_p50 - negative_p95)
    else:
        positive_median_minus_negative_p95 = None

    return {
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_fraction": float(positive_count / sample_count) if sample_count else 0.0,
        "bce": bce,
        "auprc": auprc,
        "auc": auc,
        "precision_at_0.5": precision_05,
        "recall_at_0.5": recall_05,
        "f1_at_0.5": f1_05,
        "positive_score_mean": float(np.mean(positive_scores)) if positive_count else 0.0,
        "positive_score_p10": _safe_percentile(positive_scores, 10),
        "positive_score_p50": positive_p50,
        "positive_score_p90": _safe_percentile(positive_scores, 90),
        "negative_score_mean": float(np.mean(negative_scores)) if negative_count else 0.0,
        "negative_score_p90": _safe_percentile(negative_scores, 90),
        "negative_score_p95": negative_p95,
        "negative_score_p99": _safe_percentile(negative_scores, 99),
        "negative_score_max": float(np.max(negative_scores)) if negative_count else None,
        "positive_median_minus_negative_p95": positive_median_minus_negative_p95,
        # Diagnostic only — never used to choose the deployment threshold.
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "confusion_matrix": {"tp": tp_05, "fp": fp_05, "fn": fn_05, "tn": tn_05},
    }


def _first_crossing(values: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(values >= threshold)
    return None if len(indices) == 0 else int(indices[0])


def _episode_metric(
    episode_index: int,
    task_index: int,
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    fps: float,
    threshold: float,
) -> dict[str, Any]:
    scores = _sigmoid(logits)
    result: dict[str, Any] = {
        "episode_index": episode_index,
        "task_index": task_index,
        **_frame_metrics(logits, targets),
    }
    true_crossing = _first_crossing(targets, threshold)
    predicted_crossing = _first_crossing(scores, threshold)
    pre_completion = targets < threshold
    false_positive_count = int(np.sum((scores >= threshold) & pre_completion))
    result.update(
        {
            "true_threshold_frame": true_crossing,
            "predicted_threshold_frame": predicted_crossing,
            "never_triggered": predicted_crossing is None,
            "early_triggered": predicted_crossing is not None
            and true_crossing is not None
            and predicted_crossing < true_crossing,
            "pre_threshold_frame_count": int(np.sum(pre_completion)),
            "pre_threshold_false_positive_count": false_positive_count,
            "pre_threshold_false_positive_rate": float(false_positive_count / np.sum(pre_completion))
            if np.any(pre_completion)
            else 0.0,
        }
    )
    if predicted_crossing is None or true_crossing is None:
        result["detection_delay_frames"] = None
        result["detection_delay_seconds"] = None
    else:
        delay_frames = predicted_crossing - true_crossing
        result["detection_delay_frames"] = int(delay_frames)
        result["detection_delay_seconds"] = float(delay_frames / fps)
    return result


def _aggregate_episode_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pre_threshold_frames = sum(int(row["pre_threshold_frame_count"]) for row in rows)
    false_positives = sum(int(row["pre_threshold_false_positive_count"]) for row in rows)
    delays = [float(row["detection_delay_frames"]) for row in rows if row["detection_delay_frames"] is not None]
    delay_seconds = [
        float(row["detection_delay_seconds"]) for row in rows if row["detection_delay_seconds"] is not None
    ]
    return {
        "episode_count": len(rows),
        "early_trigger_episode_count": sum(bool(row["early_triggered"]) for row in rows),
        "early_trigger_rate": float(np.mean([bool(row["early_triggered"]) for row in rows])),
        "never_trigger_episode_count": sum(bool(row["never_triggered"]) for row in rows),
        "never_trigger_rate": float(np.mean([bool(row["never_triggered"]) for row in rows])),
        "pre_threshold_false_positive_rate": float(false_positives / pre_threshold_frames)
        if pre_threshold_frames
        else 0.0,
        "mean_detection_delay_frames": float(np.mean(delays)) if delays else None,
        "median_detection_delay_frames": float(np.median(delays)) if delays else None,
        "mean_detection_delay_seconds": float(np.mean(delay_seconds)) if delay_seconds else None,
        "median_detection_delay_seconds": float(np.median(delay_seconds)) if delay_seconds else None,
    }


def compute_metrics(prediction_file: Path, *, fps: float, threshold: float) -> dict[str, Any]:
    with np.load(prediction_file, allow_pickle=False) as values:
        episode_indices = values["episode_index"]
        task_indices = values["task_index"]
        frame_indices = values["frame_index"]
        logits = values["logit"]
        targets = values["target"]
        infer_ms = values["infer_ms"]

    arrays = {
        "episode_index": episode_indices,
        "task_index": task_indices,
        "frame_index": frame_indices,
        "logit": logits,
        "target": targets,
        "infer_ms": infer_ms,
    }
    row_counts = {name: len(value) for name, value in arrays.items()}
    if len(set(row_counts.values())) != 1 or not len(logits):
        raise ValueError(f"Prediction arrays have invalid row counts: {row_counts}")
    for name, values_arr in (("logit", logits), ("target", targets), ("infer_ms", infer_ms)):
        if not np.all(np.isfinite(values_arr)):
            raise ValueError(f"{name} contains non-finite values")
    if not np.all(np.logical_or(targets == 0.0, targets == 1.0)):
        raise ValueError("Completion targets must be binary 0/1")

    episode_rows: list[dict[str, Any]] = []
    for episode_index in sorted(np.unique(episode_indices).tolist()):
        mask = episode_indices == episode_index
        episode_task_indices = np.unique(task_indices[mask])
        if len(episode_task_indices) != 1:
            raise ValueError(f"Episode {episode_index}: expected one task index, got {episode_task_indices.tolist()}")
        episode_frame_indices = frame_indices[mask]
        order = np.argsort(episode_frame_indices)
        ordered_frame_indices = episode_frame_indices[order]
        expected_frame_indices = np.arange(len(ordered_frame_indices), dtype=ordered_frame_indices.dtype)
        if not np.array_equal(ordered_frame_indices, expected_frame_indices):
            raise ValueError(f"Episode {episode_index}: frame indices are not contiguous from zero")
        episode_rows.append(
            _episode_metric(
                int(episode_index),
                int(episode_task_indices[0]),
                logits[mask][order],
                targets[mask][order],
                fps=fps,
                threshold=threshold,
            )
        )

    per_task: dict[str, Any] = {}
    for task_index in sorted(np.unique(task_indices).tolist()):
        frame_mask = task_indices == task_index
        task_rows = [row for row in episode_rows if row["task_index"] == int(task_index)]
        per_task[str(int(task_index))] = {
            **_frame_metrics(logits[frame_mask], targets[frame_mask]),
            **_aggregate_episode_metrics(task_rows),
        }

    return {
        "overall": {
            **_frame_metrics(logits, targets),
            **_aggregate_episode_metrics(episode_rows),
            "mean_infer_ms_per_frame": float(np.mean(infer_ms)),
        },
        "per_task": per_task,
        "episodes": episode_rows,
    }


def _train_fit_metric_set(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    episode_indices: np.ndarray,
    task_indices: np.ndarray,
    frame_indices: np.ndarray,
) -> dict[str, Any]:
    """Overall + per-task + per-episode train-fit metrics for one frame slice.

    ``_train_fit_metrics`` is used at every level so AUPRC, score percentiles,
    and the ``positive_median - negative_p95`` gap are reported uniformly.
    Episodes are sorted by episode index; per-task keys are the task index as a
    string (matching ``compute_metrics``).
    """

    episode_rows: list[dict[str, Any]] = []
    for episode_index in sorted(np.unique(episode_indices).tolist()):
        mask = episode_indices == episode_index
        order = np.argsort(frame_indices[mask])
        episode_task_indices = np.unique(task_indices[mask])
        if len(episode_task_indices) != 1:
            raise ValueError(f"Episode {episode_index}: expected one task index, got {episode_task_indices.tolist()}")
        episode_rows.append(
            {
                "episode_index": int(episode_index),
                "task_index": int(episode_task_indices[0]),
                **_train_fit_metrics(logits[mask][order], targets[mask][order]),
            }
        )

    per_task: dict[str, Any] = {}
    for task_index in sorted(np.unique(task_indices).tolist()):
        frame_mask = task_indices == task_index
        per_task[str(int(task_index))] = _train_fit_metrics(logits[frame_mask], targets[frame_mask])

    return {
        "overall": _train_fit_metrics(logits, targets),
        "per_task": per_task,
        "episodes": episode_rows,
    }


def _compute_train_fit_metrics(prediction_file: Path) -> dict[str, Any]:
    """Two train-fit metric sets from one prediction file.

    ``metrics_all_frames`` scores every frame in the train split; the model is
    expected to separate positives from negatives well here. ``metrics_train_sampled``
    restricts to exactly the frames ``BoundaryCompletionSampler`` trained on
    (``is_train_sample == True``): if the head has memorized the sampled set, its
    numbers will look better here than on ``metrics_all_frames``. Both sets reuse
    ``_train_fit_metrics`` (AUPRC, percentiles, separation gap); single-class
    slices return ``None`` for AUPRC / ROC-AUC rather than a placeholder.
    """

    with np.load(prediction_file, allow_pickle=False) as values:
        episode_indices = values["episode_index"]
        task_indices = values["task_index"]
        frame_indices = values["frame_index"]
        logits = values["logit"]
        targets = values["target"]
        is_train_sample = values["is_train_sample"].astype(bool, copy=False)

    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    sampled = np.asarray(is_train_sample, dtype=bool)

    return {
        "metrics_all_frames": _train_fit_metric_set(
            logits,
            targets,
            episode_indices=episode_indices,
            task_indices=task_indices,
            frame_indices=frame_indices,
        ),
        "metrics_train_sampled": _train_fit_metric_set(
            logits[sampled],
            targets[sampled],
            episode_indices=episode_indices[sampled],
            task_indices=task_indices[sampled],
            frame_indices=frame_indices[sampled],
        ),
    }


# --------------------------------------------------------------------------- #
#  Boundary train-sample mask (reuses the training sampler exactly)            #
# --------------------------------------------------------------------------- #


def _build_boundary_train_sample_sets(
    dataset_meta: Any,
    episode_ids: list[int],
    *,
    label_key: str,
    stride: int,
    forced_first_n: int,
) -> dict[int, np.ndarray]:
    """Per-episode local frame indices that ``BoundaryCompletionSampler`` used.

    Mirrors ``evaluate_completion_boundary._build_sparse_sets`` but reads the
    dataset root, per-episode parquet path, and episode length from the already
    loaded ``LeRobotDatasetMetadata`` and forwards the dataset's
    ``boundary_excluded_episode_indices`` so short / excluded episodes audit
    exactly as they did during training. The sample set itself comes from
    ``build_boundary_train_sample_set`` (all positives + every-stride ordinary
    negatives + forced first-N negatives), never from a hand-rewritten rule.
    """

    from openpi.training import completion_data as _completion_data

    root = Path(dataset_meta.root)
    info = _read_json(root / "meta" / "info.json")
    excluded_episode_ids = tuple(int(value) for value in info.get("boundary_excluded_episode_indices", []))

    sample_sets: dict[int, np.ndarray] = {}
    for episode_id in episode_ids:
        parquet_path = root / dataset_meta.get_data_file_path(episode_id)
        expected_length = int(dataset_meta.episodes[episode_id]["length"])
        group_start = (episode_id // _completion_data.BOUNDARY_GROUP) * _completion_data.BOUNDARY_GROUP
        group_episode_ids = tuple(range(group_start, group_start + _completion_data.BOUNDARY_GROUP))
        group_position = episode_id % _completion_data.BOUNDARY_GROUP
        audit = _completion_data.audit_boundary_completion_episode_parquet(
            parquet_path,
            episode_id=episode_id,
            expected_length=expected_length,
            group_episode_ids=group_episode_ids,
            group_position=group_position,
            label_key=label_key,
            excluded_episode_ids=excluded_episode_ids,
        )
        sample_sets[episode_id] = _completion_data.build_boundary_train_sample_set(
            audit, stride=stride, forced_first_n=forced_first_n
        )
    return sample_sets


def _train_sample_mask(
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    sample_sets: dict[int, np.ndarray],
) -> np.ndarray:
    """Boolean mask selecting predictions that fall in the training sample sets."""

    mask = np.zeros(len(episode_indices), dtype=bool)
    for episode_id, sparse_frames in sample_sets.items():
        episode_mask = episode_indices == episode_id
        if not np.any(episode_mask):
            continue
        mask[episode_mask] = np.isin(frame_indices[episode_mask], sparse_frames)
    return mask


# --------------------------------------------------------------------------- #
#  Checkpoint evaluation worker (runs in a subprocess)                         #
# --------------------------------------------------------------------------- #


def _evaluation_repack():
    import openpi.transforms as transforms

    return transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "images": {
                        "cam_top": "observation.image.top",
                        "cam_left_wrist": "observation.image.left_wrist",
                        "cam_right_wrist": "observation.image.right_wrist",
                    },
                    "state": "observation.state.joint",
                    "gripper_position": "observation.gripper_position",
                    "prompt": "prompt",
                }
            )
        ]
    )


def _evaluate_checkpoint_worker(args: argparse.Namespace) -> int:
    """Load checkpoint, run compute_completion_logits on every test frame."""

    import jax
    import jax.numpy as jnp
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    import torch

    import openpi.models.model as model_api
    from openpi.policies import policy_config
    import openpi.shared.nnx_utils as nnx_utils
    from openpi.training import config as training_config

    checkpoint_dir = Path(args.worker_checkpoint).resolve()
    output_path = Path(args.worker_output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_path.parent)
    if not (checkpoint_dir / "params").is_dir():
        raise FileNotFoundError(f"Checkpoint params not found: {checkpoint_dir / 'params'}")

    config = training_config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Training config has no LeRobot repo_id")

    LOGGER.info("Loading checkpoint: %s", checkpoint_dir)
    # Use the policy infrastructure to load model + transforms. The policy
    # wraps the JAX model; we access the underlying module to call
    # compute_completion_logits directly (instead of sample_actions).
    policy = policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        repack_transforms=_evaluation_repack(),
        sample_kwargs={},
    )
    model = policy._model  # noqa: SLF001
    model.eval()

    # JIT-compile compute_completion_logits for speed. The first call compiles;
    # subsequent calls with the same shapes reuse the cached XLA program.
    # ``train`` must be a static argument because preprocess_observation and
    # the completion head both branch on it with a Python ``if train:``.
    compute_fn = nnx_utils.module_jit(model.compute_completion_logits, static_argnames="train")

    # Determine the episodes to evaluate from the split manifest. ``--split`` is
    # forwarded by the parent process; the boundary evaluator invokes this worker
    # without it, so fall back to ``test`` to preserve the old behavior. Group
    # sampling (--train-group-count) is resolved BEFORE building the episode list
    # so video decode, the model forward, npz, metrics, and HTML all cover exactly
    # the selected episodes (never the whole train split).
    split = getattr(args, "split", DEFAULT_SPLIT)
    train_group_count = getattr(args, "train_group_count", 0)
    train_group_seed = getattr(args, "train_group_seed", 42)
    if split not in SPLIT_CHOICES:
        raise ValueError(f"unsupported split {split!r}; expected one of {SPLIT_CHOICES}")
    if train_group_count != 0 and split != "train":
        raise ValueError(f"--train-group-count requires --split train, got split={split!r}")
    manifest = _load_split_manifest(args.config_name)
    scope = _resolve_evaluation_scope(
        manifest,
        split,
        train_group_count=train_group_count,
        train_group_seed=train_group_seed,
    )
    selected_groups = scope.groups
    episode_ids = list(scope.episode_ids)
    if split == "train" and not scope.is_full_split:
        LOGGER.info(
            "Selected %d / %d train groups (seed=%d)",
            len(selected_groups),
            len(manifest.splits["train"]),
            train_group_seed,
        )
        LOGGER.info("Selected group IDs: %s", list(scope.group_ids))
    LOGGER.info("Split=%s episodes (%d): %s", split, len(episode_ids), episode_ids)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id)
    tasks = dataset_meta.tasks

    episode_from = dataset.episode_data_index["from"]
    episode_to = dataset.episode_data_index["to"]

    frame_specs: list[tuple[int, int, int]] = []  # (episode_index, local_frame, dataset_index)
    for episode_index in episode_ids:
        start = int(episode_from[episode_index])
        end = int(episode_to[episode_index])
        frame_specs.extend((episode_index, local_frame, start + local_frame) for local_frame in range(end - start))

    # For the train split, mark which frames the BoundaryCompletionSampler
    # actually trained on. The mask is computed by reusing the training sampler
    # (build_boundary_train_sample_set) with the config's stride / forced-first-N
    # values, so it can never drift from what training saw. val/test frames are
    # never training samples, so the mask is left empty and filled with False.
    train_sample_sets: dict[int, np.ndarray] = {}
    if split == "train":
        train_sample_sets = _build_boundary_train_sample_sets(
            dataset_meta,
            episode_ids,
            label_key=config.completion.label_key,
            stride=config.completion.negative_stride,
            forced_first_n=config.completion.boundary_copy_frames,
        )

    LOGGER.info(
        "Evaluating %d %s-split episodes (%d frames), batch_size=%d, data_workers=%d",
        len(episode_ids),
        split,
        len(frame_specs),
        args.batch_size,
        args.num_workers,
    )

    # LeRobot video access is the dominant cost on remote/OSS storage. The old
    # evaluator called dataset[index] serially in the model process, leaving
    # all GPUs idle while three videos were opened and decoded per frame.
    # DataLoader preserves sampler order while overlapping those reads across
    # workers. Policy transforms remain in this process because the policy/JAX
    # objects are not safe to pickle into worker processes.
    ordered_dataset_indices = [dataset_index for _, _, dataset_index in frame_specs]
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "sampler": ordered_dataset_indices,
        "num_workers": args.num_workers,
        "collate_fn": _list_collate,
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        # JAX owns background threads by this point. Forking a multithreaded
        # JAX process can deadlock; fresh spawned interpreters safely isolate
        # video decoding workers.
        loader_kwargs["multiprocessing_context"] = "spawn"
    frame_loader = torch.utils.data.DataLoader(**loader_kwargs)

    result_episode_indices: list[int] = []
    result_task_indices: list[int] = []
    result_frame_indices: list[int] = []
    result_logits: list[float] = []
    result_targets: list[float] = []
    result_infer_ms: list[float] = []

    rng = jax.random.key(args.seed)

    for batch_number, raw_samples in enumerate(frame_loader):
        batch_start = batch_number * args.batch_size
        valid_specs = frame_specs[batch_start : batch_start + args.batch_size]
        transformed_items: list[dict[str, Any]] = []
        batch_metadata: list[tuple[int, int, int, float]] = []

        if len(raw_samples) != len(valid_specs):
            raise RuntimeError(
                f"Data loader returned {len(raw_samples)} samples for {len(valid_specs)} ordered frame specifications"
            )
        for (episode_index, local_frame_index, _dataset_index), raw_sample in zip(
            valid_specs, raw_samples, strict=True
        ):
            sample = dict(raw_sample)
            task_index = _scalar(sample["task_index"])
            prompt = tasks.get(task_index)
            if prompt is None:
                raise ValueError(f"Task {task_index} is missing from dataset metadata")
            sample["prompt"] = prompt
            target = float(np.asarray(sample[LABEL_KEY]).reshape(-1)[0])
            transformed_items.append(policy._input_transform(sample))  # noqa: SLF001
            batch_metadata.append((episode_index, task_index, local_frame_index, target))

        valid_count = len(transformed_items)
        # Pad the final batch with repeats of the last item so the batch size
        # divides evenly across devices.
        while len(transformed_items) < args.batch_size:
            transformed_items.append(jax.tree.map(lambda v: np.array(v, copy=True), transformed_items[-1]))

        batched_inputs = jax.tree.map(
            lambda *values: jnp.asarray(np.stack([np.asarray(v) for v in values], axis=0)),
            *transformed_items,
        )
        observation = model_api.Observation.from_dict(batched_inputs)

        started = time.monotonic()
        logits = compute_fn(rng, observation, train=False)
        logits = np.asarray(jax.block_until_ready(logits))
        elapsed_ms = (time.monotonic() - started) * 1000.0 / valid_count

        for batch_index, (episode_index, task_index, local_frame_index, target) in enumerate(batch_metadata):
            result_episode_indices.append(episode_index)
            result_task_indices.append(task_index)
            result_frame_indices.append(local_frame_index)
            result_logits.append(float(logits[batch_index]))
            result_targets.append(target)
            result_infer_ms.append(elapsed_ms)

        completed = min(batch_start + valid_count, len(frame_specs))
        if completed % 100 <= valid_count or completed == len(frame_specs):
            LOGGER.info("Checkpoint %s: %d/%d frames", checkpoint_dir.name, completed, len(frame_specs))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    episode_index_array = np.asarray(result_episode_indices, dtype=np.int32)
    frame_index_array = np.asarray(result_frame_indices, dtype=np.int32)
    is_train_sample = _train_sample_mask(
        episode_index_array,
        frame_index_array,
        train_sample_sets,
    )
    if split == "train":
        sampled_count = int(np.sum(is_train_sample))
        LOGGER.info(
            "Train-sample mask: %d / %d frames are training samples",
            sampled_count,
            len(is_train_sample),
        )
    with tempfile.TemporaryDirectory(prefix="completion-eval-") as tmp:
        local_output = Path(tmp) / "predictions.npz"
        np.savez_compressed(
            local_output,
            episode_index=episode_index_array,
            task_index=np.asarray(result_task_indices, dtype=np.int16),
            frame_index=frame_index_array,
            logit=np.asarray(result_logits, dtype=np.float32),
            target=np.asarray(result_targets, dtype=np.float32),
            infer_ms=np.asarray(result_infer_ms, dtype=np.float32),
            is_train_sample=is_train_sample,
            split=np.asarray(split),
            selected_group_ids=np.asarray(scope.group_ids, dtype=np.int64),
            selected_episode_ids=np.asarray(scope.episode_ids, dtype=np.int64),
            requested_train_group_count=np.asarray(scope.requested_group_count, dtype=np.int64),
            train_group_seed=np.asarray(scope.group_seed, dtype=np.int64),
        )
        with local_output.open("rb") as src, output_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        if local_output.stat().st_size != output_path.stat().st_size:
            raise OSError("Prediction file copy size mismatch")
    LOGGER.info("Saved checkpoint predictions: %s", output_path)
    return 0


# --------------------------------------------------------------------------- #
#  Report series (for HTML / MP4 visualisation)                                #
# --------------------------------------------------------------------------- #


def _format_dataset_path(pattern: str, episode_index: int, chunk_size: int, video_key: str) -> Path:
    return Path(
        pattern.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
            video_key=video_key,
        )
    )


def _load_report_series(
    dataset_root: Path,
    prediction_file: Path,
    episode_metrics: list[dict[str, Any]],
    *,
    output_dir: Path,
    copy_videos: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Builds per-episode series for the HTML / MP4 reports.

    When ``copy_videos`` is true (the default), video files are copied into
    ``<output_dir>/videos/`` and referenced via a relative path. Otherwise the
    source video is referenced via a ``file://`` URI.
    """

    meta_dir = dataset_root / "meta"
    info = _read_json(meta_dir / "info.json")
    task_rows = _read_jsonl(meta_dir / "tasks.jsonl")
    episode_rows = {int(row["episode_index"]): row for row in _read_jsonl(meta_dir / "episodes.jsonl")}
    tasks = {int(row["task_index"]): str(row["task"]) for row in task_rows}
    metrics_by_episode = {int(row["episode_index"]): row for row in episode_metrics}
    fps = float(info["fps"])
    chunk_size = int(info["chunks_size"])

    with np.load(prediction_file, allow_pickle=False) as values:
        episode_indices = values["episode_index"]
        task_indices = values["task_index"]
        frame_indices = values["frame_index"]
        logits = values["logit"]
        targets = values["target"]
        # Old (pre-split) prediction files lack the training-sample mask; treat
        # every frame as a non-sample so the report still renders. Train-mode
        # resume refuses such files earlier, so reaching here all-false is safe.
        if "is_train_sample" in values.files:
            is_train_sample = values["is_train_sample"].astype(bool, copy=False)
        else:
            is_train_sample = np.zeros(len(targets), dtype=bool)

    videos_dir = output_dir / "videos"
    if copy_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)

    copied_videos = 0
    series: list[dict[str, Any]] = []
    for episode_index in sorted(metrics_by_episode.keys()):
        mask = episode_indices == episode_index
        frame_count = int(np.sum(mask))
        if episode_index not in episode_rows:
            raise ValueError(f"Episode {episode_index}: missing from episodes.jsonl")
        expected_length = int(episode_rows[episode_index]["length"])
        if frame_count != expected_length:
            raise ValueError(f"Episode {episode_index}: prediction rows={frame_count}, dataset rows={expected_length}")
        episode_task_indices = np.unique(task_indices[mask])
        if len(episode_task_indices) != 1:
            raise ValueError(f"Episode {episode_index}: expected one task index")
        task_index = int(episode_task_indices[0])
        if task_index not in tasks:
            raise ValueError(f"Episode {episode_index}: task {task_index} missing from metadata")
        order = np.argsort(frame_indices[mask])
        source_video = dataset_root / _format_dataset_path(
            str(info["video_path"]),
            episode_index,
            chunk_size,
            video_key=TOP_VIDEO_KEY,
        )
        if not source_video.is_file():
            raise FileNotFoundError(f"Top-camera video not found: {source_video}")
        if copy_videos:
            destination = videos_dir / f"episode_{episode_index:06d}.mp4"
            # Idempotent: skip re-copying on --resume so regenerating a report
            # (e.g. changing --report-max-episodes) never re-reads the source videos.
            if not destination.is_file():
                shutil.copy2(source_video, destination)
                copied_videos += 1
            video_path = destination.relative_to(output_dir).as_posix()
        else:
            video_path = source_video.as_uri()

        episode_logits = logits[mask][order]
        episode_scores = _sigmoid(episode_logits)
        episode_targets = targets[mask][order]
        episode_sampled = is_train_sample[mask][order]
        # Local frame indices (0..frame_count-1, the plot x-axis positions) that
        # the BoundaryCompletionSampler trained on, split by their target label
        # so the HTML can draw sampled positives and sampled negatives with
        # distinct markers. Empty for val/test (mask is all False there).
        sampled_positive_frames = np.flatnonzero(episode_sampled & (episode_targets == 1)).astype(int).tolist()
        sampled_negative_frames = np.flatnonzero(episode_sampled & (episode_targets == 0)).astype(int).tolist()
        series.append(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "prompt": tasks[task_index],
                "fps": fps,
                "frame_count": frame_count,
                "video": video_path,
                "source_video": str(source_video),
                "score": episode_scores.astype(float).tolist(),
                "target": episode_targets.astype(float).tolist(),
                "sampled_positive_frames": sampled_positive_frames,
                "sampled_negative_frames": sampled_negative_frames,
                "metrics": metrics_by_episode[episode_index],
            }
        )
    if copy_videos:
        LOGGER.info(
            "Report videos: copied %d, reused %d already-present",
            copied_videos,
            len(series) - copied_videos,
        )
    return series, info


# --------------------------------------------------------------------------- #
#  HTML report                                                                 #
# --------------------------------------------------------------------------- #


def _html_document(manifest: dict[str, Any]) -> str:
    report_json = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    report_json = report_json.replace("</", "<\\/")
    template = textwrap.dedent(
        """\
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Completion head evaluation</title>
          <style>
            :root {
              color-scheme: dark;
              font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
              background: #0b1020;
              color: #eef2ff;
            }
            * { box-sizing: border-box; }
            body { margin: 0; background: #0b1020; }
            main { width: min(1180px, 100%); margin: 0 auto; padding: 22px; }
            h1 { margin: 0; font-size: 1.45rem; font-weight: 650; }
            .subtitle { margin: 6px 0 20px; color: #aab5d1; }
            .toolbar {
              display: flex; align-items: end; gap: 10px; flex-wrap: wrap; margin-bottom: 16px;
            }
            .field { display: grid; gap: 5px; flex: 1 1 520px; }
            label { color: #aab5d1; font-size: .85rem; }
            select, button {
              border: 1px solid #33405f; border-radius: 8px; background: #151d32; color: #eef2ff;
              min-height: 38px; padding: 8px 11px; font: inherit;
            }
            button { cursor: pointer; }
            button:disabled { cursor: default; opacity: .45; }
            button:hover:not(:disabled) { background: #202b47; }
            .viewer {
              display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(300px, .65fr);
              gap: 18px; align-items: start;
            }
            video {
              display: block; width: 100%; max-height: 68vh; border-radius: 10px; background: #03050a;
            }
            .details {
              border: 1px solid #26324d; border-radius: 10px; background: #10172a; padding: 15px;
            }
            .prompt { margin: 0 0 14px; line-height: 1.45; white-space: pre-line; }
            .live {
              display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-bottom: 14px;
            }
            .live div { padding: 9px; border-radius: 8px; background: #182139; }
            .live span { display: block; color: #aab5d1; font-size: .78rem; margin-bottom: 3px; }
            .live strong { font-variant-numeric: tabular-nums; font-weight: 650; }
            table { border-collapse: collapse; width: 100%; font-size: .88rem; }
            th, td { border-bottom: 1px solid #26324d; padding: 7px 2px; text-align: left; }
            th { color: #aab5d1; font-weight: 450; }
            td { text-align: right; font-variant-numeric: tabular-nums; }
            .plot-wrap { margin-top: 18px; }
            .legend { display: flex; flex-wrap: wrap; gap: 15px; color: #aab5d1; font-size: .86rem; }
            .legend span::before {
              content: ""; display: inline-block; width: 18px; height: 3px; margin: 0 6px 3px 0; background: var(--line);
            }
            .legend span.marker::before {
              width: 10px; height: 10px; border-radius: 50%; margin: 0 8px 1px 0; background: var(--marker);
            }
            .legend span.marker.square::before { border-radius: 2px; }
            canvas {
              display: block; width: 100%; height: 290px; margin-top: 8px;
              border-radius: 10px; background: #10172a; cursor: crosshair;
            }
            .hint { color: #8490ad; font-size: .8rem; margin: 7px 0 0; }
            @media (max-width: 850px) {
              main { padding: 14px; }
              .viewer { grid-template-columns: 1fr; }
              video { max-height: none; }
            }
          </style>
        </head>
        <body>
          <main>
            <h1>Checkpoint <span id="checkpoint"></span> | completion head</h1>
            <p class="subtitle">
              Split: <strong id="split-label"></strong><span id="train-subset"></span>
              &middot; Model input: current-frame prefix only
              &middot; Training sample rule: all positives + stride-sampled negatives
            </p>
            <div class="toolbar">
              <div class="field">
                <label for="episode-select">Episode</label>
                <select id="episode-select"></select>
              </div>
              <button id="previous" type="button">Previous</button>
              <button id="next" type="button">Next</button>
            </div>
            <section class="viewer">
              <video id="video" controls preload="metadata"></video>
              <aside class="details">
                <p class="prompt" id="prompt"></p>
                <div class="live">
                  <div><span>Frame</span><strong id="frame-value">0 / 0</strong></div>
                  <div><span>Target</span><strong id="target-value">0</strong></div>
                  <div><span>Score</span><strong id="score-value">0.000</strong></div>
                </div>
                <table aria-label="Episode metrics"><tbody id="metrics"></tbody></table>
              </aside>
            </section>
            <section class="plot-wrap">
              <div class="legend">
                <span style="--line:#66d9a5">Target (0/1)</span>
                <span style="--line:#ffad5a">Predicted score (sigmoid)</span>
                <span class="marker" style="--marker:#66d9a5">Sampled positive</span>
                <span class="marker square" style="--marker:#5a8cff">Sampled negative</span>
                <span style="--line:#b49cff">Threshold</span>
              </div>
              <canvas id="plot" aria-label="Score and target by frame"></canvas>
              <p class="hint">The vertical cursor follows the video. Click the curve to seek.</p>
            </section>
          </main>
          <script>
            const REPORT = __REPORT_DATA__;
            const video = document.getElementById("video");
            const select = document.getElementById("episode-select");
            const canvas = document.getElementById("plot");
            const context = canvas.getContext("2d");
            let selectedIndex = 0;
            let currentFrame = 0;
            let animationRequest = null;

            function formatNumber(value, digits) {
              if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
              return Number(value).toFixed(digits);
            }

            function addMetric(label, value) {
              const row = document.createElement("tr");
              const heading = document.createElement("th");
              const cell = document.createElement("td");
              heading.scope = "row";
              heading.textContent = label;
              cell.textContent = value;
              row.append(heading, cell);
              document.getElementById("metrics").append(row);
            }

            function showMetrics(episode) {
              const m = episode.metrics;
              const body = document.getElementById("metrics");
              body.replaceChildren();
              addMetric("AUC", formatNumber(m.auc, 4));
              addMetric("Best F1", formatNumber(m.best_f1, 4));
              addMetric("Best threshold", formatNumber(m.best_threshold, 4));
              addMetric("Best precision", formatNumber(m.best_precision, 4));
              addMetric("Best recall", formatNumber(m.best_recall, 4));
              addMetric("F1 @ 0.5", formatNumber(m["f1_at_0.5"], 4));
              addMetric("Precision @ 0.5", formatNumber(m["precision_at_0.5"], 4));
              addMetric("Recall @ 0.5", formatNumber(m["recall_at_0.5"], 4));
              addMetric("BCE", formatNumber(m.bce, 4));
              addMetric("Positive score mean", formatNumber(m.positive_score_mean, 4));
              addMetric("Negative score mean", formatNumber(m.negative_score_mean, 4));
              addMetric("Early trigger", m.early_triggered ? "yes" : "no");
              addMetric("Never triggered", m.never_triggered ? "yes" : "no");
              addMetric("Detection delay", m.detection_delay_seconds === null
                ? "n/a"
                : formatNumber(m.detection_delay_seconds, 3) + " s");
            }

            function plotGeometry(width, height) {
              return { left: 54, top: 20, right: width - 18, bottom: height - 38 };
            }

            function drawPlot(frame) {
              const episode = REPORT.episodes[selectedIndex];
              const ratio = Math.max(1, window.devicePixelRatio || 1);
              const width = Math.max(320, Math.round(canvas.clientWidth));
              const height = Math.max(220, Math.round(canvas.clientHeight));
              if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
                canvas.width = Math.round(width * ratio);
                canvas.height = Math.round(height * ratio);
              }
              context.setTransform(ratio, 0, 0, ratio, 0, 0);
              context.clearRect(0, 0, width, height);
              context.fillStyle = "#10172a";
              context.fillRect(0, 0, width, height);

              const geometry = plotGeometry(width, height);
              const x = function(index) {
                return geometry.left + index / Math.max(1, episode.frame_count - 1) * (geometry.right - geometry.left);
              };
              const y = function(value) {
                return geometry.bottom - (value - 0) / Math.max(1e-8, 1.0 - 0) * (geometry.bottom - geometry.top);
              };

              context.font = "12px system-ui, sans-serif";
              context.lineWidth = 1;
              [0, 0.25, 0.5, 0.75, 1].forEach(function(value) {
                const ordinate = y(value);
                context.strokeStyle = "#28344f";
                context.beginPath();
                context.moveTo(geometry.left, ordinate);
                context.lineTo(geometry.right, ordinate);
                context.stroke();
                context.fillStyle = "#96a2bf";
                context.textAlign = "right";
                context.textBaseline = "middle";
                context.fillText(value.toFixed(2), geometry.left - 8, ordinate);
              });
              [0, Math.floor((episode.frame_count - 1) / 2), episode.frame_count - 1].forEach(function(value) {
                context.fillStyle = "#96a2bf";
                context.textAlign = "center";
                context.textBaseline = "top";
                context.fillText(String(value), x(value), geometry.bottom + 9);
              });

              context.save();
              context.setLineDash([6, 5]);
              context.strokeStyle = "#b49cff";
              context.beginPath();
              context.moveTo(geometry.left, y(REPORT.threshold));
              context.lineTo(geometry.right, y(REPORT.threshold));
              context.stroke();
              context.restore();

              function drawSeries(values, color, widthValue) {
                context.strokeStyle = color;
                context.lineWidth = widthValue;
                context.beginPath();
                values.forEach(function(value, index) {
                  if (index === 0) context.moveTo(x(index), y(value));
                  else context.lineTo(x(index), y(value));
                });
                context.stroke();
              }
              drawSeries(episode.target, "#66d9a5", 2);
              drawSeries(episode.score, "#ffad5a", 2);

              // Sampled training frames (train split only): a solid marker on
              // the score curve for sampled positives, a different color/shape
              // (square) for sampled negatives. Non-sampled frames stay as
              // curve only. Both lists are empty for val/test splits.
              function drawSampled(frames, color, shape) {
                if (!frames) return;
                frames.forEach(function(index) {
                  var px = x(index), py = y(episode.score[index]);
                  context.beginPath();
                  context.fillStyle = color;
                  if (shape === "square") {
                    context.rect(px - 3.5, py - 3.5, 7, 7);
                  } else {
                    context.arc(px, py, 4, 0, Math.PI * 2);
                  }
                  context.fill();
                });
              }
              drawSampled(episode.sampled_positive_frames, "#66d9a5", "circle");
              drawSampled(episode.sampled_negative_frames, "#5a8cff", "square");

              const boundedFrame = Math.max(0, Math.min(episode.frame_count - 1, frame));
              const cursorX = x(boundedFrame);
              context.strokeStyle = "#eef2ff";
              context.lineWidth = 1;
              context.beginPath();
              context.moveTo(cursorX, geometry.top);
              context.lineTo(cursorX, geometry.bottom);
              context.stroke();
              [["#66d9a5", episode.target[boundedFrame]], ["#ffad5a", episode.score[boundedFrame]]]
                .forEach(function(item) {
                  context.beginPath();
                  context.fillStyle = item[0];
                  context.arc(cursorX, y(item[1]), 4, 0, Math.PI * 2);
                  context.fill();
                });
            }

            function updateLive(frame) {
              const episode = REPORT.episodes[selectedIndex];
              currentFrame = Math.max(0, Math.min(episode.frame_count - 1, frame));
              document.getElementById("frame-value").textContent =
                String(currentFrame) + " / " + String(episode.frame_count - 1);
              document.getElementById("target-value").textContent =
                String(episode.target[currentFrame]);
              document.getElementById("score-value").textContent =
                formatNumber(episode.score[currentFrame], 3);
              drawPlot(currentFrame);
            }

            function syncToVideo() {
              const episode = REPORT.episodes[selectedIndex];
              updateLive(Math.floor(video.currentTime * episode.fps + 1e-6));
            }

            function followPlayback() {
              syncToVideo();
              if (!video.paused && !video.ended) {
                animationRequest = window.requestAnimationFrame(followPlayback);
              }
            }

            function stopFollowingPlayback() {
              if (animationRequest !== null) {
                window.cancelAnimationFrame(animationRequest);
              }
              animationRequest = null;
              syncToVideo();
            }

            function loadEpisode(index) {
              selectedIndex = Math.max(0, Math.min(REPORT.episodes.length - 1, index));
              const episode = REPORT.episodes[selectedIndex];
              select.value = String(selectedIndex);
              video.src = episode.video;
              video.load();
              document.getElementById("prompt").textContent =
                "Episode " + episode.episode_index + "\\n" + episode.prompt;
              showMetrics(episode);
              document.getElementById("previous").disabled = selectedIndex === 0;
              document.getElementById("next").disabled = selectedIndex === REPORT.episodes.length - 1;
              updateLive(0);
            }

            REPORT.episodes.forEach(function(episode, index) {
              const option = document.createElement("option");
              option.value = String(index);
              option.textContent = "Episode " + episode.episode_index
                + " | " + episode.prompt.substring(0, 50);
              select.append(option);
            });
            document.getElementById("checkpoint").textContent = String(REPORT.checkpoint_step);
            document.getElementById("split-label").textContent = String(REPORT.split || "test");
            var scope = REPORT.evaluation_scope;
            var subset = document.getElementById("train-subset");
            if (scope && !scope.is_full_split && scope.split === "train") {
              subset.textContent = " · Train subset: " + scope.selected_group_count
                + " groups / " + scope.selected_episode_count + " episodes";
            }
            select.addEventListener("change", function() { loadEpisode(Number(select.value)); });
            document.getElementById("previous").addEventListener("click", function() {
              loadEpisode(selectedIndex - 1);
            });
            document.getElementById("next").addEventListener("click", function() {
              loadEpisode(selectedIndex + 1);
            });
            video.addEventListener("timeupdate", syncToVideo);
            video.addEventListener("seeked", syncToVideo);
            video.addEventListener("loadedmetadata", syncToVideo);
            video.addEventListener("play", function() {
              if (animationRequest !== null) {
                window.cancelAnimationFrame(animationRequest);
              }
              followPlayback();
            });
            video.addEventListener("pause", stopFollowingPlayback);
            video.addEventListener("ended", stopFollowingPlayback);
            canvas.addEventListener("click", function(event) {
              const episode = REPORT.episodes[selectedIndex];
              const rectangle = canvas.getBoundingClientRect();
              const geometry = plotGeometry(rectangle.width, rectangle.height);
              const clickX = Math.max(geometry.left, Math.min(geometry.right, event.clientX - rectangle.left));
              const frame = Math.round((clickX - geometry.left) / Math.max(1, geometry.right - geometry.left)
                * (episode.frame_count - 1));
              video.currentTime = frame / episode.fps;
              updateLive(frame);
            });
            window.addEventListener("resize", function() { drawPlot(currentFrame); });
            loadEpisode(0);
          </script>
        </body>
        </html>
        """
    )
    return template.replace("__REPORT_DATA__", report_json)


def _write_html_report(report_dir: Path, manifest: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_dir / "manifest.json", manifest)
    (report_dir / "index.html").write_text(_html_document(manifest), encoding="utf-8")


# --------------------------------------------------------------------------- #
#  MP4 export                                                                  #
# --------------------------------------------------------------------------- #


def _plot_points(
    values: np.ndarray,
    *,
    left: int,
    right: int,
    top: int,
    bottom: int,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    if len(values) == 1:
        x_values = np.asarray([left], dtype=np.float64)
    else:
        x_values = np.linspace(left, right, len(values), dtype=np.float64)
    y_values = bottom - (values - minimum) / max(1e-8, maximum - minimum) * (bottom - top)
    return np.rint(np.column_stack([x_values, y_values])).astype(np.int32).reshape(-1, 1, 2)


def _make_mp4_plot_base(
    episode: dict[str, Any],
    *,
    width: int,
    height: int,
    checkpoint_step: int,
    threshold: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    import cv2

    background = np.zeros((height, width, 3), dtype=np.uint8)
    background[:] = (31, 23, 16)
    score = np.asarray(episode["score"], dtype=np.float64)
    target = np.asarray(episode["target"], dtype=np.float64)
    minimum = 0.0
    maximum = 1.0
    left = 58
    right = width - 18
    top = 62
    bottom = height - 35

    title = f"Checkpoint {checkpoint_step} | episode {episode['episode_index']}"
    cv2.putText(background, title, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (238, 242, 255), 1, cv2.LINE_AA)
    cv2.line(background, (12, 41), (29, 41), (165, 217, 102), 3, cv2.LINE_AA)
    cv2.putText(background, "target", (34, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 201, 224), 1, cv2.LINE_AA)
    cv2.line(background, (94, 41), (111, 41), (90, 173, 255), 3, cv2.LINE_AA)
    cv2.putText(background, "score", (116, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 201, 224), 1, cv2.LINE_AA)

    def y_coordinate(value: float) -> int:
        return int(round(bottom - (value - minimum) / max(1e-8, maximum - minimum) * (bottom - top)))

    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        y_value = y_coordinate(value)
        cv2.line(background, (left, y_value), (right, y_value), (70, 52, 40), 1, cv2.LINE_AA)
        cv2.putText(
            background,
            f"{value:.2f}",
            (7, y_value + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (191, 175, 151),
            1,
            cv2.LINE_AA,
        )

    threshold_y = y_coordinate(threshold)
    for start in range(left, right, 14):
        cv2.line(background, (start, threshold_y), (min(start + 7, right), threshold_y), (255, 156, 180), 1)
    target_points = _plot_points(
        target, left=left, right=right, top=top, bottom=bottom, minimum=minimum, maximum=maximum
    )
    score_points = _plot_points(score, left=left, right=right, top=top, bottom=bottom, minimum=minimum, maximum=maximum)
    cv2.polylines(background, [target_points], False, (165, 217, 102), 2, cv2.LINE_AA)
    cv2.polylines(background, [score_points], False, (90, 173, 255), 2, cv2.LINE_AA)

    final_frame = int(episode["frame_count"]) - 1
    for frame in (0, final_frame // 2, final_frame):
        x_value = int(round(left + frame / max(1, final_frame) * (right - left)))
        label = str(frame)
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        cv2.putText(
            background,
            label,
            (x_value - text_width // 2, height - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (191, 175, 151),
            1,
            cv2.LINE_AA,
        )

    return background, {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
        "minimum": minimum,
        "maximum": maximum,
    }


def _draw_mp4_plot_frame(
    base: np.ndarray,
    geometry: dict[str, float | int],
    episode: dict[str, Any],
    frame_index: int,
) -> np.ndarray:
    import cv2

    panel = base.copy()
    score = np.asarray(episode["score"], dtype=np.float64)
    target = np.asarray(episode["target"], dtype=np.float64)
    frame_index = max(0, min(len(score) - 1, frame_index))
    left = int(geometry["left"])
    right = int(geometry["right"])
    top = int(geometry["top"])
    bottom = int(geometry["bottom"])
    minimum = float(geometry["minimum"])
    maximum = float(geometry["maximum"])
    x_value = int(round(left + frame_index / max(1, len(score) - 1) * (right - left)))

    def y_coordinate(value: float) -> int:
        return int(round(bottom - (value - minimum) / max(1e-8, maximum - minimum) * (bottom - top)))

    cv2.line(panel, (x_value, top), (x_value, bottom), (245, 242, 238), 1, cv2.LINE_AA)
    cv2.circle(panel, (x_value, y_coordinate(float(target[frame_index]))), 4, (165, 217, 102), -1, cv2.LINE_AA)
    cv2.circle(panel, (x_value, y_coordinate(float(score[frame_index]))), 4, (90, 173, 255), -1, cv2.LINE_AA)
    live_text = (
        f"frame {frame_index}/{len(score) - 1}  target {target[frame_index]:.0f}  score {score[frame_index]:.3f}"
    )
    text_width = cv2.getTextSize(live_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
    cv2.putText(
        panel,
        live_text,
        (max(12, panel.shape[1] - text_width - 12), 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (238, 242, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _export_episode_mp4(
    episode: dict[str, Any],
    output_path: Path,
    *,
    checkpoint_step: int,
    threshold: float,
    ffmpeg_bin: str,
) -> None:
    import cv2

    source_path = Path(str(episode["source_video"]))
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {source_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid source video dimensions: {source_path}")
    if not math.isfinite(source_fps) or source_fps <= 0:
        source_fps = float(episode["fps"])

    output_width = width if width % 2 == 0 else width + 1
    plot_height = max(230, int(math.ceil(height * 0.38 / 2.0) * 2))
    output_height = height + plot_height
    if output_height % 2:
        output_height += 1
        plot_height += 1

    base, geometry = _make_mp4_plot_base(
        episode,
        width=output_width,
        height=plot_height,
        checkpoint_step=checkpoint_step,
        threshold=threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    local_fd, local_name = tempfile.mkstemp(prefix="completion-visual-", suffix=".mp4")
    os.close(local_fd)
    local_output = Path(local_name)
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{output_width}x{output_height}",
        "-r",
        f"{source_fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(local_output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    written_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if output_width != width:
                frame = cv2.copyMakeBorder(frame, 0, 0, 0, output_width - width, cv2.BORDER_CONSTANT)
            series_frame = min(
                int(episode["frame_count"]) - 1,
                int(round(written_frames * float(episode["fps"]) / source_fps)),
            )
            panel = _draw_mp4_plot_frame(base, geometry, episode, series_frame)
            combined = np.vstack([frame, panel])
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin was not created")
            process.stdin.write(combined.tobytes())
            written_frames += 1
        capture.release()
        if process.stdin is not None:
            process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except Exception:
        capture.release()
        process.kill()
        process.wait()
        local_output.unlink(missing_ok=True)
        raise

    if written_frames == 0:
        local_output.unlink(missing_ok=True)
        raise RuntimeError(f"Source video contains no frames: {source_path}")
    if return_code != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        local_output.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {source_path}: {message}")
    if not local_output.is_file() or local_output.stat().st_size == 0:
        local_output.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg produced no MP4 for {source_path}")

    # MP4 finalization seeks backwards to write trailer/moov metadata, which
    # OSS/FUSE mounts do not reliably support. Encode on local disk, then copy
    # the already closed file sequentially to the mounted output directory.
    mounted_tmp = output_path.with_suffix(".mp4.tmp")
    mounted_tmp.unlink(missing_ok=True)
    try:
        shutil.copyfile(local_output, mounted_tmp)
        os.replace(mounted_tmp, output_path)
    finally:
        local_output.unlink(missing_ok=True)
        mounted_tmp.unlink(missing_ok=True)


def _export_mp4_report(
    series: list[dict[str, Any]],
    output_dir: Path,
    *,
    checkpoint_step: int,
    threshold: float,
    ffmpeg: str,
) -> None:
    ffmpeg_bin = shutil.which(ffmpeg)
    if ffmpeg_bin is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {ffmpeg}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for position, episode in enumerate(series, start=1):
        output_path = output_dir / f"episode_{int(episode['episode_index']):06d}.mp4"
        _export_episode_mp4(
            episode,
            output_path,
            checkpoint_step=checkpoint_step,
            threshold=threshold,
            ffmpeg_bin=ffmpeg_bin,
        )
        LOGGER.info(
            "Checkpoint %s MP4: %d/%d (%s)",
            checkpoint_step,
            position,
            len(series),
            output_path.name,
        )


# --------------------------------------------------------------------------- #
#  Comparison CSV and orchestration                                            #
# --------------------------------------------------------------------------- #


def _comparison_row(step: int, metrics: dict[str, Any], *, split: str) -> dict[str, Any]:
    overall = metrics["overall"]
    row: dict[str, Any] = {"checkpoint_step": step, "split": split}
    for key in (
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
    ):
        row[key] = overall[key]
    for task_index, task_metrics in sorted(metrics["per_task"].items(), key=lambda item: int(item[0])):
        row[f"subtask_{task_index}_auc"] = task_metrics["auc"]
        row[f"subtask_{task_index}_best_f1"] = task_metrics["best_f1"]
    return row


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty checkpoint comparison")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_worker_command(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    prediction_file: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config-name",
        args.config_name,
        "--split",
        args.split,
        "--train-group-count",
        str(getattr(args, "train_group_count", 0)),
        "--train-group-seed",
        str(getattr(args, "train_group_seed", 42)),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--worker-checkpoint",
        str(checkpoint_dir),
        "--worker-output",
        str(prediction_file),
    ]


def _run_checkpoint_worker(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    prediction_file: Path,
) -> None:
    command = _checkpoint_worker_command(args, checkpoint_dir=checkpoint_dir, prediction_file=prediction_file)
    environment = os.environ.copy()
    environment["HF_LEROBOT_HOME"] = str(args.hf_lerobot_home.resolve())
    LOGGER.info("Starting isolated checkpoint evaluation: %s", checkpoint_dir.name)
    subprocess.run(command, check=True, env=environment)


# --------------------------------------------------------------------------- #
#  Evaluation-scope group selection (shared by worker + parent)                #
# --------------------------------------------------------------------------- #


class EvaluationScope(NamedTuple):
    """Which episodes a checkpoint evaluation covers, after optional group sampling."""

    split: str
    groups: tuple[Any, ...]
    episode_ids: tuple[int, ...]
    group_ids: tuple[int, ...]
    requested_group_count: int
    group_seed: int
    is_full_split: bool


def _select_evaluation_groups(
    groups: Sequence[Any],
    *,
    group_count: int,
    seed: int,
) -> tuple[Any, ...]:
    """Pure, deterministic selection of evaluation task groups.

    ``group_count == 0`` returns every group in manifest order (no sampling).
    Otherwise ``group_count`` groups are drawn without replacement using an
    independent ``random.Random(seed)`` (never the global RNG) and the result is
    sorted by ``group_id`` so the order is stable and independent of the sampling
    order. Requesting more groups than available is an error -- there is no
    silent truncation. Each selected group is kept whole (all four episodes).
    """

    if group_count < 0:
        raise ValueError(f"group_count must be non-negative, got {group_count}")
    if group_count == 0:
        return tuple(groups)
    available = len(groups)
    if group_count > available:
        raise ValueError(
            f"Requested {group_count} train groups but the split only has {available}; "
            "reduce --train-group-count or use 0 for the full split."
        )
    rng = random.Random(seed)
    sampled = rng.sample(list(groups), group_count)
    return tuple(sorted(sampled, key=lambda group: int(group.group_id)))


def _resolve_evaluation_scope(
    manifest: Any,
    split: str,
    *,
    train_group_count: int,
    train_group_seed: int,
) -> EvaluationScope:
    """Resolve the episodes/groups a run covers from the split manifest.

    Group sampling only applies to the train split; val/test always use the full
    split (``train_group_count`` is ignored there). The result is the single
    source of truth shared by the worker (which writes it into the npz) and the
    parent (which uses it for resume validation).
    """

    if split not in SPLIT_CHOICES:
        raise ValueError(f"unsupported split {split!r}; expected one of {SPLIT_CHOICES}")
    if split == "train":
        groups = _select_evaluation_groups(
            manifest.splits["train"],
            group_count=train_group_count,
            seed=train_group_seed,
        )
        requested_count = train_group_count
        group_seed = train_group_seed
        is_full_split = train_group_count == 0
    else:
        groups = tuple(manifest.splits[split])
        requested_count = 0
        group_seed = train_group_seed
        is_full_split = True
    episode_ids = tuple(int(eid) for group in groups for eid in group.episode_ids)
    group_ids = tuple(int(group.group_id) for group in groups)
    return EvaluationScope(
        split=split,
        groups=groups,
        episode_ids=episode_ids,
        group_ids=group_ids,
        requested_group_count=requested_count,
        group_seed=group_seed,
        is_full_split=is_full_split,
    )


def _evaluation_scope_to_summary(scope: EvaluationScope) -> dict[str, Any]:
    """Serialize an ``EvaluationScope`` to the JSON-friendly ``evaluation_scope`` block."""

    return {
        "split": scope.split,
        "requested_group_count": scope.requested_group_count,
        "selected_group_count": len(scope.group_ids),
        "selected_episode_count": len(scope.episode_ids),
        "group_sample_seed": scope.group_seed,
        "selected_group_ids": list(scope.group_ids),
        "selected_episode_ids": list(scope.episode_ids),
        "is_full_split": scope.is_full_split,
    }


def _load_split_manifest(config_name: str) -> Any:
    """Load the leak-free split manifest named by a training config (lazy import)."""

    from openpi.training import completion_data as _completion_data
    from openpi.training import config as training_config

    config = training_config.get_config(config_name)
    manifest_path = config.completion.split_manifest_path
    return _completion_data.SplitManifest.from_dict(json.loads(Path(manifest_path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
#  Resume / report-episode selection                                           #
# --------------------------------------------------------------------------- #


def _read_prediction_split(prediction_file: Path) -> str | None:
    """Returns the split a prediction file was generated for, or ``None``.

    ``None`` means the file predates the ``--split`` feature (old test-only
    evaluator); such files carry no ``is_train_sample`` mask and cannot resume a
    train run.
    """

    with np.load(prediction_file, allow_pickle=False) as values:
        if "split" in values.files:
            return str(values["split"])
    return None


def _read_prediction_scope_arrays(prediction_file: Path) -> dict[str, Any] | None:
    """Read group-sampling scope arrays from a prediction file.

    Returns ``None`` for files predating the group-sampling feature (no
    ``selected_group_ids`` array). Files written by this version always carry
    ``selected_group_ids`` / ``selected_episode_ids`` /
    ``requested_train_group_count`` / ``train_group_seed`` together.
    """

    with np.load(prediction_file, allow_pickle=False) as values:
        if "selected_group_ids" not in values.files:
            return None
        return {
            "group_ids": tuple(int(value) for value in values["selected_group_ids"]),
            "episode_ids": tuple(int(value) for value in values["selected_episode_ids"]),
            "requested_count": int(values["requested_train_group_count"]),
            "seed": int(values["train_group_seed"]),
        }


def _build_evaluation_scope_summary(
    prediction_file: Path,
    *,
    split: str,
    train_group_seed: int,
) -> dict[str, Any]:
    """Build the ``evaluation_scope`` block for summaries/HTML from the npz.

    The npz is the source of truth for what was actually inferred. Legacy files
    (no scope arrays) produce a minimal full-split scope.
    """

    scope = _read_prediction_scope_arrays(prediction_file)
    if scope is None:
        return {
            "split": split,
            "requested_group_count": 0,
            "selected_group_count": None,
            "selected_episode_count": None,
            "group_sample_seed": train_group_seed,
            "selected_group_ids": [],
            "selected_episode_ids": [],
            "is_full_split": True,
        }
    return {
        "split": split,
        "requested_group_count": int(scope["requested_count"]),
        "selected_group_count": len(scope["group_ids"]),
        "selected_episode_count": len(scope["episode_ids"]),
        "group_sample_seed": int(scope["seed"]),
        "selected_group_ids": list(scope["group_ids"]),
        "selected_episode_ids": list(scope["episode_ids"]),
        "is_full_split": int(scope["requested_count"]) == 0,
    }


def _assert_prediction_scope_matches(
    prediction_file: Path,
    *,
    expected_split: str,
    train_group_count: int,
    train_group_seed: int,
    expected_scope: EvaluationScope | None,
) -> None:
    """Refuse to resume if the npz's evaluation scope does not match this run.

    The split field and the group-sampling scope (selected group/episode IDs,
    requested count, seed) must all agree. Files that predate the scope arrays
    can only resume full-split (count == 0) runs; a group-sampled run must
    re-run without ``--resume``.
    """

    stored_split = _read_prediction_split(prediction_file)
    scope = _read_prediction_scope_arrays(prediction_file)
    if stored_split is None:
        # Truly legacy file (predates --split). Only full test runs may resume.
        if train_group_count > 0:
            raise ValueError(
                f"Cannot resume group-sampled train run from {prediction_file}: it predates "
                "the split/scope fields. Re-run without --resume."
            )
        if expected_split != "test":
            raise ValueError(
                f"Cannot resume {expected_split}-split evaluation from {prediction_file}: it predates the split field."
            )
        LOGGER.warning(
            "Resuming test evaluation from a legacy prediction file (no split field): %s",
            prediction_file,
        )
        return
    if stored_split != expected_split:
        raise ValueError(
            f"Prediction file {prediction_file} was generated for split {stored_split!r}, "
            f"cannot resume as {expected_split!r}."
        )
    if train_group_count > 0:
        if scope is None:
            raise ValueError(
                f"Cannot resume group-sampled train run from {prediction_file}: it lacks "
                "group-sampling scope metadata. Re-run without --resume."
            )
        if expected_scope is None:
            raise ValueError("expected_scope must be provided when train_group_count > 0")
        if scope["requested_count"] != train_group_count:
            raise ValueError(
                f"Cannot resume: npz used train_group_count={scope['requested_count']}, requested {train_group_count}."
            )
        if scope["seed"] != train_group_seed:
            raise ValueError(f"Cannot resume: npz used train_group_seed={scope['seed']}, requested {train_group_seed}.")
        if list(scope["group_ids"]) != list(expected_scope.group_ids):
            raise ValueError(
                f"Cannot resume: npz selected group IDs {list(scope['group_ids'])} differ "
                f"from the expected {list(expected_scope.group_ids)}."
            )
        if list(scope["episode_ids"]) != list(expected_scope.episode_ids):
            raise ValueError("Cannot resume: npz selected episode IDs differ from the expected set.")
    elif scope is not None and scope["requested_count"] != 0:
        # The npz was a group-sampled run; refuse to silently resume as a full split.
        raise ValueError(
            f"Cannot resume full train split from {prediction_file}: it was a group-sampled "
            f"run (count={scope['requested_count']}, seed={scope['seed']}). Re-run without --resume."
        )


def _auto_select_episodes(series: list[dict[str, Any]], max_episodes: int) -> list[dict[str, Any]]:
    """Deterministic, task-balanced subselection including best/median/worst BCE.

    Reserves one slot each for the global lowest (best), median, and highest
    (worst) episode BCE, then fills the remaining slots round-robin across the
    four tasks (sorted task index), taking each task's episodes in ascending
    BCE order and skipping any already selected. The result is sorted by
    episode index for stable display.
    """

    by_episode = {ep["episode_index"]: ep for ep in series}
    ordered_by_bce = sorted(series, key=lambda ep: (float(ep["metrics"]["bce"]), ep["episode_index"]))
    selected_ids: set[int] = set()
    ordered: list[int] = []

    def take(ep: dict[str, Any]) -> None:
        if ep["episode_index"] not in selected_ids:
            selected_ids.add(ep["episode_index"])
            ordered.append(ep["episode_index"])

    if max_episodes >= 1 and ordered_by_bce:
        take(ordered_by_bce[0])  # best (lowest BCE)
    if max_episodes >= 2 and len(ordered_by_bce) > 1:
        take(ordered_by_bce[-1])  # worst (highest BCE)
    if max_episodes >= 3 and len(ordered_by_bce) > 2:
        take(ordered_by_bce[len(ordered_by_bce) // 2])  # median

    by_task: dict[int, list[dict[str, Any]]] = {}
    for ep in series:
        by_task.setdefault(int(ep["task_index"]), []).append(ep)
    task_queues = {
        task_index: iter(sorted(eps, key=lambda ep: (float(ep["metrics"]["bce"]), ep["episode_index"])))
        for task_index, eps in by_task.items()
    }

    remaining = max_episodes - len(ordered)
    while remaining > 0:
        progressed = False
        for task_index in sorted(task_queues):
            if remaining <= 0:
                break
            candidate = next(task_queues[task_index], None)
            if candidate is not None and candidate["episode_index"] not in selected_ids:
                take(candidate)
                remaining -= 1
                progressed = True
        if not progressed:
            break

    return [by_episode[eid] for eid in sorted(ordered)]


def _select_report_episodes(
    series: list[dict[str, Any]],
    *,
    max_episodes: int,
    episode_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Choose which episodes to visualize (HTML/MP4 only; never affects metrics)."""

    if episode_ids:
        wanted = {int(value) for value in episode_ids}
        selected = [ep for ep in series if ep["episode_index"] in wanted]
        missing = sorted(wanted - {ep["episode_index"] for ep in selected})
        if missing:
            raise ValueError(f"--report-episode-ids not found in this split: {missing}")
        return selected
    if max_episodes <= 0 or len(series) <= max_episodes:
        return series
    return _auto_select_episodes(series, max_episodes)


def _run_evaluation(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.resolve()
    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"Dataset metadata is incomplete: {dataset_root}")
    dataset_info = _read_json(info_path)
    fps = float(dataset_info["fps"])
    if fps <= 0:
        raise ValueError(f"Dataset fps must be positive, got {fps}")

    checkpoint_root = (args.checkpoint_base / args.config_name / args.exp_name).resolve()
    steps = _resolve_checkpoint_steps(checkpoint_root, args.checkpoints)
    if args.output_dir is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        output_dir = (args.evaluation_base / args.exp_name / args.split / timestamp).resolve()
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    _setup_logging(output_dir)
    started_at = datetime.now(UTC)
    LOGGER.info("Evaluation output: %s", output_dir)
    LOGGER.info("Split: %s", args.split)
    LOGGER.info("Dataset: %s", dataset_root)
    LOGGER.info("Checkpoint steps: %s", steps)

    metrics_scope = (
        "selected_train_groups" if args.split == "train" and args.train_group_count > 0 else f"full_{args.split}_split"
    )
    # For group-sampled train runs, resolve the expected scope once (this also
    # validates that the requested count does not exceed the available train
    # groups) so resume can compare it against each npz before reusing it.
    expected_scope: EvaluationScope | None = None
    if args.split == "train" and args.train_group_count > 0:
        expected_scope = _resolve_evaluation_scope(
            _load_split_manifest(args.config_name),
            args.split,
            train_group_count=args.train_group_count,
            train_group_seed=args.train_group_seed,
        )
        LOGGER.info(
            "Expected train scope: %d groups, %d episodes (seed=%d)",
            len(expected_scope.group_ids),
            len(expected_scope.episode_ids),
            args.train_group_seed,
        )

    comparison_rows: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    run_evaluation_scope: dict[str, Any] | None = None
    for step in steps:
        checkpoint_dir = checkpoint_root / str(step)
        checkpoint_output = output_dir / f"checkpoint_{step}"
        checkpoint_output.mkdir(exist_ok=args.resume)
        prediction_file = checkpoint_output / "predictions.npz"
        if args.resume and prediction_file.is_file():
            _assert_prediction_scope_matches(
                prediction_file,
                expected_split=args.split,
                train_group_count=args.train_group_count,
                train_group_seed=args.train_group_seed,
                expected_scope=expected_scope,
            )
            LOGGER.info("Reusing completed checkpoint predictions: %s", prediction_file)
        else:
            _run_checkpoint_worker(
                args,
                checkpoint_dir=checkpoint_dir,
                prediction_file=prediction_file,
            )

        metrics = compute_metrics(prediction_file, fps=fps, threshold=args.threshold)
        evaluation_scope = _build_evaluation_scope_summary(
            prediction_file,
            split=args.split,
            train_group_seed=args.train_group_seed,
        )
        if run_evaluation_scope is None:
            run_evaluation_scope = evaluation_scope
        checkpoint_summary: dict[str, Any] = {
            "checkpoint_step": step,
            "checkpoint_dir": str(checkpoint_dir),
            "dataset_root": str(dataset_root),
            "split": args.split,
            "threshold": args.threshold,
            "overall": metrics["overall"],
            "per_task": metrics["per_task"],
            "evaluation_scope": evaluation_scope,
            "metrics_scope": metrics_scope,
        }
        if args.split == "train":
            # Two train-fit metric sets from the same predictions: all train
            # frames vs only the frames the BoundaryCompletionSampler trained on.
            train_fit = _compute_train_fit_metrics(prediction_file)
            checkpoint_summary["metrics_all_frames"] = train_fit["metrics_all_frames"]
            checkpoint_summary["metrics_train_sampled"] = train_fit["metrics_train_sampled"]
        _write_json(checkpoint_output / "summary.json", checkpoint_summary)
        _write_jsonl(checkpoint_output / "episodes.jsonl", metrics["episodes"])

        report_dir = checkpoint_output / "report"
        full_series, _ = _load_report_series(
            dataset_root,
            prediction_file,
            metrics["episodes"],
            output_dir=report_dir,
            copy_videos=args.copy_videos,
        )
        # Visualize a capped, task-balanced subset for the train split (which has
        # hundreds of episodes); this never trims inference, metrics, or npz.
        report_series = _select_report_episodes(
            full_series,
            max_episodes=args.report_max_episodes,
            episode_ids=args.report_episode_ids,
        )
        manifest = {
            "checkpoint_step": step,
            "split": args.split,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset_root": str(dataset_root),
            "threshold": args.threshold,
            "top_camera_key": TOP_VIDEO_KEY,
            "evaluation_scope": evaluation_scope,
            "metrics_scope": metrics_scope,
            "episodes": report_series,
        }
        _write_html_report(report_dir, manifest)
        if args.export_mp4:
            _export_mp4_report(
                report_series,
                report_dir / "mp4",
                checkpoint_step=step,
                threshold=args.threshold,
                ffmpeg=args.ffmpeg,
            )

        comparison_rows.append(_comparison_row(step, metrics, split=args.split))
        checkpoint_summaries.append(
            {
                **checkpoint_summary,
                "checkpoint_output": str(checkpoint_output),
                "html_report": str(report_dir / "index.html"),
                "mp4_directory": str(report_dir / "mp4") if args.export_mp4 else None,
                "reported_episode_count": len(report_series),
                "total_episode_count": len(full_series),
            }
        )
        LOGGER.info(
            "Completed checkpoint %s (%s-split): AUC=%.4f best_f1=%.4f report=%s",
            step,
            args.split,
            metrics["overall"]["auc"],
            metrics["overall"]["best_f1"],
            report_dir / "index.html",
        )

    _write_comparison_csv(output_dir / "comparison.csv", comparison_rows)
    finished_at = datetime.now(UTC)
    run_summary = {
        "result": "success",
        "config_name": args.config_name,
        "experiment_name": args.exp_name,
        "split": args.split,
        "dataset_root": str(dataset_root),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_steps": steps,
        "settings": {
            "batch_size": args.batch_size,
            "noise_seed": args.seed,
            "threshold": args.threshold,
            "copy_videos": args.copy_videos,
            "export_mp4": args.export_mp4,
            "report_max_episodes": args.report_max_episodes,
            "report_episode_ids": list(args.report_episode_ids),
            "train_group_count": args.train_group_count,
            "train_group_seed": args.train_group_seed,
        },
        "evaluation_scope": run_evaluation_scope,
        "metrics_scope": metrics_scope,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": (finished_at - started_at).total_seconds(),
        "checkpoints": checkpoint_summaries,
    }
    _write_json(output_dir / "summary.json", run_summary)
    LOGGER.info("All checkpoint reports completed: %s", output_dir)
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--checkpoint-base", type=Path, default=DEFAULT_CHECKPOINT_BASE)
    parser.add_argument("--evaluation-base", type=Path, default=DEFAULT_EVALUATION_BASE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--hf-lerobot-home", type=Path, default=DEFAULT_HF_LEROBOT_HOME)
    parser.add_argument(
        "--split",
        choices=SPLIT_CHOICES,
        default=DEFAULT_SPLIT,
        help="Manifest split to evaluate (default: test). train runs every train frame and "
        "adds the metrics_all_frames / metrics_train_sampled train-fit metric sets.",
    )
    parser.add_argument(
        "--train-group-count",
        type=int,
        default=0,
        help="Sample this many whole TaskGroups (4 episodes each) directly from "
        "manifest.splits['train'] before any video decode / model forward, so inference, "
        "metrics, npz, and HTML cover exactly the selected groups. 0 = the full split (no "
        "sampling). Requires --split train. Unlike --report-max-episodes (which only caps "
        "HTML/MP4 display and never trims inference), this reduces the data actually scored.",
    )
    parser.add_argument(
        "--train-group-seed",
        type=int,
        default=42,
        help="Independent RNG seed for --train-group-count sampling (decoupled from --seed).",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=list(DEFAULT_CHECKPOINT_STEPS),
        help="Numeric checkpoint steps and/or 'latest'. Each checkpoint receives a complete independent report.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse predictions.npz in an existing output directory and continue report/MP4 generation.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Parallel LeRobot video decoding workers used by checkpoint inference.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for detection metrics. The best-threshold F1 is always reported separately.",
    )
    parser.add_argument(
        "--copy-videos",
        dest="copy_videos",
        action="store_true",
        help="Copy dataset videos into each HTML report (overrides the train-split default).",
    )
    parser.add_argument(
        "--no-copy-videos",
        dest="copy_videos",
        action="store_false",
        help="Reference dataset videos by absolute file URI instead of copying them into each HTML report.",
    )
    parser.add_argument(
        "--export-mp4",
        action="store_true",
        help="Also export one top-camera-plus-score MP4 for every reported episode and checkpoint.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable used by --export-mp4.")
    parser.add_argument(
        "--report-max-episodes",
        type=int,
        default=0,
        help="Cap the number of episodes visualized in the HTML/MP4 report (0 = all). "
        "Never affects inference, metrics, or predictions.npz.",
    )
    parser.add_argument(
        "--report-episode-ids",
        type=int,
        nargs="+",
        default=None,
        help="Visualize exactly these episode indices (overrides --report-max-episodes).",
    )
    parser.add_argument("--worker-checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    # copy_videos defaults to None so the split can pick a sensible default: the
    # train split has hundreds of episodes, so copying every video is wasteful
    # unless explicitly requested.
    parser.set_defaults(copy_videos=None)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be in (0, 1]")
    if not args.checkpoints:
        parser.error("--checkpoints cannot be empty")
    if args.report_max_episodes < 0:
        parser.error("--report-max-episodes must be non-negative")
    if args.train_group_count < 0:
        parser.error("--train-group-count must be non-negative")
    if args.train_group_count > 0 and args.split != "train":
        parser.error("--train-group-count requires --split train")
    if (args.worker_checkpoint is None) != (args.worker_output is None):
        parser.error("--worker-checkpoint and --worker-output must be provided together")
    if args.copy_videos is None:
        args.copy_videos = args.split != "train"
    if args.report_episode_ids is None:
        args.report_episode_ids = ()
    return args


def main() -> int:
    args = _parse_args()
    os.environ["HF_LEROBOT_HOME"] = str(args.hf_lerobot_home.resolve())
    _setup_logging()
    try:
        if args.worker_checkpoint is not None:
            return _evaluate_checkpoint_worker(args)
        _run_evaluation(args)
    except KeyboardInterrupt:
        LOGGER.error("Evaluation interrupted by user")
        return 130
    except Exception:
        LOGGER.exception("Completion head evaluation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
