"""Offline evaluation and visualization for completion head checkpoints.

Evaluates one or more S2 completion-head checkpoints on the held-out **test**
split and writes complete, independent reports (HTML + JSON + JSONL + CSV).
Pass ``--export-mp4`` to additionally render one annotated MP4 per episode.

The completion head outputs a binary "is the episode about to end?" logit per
frame. Unlike the progress script (which predicts a continuous 0→1 curve),
here the ground-truth label is 0 everywhere except the last 2 frames (=1).
The visualisation therefore plots sigmoid scores and the 0/1 target, plus
per-episode detection metrics (precision, recall, F1 at the best threshold).
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any

import numpy as np


LOGGER = logging.getLogger("completion_head_evaluation")

DEFAULT_CONFIG_NAME = "pi05_agilex_breakfast_frozen_head_s2_completion_head"
DEFAULT_EXP_NAME = "s2_completion_head"
DEFAULT_HF_LEROBOT_HOME = Path("/mnt/data/models/wyt/data")
DEFAULT_CHECKPOINT_BASE = Path("/mnt/data/models/wyt/checkpoints")
DEFAULT_EVALUATION_BASE = Path("/mnt/data/models/wyt/evaluations")
DEFAULT_DATASET_ROOT = Path(
    "/mnt/data/models/wyt/data/agilex_make_breakfast_subtask_730_frozen_head"
)
DEFAULT_CHECKPOINT_STEPS = ("200", "1000", "latest")
TOP_VIDEO_KEY = "observation.image.top"
LABEL_KEY = "completion"


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
        (np.sum(ranks[positives]) - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
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
        "frame_count": int(len(targets)),
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
        float(row["detection_delay_seconds"])
        for row in rows
        if row["detection_delay_seconds"] is not None
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
            raise ValueError(
                f"Episode {episode_index}: expected one task index, got {episode_task_indices.tolist()}"
            )
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

    import openpi.models.model as model_api
    from openpi.policies import policy_config
    from openpi.training import config as training_config
    from openpi.training import completion_data as _completion_data
    import openpi.shared.nnx_utils as nnx_utils

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
    compute_fn = nnx_utils.module_jit(
        model.compute_completion_logits, static_argnames="train"
    )

    # Determine test episodes from the split manifest.
    manifest_path = config.completion.split_manifest_path
    manifest = _completion_data.SplitManifest.from_dict(
        json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    )
    test_episode_ids = manifest.episode_ids("test")
    LOGGER.info("Test episodes (%d): %s", len(test_episode_ids), test_episode_ids)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id)
    tasks = dataset_meta.tasks

    episode_from = dataset.episode_data_index["from"]
    episode_to = dataset.episode_data_index["to"]

    frame_specs: list[tuple[int, int, int]] = []  # (episode_index, local_frame, dataset_index)
    for episode_index in test_episode_ids:
        start = int(episode_from[episode_index])
        end = int(episode_to[episode_index])
        frame_specs.extend(
            (episode_index, local_frame, start + local_frame)
            for local_frame in range(end - start)
        )

    LOGGER.info(
        "Evaluating %d test episodes (%d frames), batch_size=%d",
        len(test_episode_ids),
        len(frame_specs),
        args.batch_size,
    )

    result_episode_indices: list[int] = []
    result_task_indices: list[int] = []
    result_frame_indices: list[int] = []
    result_logits: list[float] = []
    result_targets: list[float] = []
    result_infer_ms: list[float] = []

    rng = jax.random.key(args.seed)

    for batch_start in range(0, len(frame_specs), args.batch_size):
        valid_specs = frame_specs[batch_start : batch_start + args.batch_size]
        transformed_items: list[dict[str, Any]] = []
        batch_metadata: list[tuple[int, int, int, float]] = []

        for episode_index, local_frame_index, dataset_index in valid_specs:
            sample = dict(dataset[dataset_index])
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

        for batch_index, (episode_index, task_index, local_frame_index, target) in enumerate(
            batch_metadata
        ):
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
    with tempfile.TemporaryDirectory(prefix="completion-eval-") as tmp:
        local_output = Path(tmp) / "predictions.npz"
        np.savez_compressed(
            local_output,
            episode_index=np.asarray(result_episode_indices, dtype=np.int32),
            task_index=np.asarray(result_task_indices, dtype=np.int16),
            frame_index=np.asarray(result_frame_indices, dtype=np.int32),
            logit=np.asarray(result_logits, dtype=np.float32),
            target=np.asarray(result_targets, dtype=np.float32),
            infer_ms=np.asarray(result_infer_ms, dtype=np.float32),
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
    """Builds per-episode series for the HTML / MP4 reports."""

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

    videos_dir = output_dir / "videos"
    if copy_videos:
        videos_dir.mkdir(parents=True, exist_ok=True)

    series: list[dict[str, Any]] = []
    for episode_index in sorted(metrics_by_episode.keys()):
        mask = episode_indices == episode_index
        frame_count = int(np.sum(mask))
        if episode_index not in episode_rows:
            raise ValueError(f"Episode {episode_index}: missing from episodes.jsonl")
        expected_length = int(episode_rows[episode_index]["length"])
        if frame_count != expected_length:
            raise ValueError(
                f"Episode {episode_index}: prediction rows={frame_count}, dataset rows={expected_length}"
            )
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
            shutil.copy2(source_video, destination)
            video_path = destination.relative_to(output_dir).as_posix()
        else:
            video_path = source_video.as_uri()

        episode_logits = logits[mask][order]
        episode_scores = _sigmoid(episode_logits)
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
                "logit": episode_logits.astype(float).tolist(),
                "target": targets[mask][order].astype(float).tolist(),
                "metrics": metrics_by_episode[episode_index],
            }
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
            <p class="subtitle">Held-out test episodes. Top camera and raw sigmoid scores vs binary target.</p>
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
                <span style="--line:#b49cff">Best threshold</span>
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
              addMetric("F1 @ 0.5", formatNumber(m.f1_at_0.5, 4));
              addMetric("Precision @ 0.5", formatNumber(m.precision_at_0.5, 4));
              addMetric("Recall @ 0.5", formatNumber(m.recall_at_0.5, 4));
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
            background, f"{value:.2f}", (7, y_value + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (191, 175, 151), 1, cv2.LINE_AA,
        )

    threshold_y = y_coordinate(threshold)
    for start in range(left, right, 14):
        cv2.line(background, (start, threshold_y), (min(start + 7, right), threshold_y), (255, 156, 180), 1)
    target_points = _plot_points(
        target, left=left, right=right, top=top, bottom=bottom, minimum=minimum, maximum=maximum)
    score_points = _plot_points(
        score, left=left, right=right, top=top, bottom=bottom, minimum=minimum, maximum=maximum)
    cv2.polylines(background, [target_points], False, (165, 217, 102), 2, cv2.LINE_AA)
    cv2.polylines(background, [score_points], False, (90, 173, 255), 2, cv2.LINE_AA)

    final_frame = int(episode["frame_count"]) - 1
    for frame in (0, final_frame // 2, final_frame):
        x_value = int(round(left + frame / max(1, final_frame) * (right - left)))
        label = str(frame)
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
        cv2.putText(
            background, label, (x_value - text_width // 2, height - 13),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (191, 175, 151), 1, cv2.LINE_AA,
        )

    return background, {"left": left, "right": right, "top": top, "bottom": bottom,
                        "minimum": minimum, "maximum": maximum}


def _draw_mp4_plot_frame(
    base: np.ndarray, geometry: dict[str, float | int],
    episode: dict[str, Any], frame_index: int,
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
        f"frame {frame_index}/{len(score) - 1}  "
        f"target {target[frame_index]:.0f}  score {score[frame_index]:.3f}"
    )
    text_width = cv2.getTextSize(live_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
    cv2.putText(
        panel, live_text, (max(12, panel.shape[1] - text_width - 12), 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (238, 242, 255), 1, cv2.LINE_AA,
    )
    return panel


def _export_episode_mp4(
    episode: dict[str, Any], output_path: Path,
    *, checkpoint_step: int, threshold: float, ffmpeg_bin: str,
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
        episode, width=output_width, height=plot_height,
        checkpoint_step=checkpoint_step, threshold=threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{output_width}x{output_height}",
        "-r", f"{source_fps:.8f}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        str(output_path),
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
        raise

    if written_frames == 0:
        raise RuntimeError(f"Source video contains no frames: {source_path}")
    if return_code != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for {source_path}: {message}")


def _export_mp4_report(
    series: list[dict[str, Any]], output_dir: Path,
    *, checkpoint_step: int, threshold: float, ffmpeg: str,
) -> None:
    ffmpeg_bin = shutil.which(ffmpeg)
    if ffmpeg_bin is None:
        raise FileNotFoundError(f"ffmpeg executable not found: {ffmpeg}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for position, episode in enumerate(series, start=1):
        output_path = output_dir / f"episode_{int(episode['episode_index']):06d}.mp4"
        _export_episode_mp4(
            episode, output_path,
            checkpoint_step=checkpoint_step, threshold=threshold, ffmpeg_bin=ffmpeg_bin,
        )
        LOGGER.info(
            "Checkpoint %s MP4: %d/%d (%s)",
            checkpoint_step, position, len(series), output_path.name,
        )


# --------------------------------------------------------------------------- #
#  Comparison CSV and orchestration                                            #
# --------------------------------------------------------------------------- #

def _comparison_row(step: int, metrics: dict[str, Any]) -> dict[str, Any]:
    overall = metrics["overall"]
    row: dict[str, Any] = {"checkpoint_step": step}
    for key in (
        "frame_count", "positive_count", "negative_count",
        "auc", "best_f1", "best_threshold", "best_precision", "best_recall",
        "f1_at_0.5", "precision_at_0.5", "recall_at_0.5", "bce",
        "positive_score_mean", "negative_score_mean",
        "early_trigger_rate", "never_trigger_rate",
        "pre_threshold_false_positive_rate",
        "mean_detection_delay_frames", "median_detection_delay_frames",
        "mean_detection_delay_seconds", "median_detection_delay_seconds",
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


def _run_checkpoint_worker(
    args: argparse.Namespace, *, checkpoint_dir: Path, prediction_file: Path,
) -> None:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--config-name", args.config_name,
        "--batch-size", str(args.batch_size),
        "--seed", str(args.seed),
        "--worker-checkpoint", str(checkpoint_dir),
        "--worker-output", str(prediction_file),
    ]
    environment = os.environ.copy()
    environment["HF_LEROBOT_HOME"] = str(args.hf_lerobot_home.resolve())
    LOGGER.info("Starting isolated checkpoint evaluation: %s", checkpoint_dir.name)
    subprocess.run(command, check=True, env=environment)


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
        output_dir = (args.evaluation_base / args.exp_name / timestamp).resolve()
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    _setup_logging(output_dir)
    started_at = datetime.now(UTC)
    LOGGER.info("Evaluation output: %s", output_dir)
    LOGGER.info("Dataset: %s", dataset_root)
    LOGGER.info("Checkpoint steps: %s", steps)

    comparison_rows: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    for step in steps:
        checkpoint_dir = checkpoint_root / str(step)
        checkpoint_output = output_dir / f"checkpoint_{step}"
        checkpoint_output.mkdir()
        prediction_file = checkpoint_output / "predictions.npz"
        _run_checkpoint_worker(
            args, checkpoint_dir=checkpoint_dir, prediction_file=prediction_file,
        )

        metrics = compute_metrics(prediction_file, fps=fps, threshold=args.threshold)
        checkpoint_summary = {
            "checkpoint_step": step,
            "checkpoint_dir": str(checkpoint_dir),
            "dataset_root": str(dataset_root),
            "threshold": args.threshold,
            "overall": metrics["overall"],
            "per_task": metrics["per_task"],
        }
        _write_json(checkpoint_output / "summary.json", checkpoint_summary)
        _write_jsonl(checkpoint_output / "episodes.jsonl", metrics["episodes"])

        report_dir = checkpoint_output / "report"
        series, _ = _load_report_series(
            dataset_root, prediction_file, metrics["episodes"],
            output_dir=report_dir, copy_videos=args.copy_videos,
        )
        manifest = {
            "checkpoint_step": step,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset_root": str(dataset_root),
            "threshold": args.threshold,
            "top_camera_key": TOP_VIDEO_KEY,
            "episodes": series,
        }
        _write_html_report(report_dir, manifest)
        if args.export_mp4:
            _export_mp4_report(
                series, report_dir / "mp4",
                checkpoint_step=step, threshold=args.threshold, ffmpeg=args.ffmpeg,
            )

        comparison_rows.append(_comparison_row(step, metrics))
        checkpoint_summaries.append(
            {
                **checkpoint_summary,
                "checkpoint_output": str(checkpoint_output),
                "html_report": str(report_dir / "index.html"),
                "mp4_directory": str(report_dir / "mp4") if args.export_mp4 else None,
            }
        )
        LOGGER.info(
            "Completed checkpoint %s: AUC=%.4f best_f1=%.4f report=%s",
            step, metrics["overall"]["auc"], metrics["overall"]["best_f1"],
            report_dir / "index.html",
        )

    _write_comparison_csv(output_dir / "comparison.csv", comparison_rows)
    finished_at = datetime.now(UTC)
    run_summary = {
        "result": "success",
        "config_name": args.config_name,
        "experiment_name": args.exp_name,
        "dataset_root": str(dataset_root),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_steps": steps,
        "settings": {
            "batch_size": args.batch_size,
            "noise_seed": args.seed,
            "threshold": args.threshold,
            "copy_videos": args.copy_videos,
            "export_mp4": args.export_mp4,
        },
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
        "--checkpoints",
        nargs="+",
        default=list(DEFAULT_CHECKPOINT_STEPS),
        help="Numeric checkpoint steps and/or 'latest'. Each checkpoint receives a complete independent report.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Score threshold for detection metrics. The best-threshold F1 is always reported separately.",
    )
    parser.add_argument(
        "--no-copy-videos",
        dest="copy_videos", action="store_false",
        help="Reference dataset videos by absolute file URI instead of copying them into each HTML report.",
    )
    parser.add_argument(
        "--export-mp4", action="store_true",
        help="Also export one top-camera-plus-score MP4 for every test episode and checkpoint.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable used by --export-mp4.")
    parser.add_argument("--worker-checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.set_defaults(copy_videos=True)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be in (0, 1]")
    if not args.checkpoints:
        parser.error("--checkpoints cannot be empty")
    if (args.worker_checkpoint is None) != (args.worker_output is None):
        parser.error("--worker-checkpoint and --worker-output must be provided together")
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
