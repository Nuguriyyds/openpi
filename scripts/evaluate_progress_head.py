"""Offline regression evaluation for frozen-prefix subtask progress heads.

The evaluator runs ``compute_completion_logits`` on every frame of the
persisted test split, applies sigmoid outside the head, and writes frame-level
predictions, per-episode metrics, aggregate metrics, and one progress curve
per episode. It deliberately has no binary threshold, F1, or AUC checks.
"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

LOGGER = logging.getLogger("progress_head_evaluation")
DEFAULT_CONFIG_NAME = "pi05_agilex_breakfast_frozen_head_s2_progress_head"
DEFAULT_EXP_NAME = "s2_progress_head"
DEFAULT_HF_LEROBOT_HOME = Path("/mnt/data/models/wyt/data")
DEFAULT_CHECKPOINT_BASE = Path("/mnt/data/models/wyt/checkpoints")
DEFAULT_EVALUATION_BASE = Path("/mnt/data/models/wyt/evaluations")
DEFAULT_DATASET_ROOT = Path("/mnt/data/models/wyt/data/agilex_make_breakfast_subtask_730_frozen_head_progress")
DEFAULT_CHECKPOINT_STEPS = ("200", "1000", "latest")


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
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _resolve_checkpoint_steps(checkpoint_root: Path, requested: list[str]) -> list[int]:
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"checkpoint experiment directory not found: {checkpoint_root}")
    available = sorted(int(path.name) for path in checkpoint_root.iterdir() if path.is_dir() and path.name.isdigit())
    if not available:
        raise FileNotFoundError(f"no numeric checkpoint directories found in: {checkpoint_root}")
    resolved: list[int] = []
    for requested_step in requested:
        step = available[-1] if requested_step.lower() == "latest" else int(requested_step)
        if step not in available:
            raise FileNotFoundError(f"checkpoint step {step} is unavailable; found {available}")
        if step not in resolved:
            resolved.append(step)
    return resolved


def _scalar(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    predictions = np.empty_like(logits)
    positive = logits >= 0
    predictions[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    predictions[~positive] = exp_logits / (1.0 + exp_logits)
    return predictions


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    sorter = np.argsort(values, kind="mergesort")
    inverse = np.empty(sorter.size, dtype=np.int64)
    inverse[sorter] = np.arange(sorter.size, dtype=np.int64)
    sorted_values = values[sorter]
    group_starts = np.concatenate(([True], sorted_values[1:] != sorted_values[:-1]))
    sorted_group_ids = np.cumsum(group_starts) - 1
    counts = np.bincount(sorted_group_ids)
    rank_sums = np.zeros(counts.size, dtype=np.float64)
    np.add.at(rank_sums, sorted_group_ids, np.arange(1, sorter.size + 1, dtype=np.float64))
    return (rank_sums / np.maximum(counts, 1))[sorted_group_ids[inverse]]


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(left) < 2:
        return 0.0
    centered_left = left - np.mean(left)
    centered_right = right - np.mean(right)
    denominator = float(np.sqrt(np.sum(np.square(centered_left)) * np.sum(np.square(centered_right))))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.sum(centered_left * centered_right) / denominator)


def progress_metrics(predictions: np.ndarray, targets: np.ndarray, *, huber_delta: float) -> dict[str, float]:
    """Regression metrics for bounded progress predictions and targets."""

    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predictions.shape != targets.shape or not len(predictions):
        raise ValueError(f"invalid progress arrays: predictions={predictions.shape}, targets={targets.shape}")
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(targets)):
        raise ValueError("progress predictions and targets must be finite")
    if np.any((predictions < 0.0) | (predictions > 1.0)):
        raise ValueError("progress predictions must lie in [0, 1]")
    if np.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("progress targets must lie in [0, 1]")
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")

    absolute_error = np.abs(predictions - targets)
    huber = np.where(
        absolute_error <= huber_delta,
        0.5 * np.square(absolute_error),
        huber_delta * (absolute_error - 0.5 * huber_delta),
    )
    early = targets <= 0.1
    late = targets >= 0.9
    return {
        "frame_count": len(targets),
        "loss": float(np.mean(huber)),
        "mae": float(np.mean(absolute_error)),
        "rmse": float(np.sqrt(np.mean(np.square(predictions - targets)))),
        "pearson": _safe_correlation(predictions, targets),
        "spearman": _safe_correlation(_rankdata(predictions), _rankdata(targets)),
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": float(np.std(predictions)),
        "prediction_min": float(np.min(predictions)),
        "prediction_max": float(np.max(predictions)),
        "target_mean": float(np.mean(targets)),
        "target_std": float(np.std(targets)),
        "target_min": float(np.min(targets)),
        "target_max": float(np.max(targets)),
        "early_mae": float(np.mean(absolute_error[early])) if np.any(early) else 0.0,
        "late_mae": float(np.mean(absolute_error[late])) if np.any(late) else 0.0,
        "early_frame_count": int(np.sum(early)),
        "late_frame_count": int(np.sum(late)),
    }


def compute_metrics(prediction_file: Path, *, huber_delta: float) -> dict[str, Any]:
    """Validates and summarizes frame-level predictions without binary labels."""

    with np.load(prediction_file, allow_pickle=False) as values:
        episode_indices = np.asarray(values["episode_index"])
        task_indices = np.asarray(values["task_index"])
        frame_indices = np.asarray(values["frame_index"])
        logits = np.asarray(values["logit"])
        targets = np.asarray(values["target"])
        infer_ms = np.asarray(values["infer_ms"])
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
        raise ValueError(f"prediction arrays have invalid row counts: {row_counts}")
    for name in ("logit", "target", "infer_ms"):
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"{name} contains non-finite values")
    if np.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("progress targets must lie in [0, 1]")
    predictions = _sigmoid(logits)

    episode_rows: list[dict[str, Any]] = []
    for raw_episode_index in sorted(np.unique(episode_indices).tolist()):
        episode_index = int(raw_episode_index)
        episode_mask = episode_indices == episode_index
        episode_task_indices = np.unique(task_indices[episode_mask])
        if len(episode_task_indices) != 1:
            raise ValueError(
                f"episode {episode_index}: expected exactly one task_index, got {episode_task_indices.tolist()}"
            )
        order = np.argsort(frame_indices[episode_mask])
        ordered_frames = frame_indices[episode_mask][order]
        expected_frames = np.arange(len(ordered_frames), dtype=ordered_frames.dtype)
        if not np.array_equal(ordered_frames, expected_frames):
            raise ValueError(f"episode {episode_index}: frame_index must be contiguous from zero")
        episode_rows.append(
            {
                "episode_index": episode_index,
                "task_index": int(episode_task_indices[0]),
                **progress_metrics(
                    predictions[episode_mask][order], targets[episode_mask][order], huber_delta=huber_delta
                ),
                "mean_infer_ms_per_frame": float(np.mean(infer_ms[episode_mask][order])),
            }
        )

    per_task: dict[str, dict[str, float]] = {}
    for raw_task_index in sorted(np.unique(task_indices).tolist()):
        task_index = int(raw_task_index)
        task_mask = task_indices == task_index
        per_task[str(task_index)] = progress_metrics(
            predictions[task_mask], targets[task_mask], huber_delta=huber_delta
        )
    return {
        "overall": {
            **progress_metrics(predictions, targets, huber_delta=huber_delta),
            "mean_infer_ms_per_frame": float(np.mean(infer_ms)),
        },
        "per_task": per_task,
        "episodes": episode_rows,
    }


def _evaluation_repack():
    import openpi.transforms as transforms  # noqa: PLC0415

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
                    "actions": "actions",
                    "prompt": "prompt",
                }
            )
        ]
    )


def _evaluate_checkpoint_worker(args: argparse.Namespace) -> int:
    """Loads one checkpoint in a fresh process and predicts every test frame."""

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: PLC0415

    import openpi.models.model as model_api  # noqa: PLC0415
    from openpi.policies import policy_config  # noqa: PLC0415
    import openpi.shared.nnx_utils as nnx_utils  # noqa: PLC0415
    from openpi.training import completion_data  # noqa: PLC0415
    from openpi.training import config as training_config  # noqa: PLC0415

    checkpoint_dir = Path(args.worker_checkpoint).resolve()
    output_path = Path(args.worker_output).resolve()
    if not (checkpoint_dir / "params").is_dir():
        raise FileNotFoundError(f"checkpoint params not found: {checkpoint_dir / 'params'}")
    config = training_config.get_config(args.config_name)
    if not config.completion.uses_progress_objective:
        raise ValueError(f"config {args.config_name!r} is not a progress-head training config")
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("training config has no LeRobot repo_id")

    policy = policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        repack_transforms=_evaluation_repack(),
        sample_kwargs={},
    )
    model = policy._model  # noqa: SLF001
    model.eval()
    compute_fn = nnx_utils.module_jit(model.compute_completion_logits, static_argnames="train")

    manifest_path = config.completion.split_manifest_path
    if manifest_path is None:
        raise ValueError("progress evaluation requires completion.split_manifest_path")
    manifest = completion_data.SplitManifest.from_dict(json.loads(Path(manifest_path).read_text(encoding="utf-8")))
    test_episode_ids = manifest.episode_ids("test")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id)
    tasks = dataset_meta.tasks
    episode_from = dataset.episode_data_index["from"]
    episode_to = dataset.episode_data_index["to"]
    frame_specs = [
        (episode_index, local_frame, int(episode_from[episode_index]) + local_frame)
        for episode_index in test_episode_ids
        for local_frame in range(int(episode_to[episode_index]) - int(episode_from[episode_index]))
    ]
    if not frame_specs:
        raise ValueError("test split contains no frames")
    LOGGER.info("Evaluating %d test frames from %d episodes", len(frame_specs), len(test_episode_ids))

    result_episode_indices: list[int] = []
    result_task_indices: list[int] = []
    result_frame_indices: list[int] = []
    result_logits: list[float] = []
    result_targets: list[float] = []
    result_infer_ms: list[float] = []
    rng = jax.random.key(args.seed)
    label_key = config.completion.label_key
    for batch_start in range(0, len(frame_specs), args.batch_size):
        valid_specs = frame_specs[batch_start : batch_start + args.batch_size]
        transformed_items: list[dict[str, Any]] = []
        batch_metadata: list[tuple[int, int, int, float]] = []
        for episode_index, local_frame_index, dataset_index in valid_specs:
            sample = dict(dataset[dataset_index])
            task_index = _scalar(sample["task_index"])
            if task_index not in tasks:
                raise ValueError(f"task {task_index} is missing from dataset metadata")
            sample["prompt"] = tasks[task_index]
            target = float(np.asarray(sample[label_key]).reshape(-1)[0])
            transformed_items.append(policy._input_transform(sample))  # noqa: SLF001
            batch_metadata.append((episode_index, task_index, local_frame_index, target))
        valid_count = len(transformed_items)
        while len(transformed_items) < args.batch_size:
            transformed_items.append(jax.tree.map(lambda value: np.array(value, copy=True), transformed_items[-1]))
        batched_inputs = jax.tree.map(
            lambda *values: jnp.asarray(np.stack([np.asarray(value) for value in values], axis=0)),
            *transformed_items,
        )
        observation = model_api.Observation.from_dict(batched_inputs)
        started = time.monotonic()
        logits = np.asarray(jax.block_until_ready(compute_fn(rng, observation, train=False)))
        infer_ms = (time.monotonic() - started) * 1000.0 / valid_count
        for index, (episode_index, task_index, local_frame_index, target) in enumerate(batch_metadata):
            result_episode_indices.append(episode_index)
            result_task_indices.append(task_index)
            result_frame_indices.append(local_frame_index)
            result_logits.append(float(logits[index]))
            result_targets.append(target)
            result_infer_ms.append(infer_ms)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="progress-eval-") as temporary_dir:
        local_output = Path(temporary_dir) / "predictions.npz"
        np.savez_compressed(
            local_output,
            episode_index=np.asarray(result_episode_indices, dtype=np.int32),
            task_index=np.asarray(result_task_indices, dtype=np.int16),
            frame_index=np.asarray(result_frame_indices, dtype=np.int32),
            logit=np.asarray(result_logits, dtype=np.float32),
            target=np.asarray(result_targets, dtype=np.float32),
            infer_ms=np.asarray(result_infer_ms, dtype=np.float32),
        )
        with local_output.open("rb") as source, output_path.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    return 0


def _write_frame_csv(prediction_file: Path, output_path: Path) -> None:
    with np.load(prediction_file, allow_pickle=False) as values:
        episode_indices = np.asarray(values["episode_index"])
        task_indices = np.asarray(values["task_index"])
        frame_indices = np.asarray(values["frame_index"])
        logits = np.asarray(values["logit"])
        targets = np.asarray(values["target"])
        infer_ms = np.asarray(values["infer_ms"])
    predictions = _sigmoid(logits)
    order = np.lexsort((frame_indices, episode_indices))
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("episode_index", "task_index", "frame_index", "target", "prediction", "logit", "infer_ms"),
        )
        writer.writeheader()
        for index in order:
            writer.writerow(
                {
                    "episode_index": int(episode_indices[index]),
                    "task_index": int(task_indices[index]),
                    "frame_index": int(frame_indices[index]),
                    "target": float(targets[index]),
                    "prediction": float(predictions[index]),
                    "logit": float(logits[index]),
                    "infer_ms": float(infer_ms[index]),
                }
            )


def _write_progress_curves(prediction_file: Path, output_dir: Path) -> None:
    with np.load(prediction_file, allow_pickle=False) as values:
        episode_indices = np.asarray(values["episode_index"])
        task_indices = np.asarray(values["task_index"])
        frame_indices = np.asarray(values["frame_index"])
        targets = np.asarray(values["target"])
        predictions = _sigmoid(np.asarray(values["logit"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    width = 900
    height = 450
    left = 64
    right = width - 24
    top = 32
    bottom = height - 52

    def point_string(frames: np.ndarray, values: np.ndarray) -> str:
        last_frame = max(int(frames[-1]), 1)
        return " ".join(
            f"{left + (int(frame) / last_frame) * (right - left):.2f},{bottom - float(value) * (bottom - top):.2f}"
            for frame, value in zip(frames, values, strict=True)
        )

    for raw_episode_index in sorted(np.unique(episode_indices).tolist()):
        episode_index = int(raw_episode_index)
        mask = episode_indices == episode_index
        order = np.argsort(frame_indices[mask])
        task_index = int(np.unique(task_indices[mask])[0])
        episode_frames = frame_indices[mask][order]
        episode_targets = targets[mask][order]
        episode_predictions = predictions[mask][order]
        grid_lines = "".join(
            f'<line x1="{left}" y1="{bottom - value * (bottom - top):.2f}" '
            f'x2="{right}" y2="{bottom - value * (bottom - top):.2f}" stroke="#d9e1e8" />'
            f'<text x="{left - 10}" y="{bottom - value * (bottom - top) + 4:.2f}" '
            f'text-anchor="end" font-size="12">{value:.2f}</text>'
            for value in (0.0, 0.25, 0.5, 0.75, 1.0)
        )
        svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white" />
<text x="{left}" y="20" font-family="sans-serif" font-size="16">Episode {episode_index} | subtask {task_index}</text>
{grid_lines}
<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222" />
<polyline fill="none" stroke="#2a9d8f" stroke-width="2.5" points="{point_string(episode_frames, episode_targets)}" />
<polyline fill="none" stroke="#e76f51" stroke-width="2.5" points="{point_string(episode_frames, episode_predictions)}" />
<line x1="{right - 215}" y1="{top + 8}" x2="{right - 190}" y2="{top + 8}" stroke="#2a9d8f" stroke-width="3" />
<text x="{right - 185}" y="{top + 12}" font-family="sans-serif" font-size="12">target</text>
<line x1="{right - 115}" y1="{top + 8}" x2="{right - 90}" y2="{top + 8}" stroke="#e76f51" stroke-width="3" />
<text x="{right - 85}" y="{top + 12}" font-family="sans-serif" font-size="12">prediction</text>
<text x="{(left + right) / 2:.2f}" y="{height - 16}" text-anchor="middle" font-family="sans-serif" font-size="13">Local frame index</text>
<text x="18" y="{(top + bottom) / 2:.2f}" transform="rotate(-90 18 {(top + bottom) / 2:.2f})" text-anchor="middle" font-family="sans-serif" font-size="13">Progress</text>
</svg>
"""
        (output_dir / f"episode_{episode_index:06d}.svg").write_text(svg, encoding="utf-8")


def _run_checkpoint_worker(args: argparse.Namespace, *, checkpoint_dir: Path, prediction_file: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config-name",
        args.config_name,
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(args.seed),
        "--worker-checkpoint",
        str(checkpoint_dir),
        "--worker-output",
        str(prediction_file),
    ]
    environment = os.environ.copy()
    environment["HF_LEROBOT_HOME"] = str(args.hf_lerobot_home.resolve())
    subprocess.run(command, check=True, env=environment)


def _comparison_row(step: int, metrics: dict[str, Any]) -> dict[str, float | int]:
    overall = metrics["overall"]
    return {"checkpoint_step": step, **{key: overall[key] for key in overall}}


def _write_comparison_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_evaluation(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"dataset metadata is incomplete: {dataset_root}")
    checkpoint_root = (args.checkpoint_base / args.config_name / args.exp_name).resolve()
    steps = _resolve_checkpoint_steps(checkpoint_root, args.checkpoints)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = (
        (args.evaluation_base / args.exp_name / timestamp).resolve()
        if args.output_dir is None
        else args.output_dir.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    _setup_logging(output_dir)

    comparison_rows: list[dict[str, float | int]] = []
    summaries: list[dict[str, Any]] = []
    for step in steps:
        checkpoint_dir = checkpoint_root / str(step)
        checkpoint_output = output_dir / f"checkpoint_{step}"
        checkpoint_output.mkdir()
        prediction_file = checkpoint_output / "predictions.npz"
        _run_checkpoint_worker(args, checkpoint_dir=checkpoint_dir, prediction_file=prediction_file)
        metrics = compute_metrics(prediction_file, huber_delta=args.huber_delta)
        _write_frame_csv(prediction_file, checkpoint_output / "predictions.csv")
        _write_jsonl(checkpoint_output / "episodes.jsonl", metrics["episodes"])
        if not args.no_curves:
            _write_progress_curves(prediction_file, checkpoint_output / "curves")
        summary = {
            "checkpoint_step": step,
            "checkpoint_dir": str(checkpoint_dir),
            "dataset_root": str(dataset_root),
            "huber_delta": args.huber_delta,
            "overall": metrics["overall"],
            "per_task": metrics["per_task"],
            "episode_metrics": metrics["episodes"],
        }
        _write_json(checkpoint_output / "summary.json", summary)
        comparison_rows.append(_comparison_row(step, metrics))
        summaries.append(summary)
        LOGGER.info(
            "Checkpoint %s: MAE=%.6f RMSE=%.6f Pearson=%.6f Spearman=%.6f",
            step,
            metrics["overall"]["mae"],
            metrics["overall"]["rmse"],
            metrics["overall"]["pearson"],
            metrics["overall"]["spearman"],
        )
    _write_comparison_csv(output_dir / "comparison.csv", comparison_rows)
    _write_json(
        output_dir / "summary.json",
        {
            "result": "success",
            "config_name": args.config_name,
            "experiment_name": args.exp_name,
            "dataset_root": str(dataset_root),
            "checkpoint_steps": steps,
            "huber_delta": args.huber_delta,
            "checkpoints": summaries,
        },
    )
    return output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--checkpoint-base", type=Path, default=DEFAULT_CHECKPOINT_BASE)
    parser.add_argument("--evaluation-base", type=Path, default=DEFAULT_EVALUATION_BASE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--hf-lerobot-home", type=Path, default=DEFAULT_HF_LEROBOT_HOME)
    parser.add_argument("--checkpoints", nargs="+", default=list(DEFAULT_CHECKPOINT_STEPS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--huber-delta", type=float, default=0.1)
    parser.add_argument("--no-curves", action="store_true")
    parser.add_argument("--worker-checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.huber_delta <= 0:
        parser.error("--huber-delta must be positive")
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
        output_dir = _run_evaluation(args)
        LOGGER.info("evaluation completed: %s", output_dir)
    except KeyboardInterrupt:
        LOGGER.error("evaluation interrupted by user")
        return 130
    except Exception:
        LOGGER.exception("progress head evaluation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
