"""Offline evaluation for the boundary completion scheme.

Evaluates a **single predetermined checkpoint** on the test split and produces
``test_full`` (all frames, 30 fps, primary) and ``test_sparse`` (same 0.5 s
negative sampling as training, auxiliary) metric sets.  The deployment
threshold is fixed at 0.5; best-F1 / best-threshold are reported as
**diagnostics only** and never influence the threshold or checkpoint choice.

Only one checkpoint is evaluated — there is no multi-checkpoint iteration and
no ``latest`` alias.  The checkpoint step must be specified explicitly and must
match the step preregistered in ``eval_checkpoint.json`` (written during
training).  Bypassing preregistration is forbidden to prevent test-set model
selection.  When the managed step directory was deleted by ``max_to_keep=1``,
the protected ``eval_checkpoint/<step>/`` copy is used, but only if its
``_protected_step.json`` marker matches the requested step (P1-A).
"""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

# Import shared infrastructure from the existing completion-head evaluator.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from evaluate_completion_head import LOGGER  # noqa: E402
from evaluate_completion_head import _aggregate_episode_metrics  # noqa: E402
from evaluate_completion_head import _episode_metric  # noqa: E402
from evaluate_completion_head import _evaluate_checkpoint_worker  # noqa: E402
from evaluate_completion_head import _frame_metrics  # noqa: E402
from evaluate_completion_head import _read_json  # noqa: E402
from evaluate_completion_head import _setup_logging  # noqa: E402
from evaluate_completion_head import _sigmoid  # noqa: E402
from evaluate_completion_head import _write_json  # noqa: E402

from openpi.training.completion_data import BOUNDARY_COPY_FRAMES  # noqa: E402
from openpi.training.completion_data import BOUNDARY_GROUP  # noqa: E402
from openpi.training.completion_data import SplitManifest  # noqa: E402
from openpi.training.completion_data import audit_boundary_completion_episode_parquet  # noqa: E402
from openpi.training.completion_data import build_boundary_train_sample_set  # noqa: E402

DEFAULT_CONFIG_NAME = "pi05_agilex_breakfast_frozen_head_s2_completion_boundary"
DEFAULT_EXP_NAME = "s2_boundary"
DEFAULT_HF_LEROBOT_HOME = Path("/mnt/data/models/wyt/data")
DEFAULT_CHECKPOINT_BASE = Path("/mnt/data/models/wyt/checkpoints")
DEFAULT_EVALUATION_BASE = Path("/mnt/data/models/wyt/evaluations_boundary")
DEFAULT_DATASET_ROOT = Path(
    "/mnt/data/models/wyt/data/agilex_make_breakfast_subtask_730_frozen_head_completion_boundary"
)

# P2-3: The deployment threshold is fixed at 0.5.  It is not configurable —
# best-F1/threshold are diagnostics only and never used for deployment.
DEPLOYMENT_THRESHOLD = 0.5

# P1-5: Time-delay metrics that are meaningless for the sparse view because
# they rely on array positions (not real frame_index) to compute crossings.
_TIME_METRIC_KEYS = {
    "true_threshold_frame",
    "predicted_threshold_frame",
    "detection_delay_frames",
    "detection_delay_seconds",
    "mean_detection_delay_frames",
    "median_detection_delay_frames",
    "mean_detection_delay_seconds",
    "median_detection_delay_seconds",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _confusion_matrix(scores: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> dict[str, int]:
    preds = scores >= threshold
    pos = targets == 1
    return {
        "tp": int(np.sum(preds & pos)),
        "fp": int(np.sum(preds & ~pos)),
        "fn": int(np.sum(~preds & pos)),
        "tn": int(np.sum(~preds & ~pos)),
    }


def _compute_set_metrics(
    episode_indices: np.ndarray,
    task_indices: np.ndarray,
    frame_indices: np.ndarray,
    logits: np.ndarray,
    targets: np.ndarray,
    infer_ms: np.ndarray,
    *,
    fps: float,
    threshold: float,
    report_time_metrics: bool = True,
) -> dict[str, Any]:
    """Computes overall + per-task + per-episode metrics for a set of predictions.

    When ``report_time_metrics`` is False (sparse view), time-delay metrics
    that depend on array positions (not real frame indices) are nullified
    because the compressed sparse array makes them meaningless.
    """

    scores = _sigmoid(logits)

    episode_rows: list[dict[str, Any]] = []
    for eid in sorted(np.unique(episode_indices).tolist()):
        mask = episode_indices == eid
        ep_task = np.unique(task_indices[mask])
        if len(ep_task) != 1:
            raise ValueError(f"Episode {eid}: expected one task index, got {ep_task.tolist()}")
        order = np.argsort(frame_indices[mask])
        episode_rows.append(
            _episode_metric(
                int(eid),
                int(ep_task[0]),
                logits[mask][order],
                targets[mask][order],
                fps=fps,
                threshold=threshold,
            )
        )

    overall = _frame_metrics(logits, targets)
    overall["confusion_matrix"] = _confusion_matrix(scores, targets, threshold)
    overall.update(_aggregate_episode_metrics(episode_rows))
    overall["mean_infer_ms_per_frame"] = float(np.mean(infer_ms)) if len(infer_ms) > 0 else 0.0

    per_task: dict[str, Any] = {}
    for tid in sorted(np.unique(task_indices).tolist()):
        tmask = task_indices == tid
        task_rows = [r for r in episode_rows if r["task_index"] == int(tid)]
        per_task[str(int(tid))] = {
            **_frame_metrics(logits[tmask], targets[tmask]),
            "confusion_matrix": _confusion_matrix(scores[tmask], targets[tmask], threshold),
            **_aggregate_episode_metrics(task_rows),
        }

    if not report_time_metrics:
        note = (
            "Time-delay metrics are nullified for the sparse view because the "
            "compressed array makes array-position-based delays meaningless."
        )
        for row in [overall, *per_task.values(), *episode_rows]:
            for key in _TIME_METRIC_KEYS:
                if key in row:
                    row[key] = None
        overall["time_metrics_note"] = note

    return {"overall": overall, "per_task": per_task, "episodes": episode_rows}


# ---------------------------------------------------------------------------
# Sparse sampling
# ---------------------------------------------------------------------------


def _read_episodes_lengths(meta_dir: Path) -> dict[int, int]:
    lengths: dict[int, int] = {}
    for line in (meta_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        lengths[int(record["episode_index"])] = int(record["length"])
    return lengths


def _build_sparse_sets(
    dataset_root: Path,
    test_ids: list[int],
    *,
    chunks_size: int,
    stride: int = 15,
    forced_first_n: int = BOUNDARY_COPY_FRAMES,
) -> dict[int, np.ndarray]:
    """Returns ``{episode_id: sparse local frame indices}`` for test episodes."""

    info = _read_json(dataset_root / "meta" / "info.json")
    data_path_template = info["data_path"]
    episode_lengths = _read_episodes_lengths(dataset_root / "meta")
    sparse_sets: dict[int, np.ndarray] = {}
    for eid in test_ids:
        chunk = eid // chunks_size
        parquet_path = dataset_root / data_path_template.format(episode_chunk=chunk, episode_index=eid)
        group_start = (eid // BOUNDARY_GROUP) * BOUNDARY_GROUP
        group_ep_ids = tuple(range(group_start, group_start + BOUNDARY_GROUP))
        group_position = eid % BOUNDARY_GROUP
        audit = audit_boundary_completion_episode_parquet(
            parquet_path,
            episode_id=eid,
            expected_length=episode_lengths[eid],
            group_episode_ids=group_ep_ids,
            group_position=group_position,
        )
        sparse_sets[eid] = build_boundary_train_sample_set(audit, stride=stride, forced_first_n=forced_first_n)
    return sparse_sets


def _sparse_mask(
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    sparse_sets: dict[int, np.ndarray],
) -> np.ndarray:
    """Boolean mask selecting predictions that fall in the sparse sample sets."""

    mask = np.zeros(len(episode_indices), dtype=bool)
    for eid, sparse_frames in sparse_sets.items():
        ep_mask = episode_indices == eid
        if not np.any(ep_mask):
            continue
        mask[ep_mask] = np.isin(frame_indices[ep_mask], sparse_frames)
    return mask


def _validate_prediction_coverage(
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    *,
    arrays: tuple[np.ndarray, ...],
    expected_episode_ids: list[int],
    episode_lengths: dict[int, int],
) -> None:
    """Require exactly one prediction for every expected test frame."""

    expected_count = sum(episode_lengths[eid] for eid in expected_episode_ids)
    all_arrays = (episode_indices, frame_indices, *arrays)
    lengths = {len(value) for value in all_arrays}
    if lengths != {expected_count}:
        raise ValueError(
            f"Prediction arrays must all contain {expected_count} test frames; observed lengths={sorted(lengths)}"
        )

    actual_episode_ids = sorted(int(value) for value in np.unique(episode_indices))
    if actual_episode_ids != sorted(expected_episode_ids):
        raise ValueError(
            f"Prediction episodes do not match the test manifest: "
            f"expected={sorted(expected_episode_ids)}, actual={actual_episode_ids}"
        )

    for episode_id in expected_episode_ids:
        actual_frames = np.sort(frame_indices[episode_indices == episode_id].astype(np.int64, copy=False))
        expected_frames = np.arange(episode_lengths[episode_id], dtype=np.int64)
        if not np.array_equal(actual_frames, expected_frames):
            raise ValueError(
                f"Episode {episode_id} predictions must cover every frame exactly once; "
                f"expected {len(expected_frames)} frames, observed {len(actual_frames)}"
            )


# ---------------------------------------------------------------------------
# Checkpoint worker (subprocess isolation)
# ---------------------------------------------------------------------------


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
    LOGGER.info("Starting isolated checkpoint evaluation: %s", checkpoint_dir.name)
    subprocess.run(command, check=True, env=environment)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _resolve_single_checkpoint(checkpoint_root: Path, step: int) -> Path:
    """Resolves the checkpoint directory for ``step``.

    Checks the managed ``<step>/`` directory first, then falls back to the
    protected ``eval_checkpoint/`` copy (written by train.py when epochs>1 to
    survive max_to_keep=1 cleanup).  When falling back, the protected copy's
    ``_protected_step.json`` marker must match the requested step (P1-A).
    """

    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"Checkpoint experiment directory not found: {checkpoint_root}")
    checkpoint_dir = checkpoint_root / str(step)
    if checkpoint_dir.is_dir() and (checkpoint_dir / "params").is_dir():
        return checkpoint_dir
    # Fall back to the step-specific protected copy and verify its marker.
    protected = checkpoint_root / "eval_checkpoint" / str(step)
    if protected.is_dir() and (protected / "params").is_dir():
        marker_path = protected / "_protected_step.json"
        if not marker_path.is_file():
            raise FileNotFoundError(
                f"Protected eval_checkpoint copy exists at {protected} but has no "
                f"_protected_step.json marker. Cannot verify it matches requested step {step}."
            )
        marker = _read_json(marker_path)
        protected_step = int(marker["step"])
        if protected_step != step:
            raise ValueError(
                f"Protected eval_checkpoint copy is for step {protected_step}, not {step}. "
                f"The requested step was deleted by the checkpoint manager (max_to_keep=1) "
                f"and the protected copy does not match."
            )
        LOGGER.info("Using protected eval_checkpoint copy (managed step %d may have been cleaned up)", step)
        return protected
    available = sorted(p.name for p in checkpoint_root.iterdir() if p.is_dir() and p.name.isdigit())
    raise FileNotFoundError(f"Checkpoint step {step} not found in {checkpoint_root}; available: {available}")


def _load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {
            "episode_index": values["episode_index"],
            "task_index": values["task_index"],
            "frame_index": values["frame_index"],
            "logit": values["logit"],
            "target": values["target"],
            "infer_ms": values["infer_ms"],
        }


def _verify_eval_checkpoint_binding(
    checkpoint_root: Path,
    requested_step: int,
) -> dict[str, Any]:
    """P1-2 + P1-A: Binds the requested checkpoint step to the preregistered step.

    Reads ``eval_checkpoint.json`` from the checkpoint directory.  The file
    must exist (training must have preregistered the eval checkpoint) and the
    requested step must match the preregistered step.  There is no override —
    bypassing preregistration is forbidden to prevent test-set model selection.
    """

    eval_ckpt_path = checkpoint_root / "eval_checkpoint.json"
    if not eval_ckpt_path.is_file():
        raise FileNotFoundError(
            f"No eval_checkpoint.json found in {checkpoint_root}. "
            "The checkpoint step was not preregistered during training. "
            "Bypassing preregistration is forbidden — the eval checkpoint must be "
            "predetermined before training to prevent test-set model selection."
        )

    eval_info = _read_json(eval_ckpt_path)
    preregistered_step = int(eval_info["eval_checkpoint_step"])
    if requested_step != preregistered_step:
        raise ValueError(
            f"Requested checkpoint step {requested_step} does not match the preregistered "
            f"eval checkpoint step {preregistered_step} (from {eval_ckpt_path}). "
            "Bypassing preregistration is forbidden — the eval checkpoint must be "
            "predetermined before training to prevent test-set model selection."
        )
    return {"preregistered": True, "step": preregistered_step}


def _run_evaluation(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset_root.resolve()
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
    dataset_info = _read_json(info_path)
    fps = float(dataset_info["fps"])
    chunks_size = int(dataset_info.get("chunks_size", 1000))

    # Resolve the repo_id from the actual training config rather than trusting a
    # second user-provided string that could disagree with the worker config.
    from openpi.training import config as training_config  # noqa: PLC0415

    train_config = training_config.get_config(args.config_name)
    config_repo_id = getattr(train_config.data, "repo_id", None)
    if not isinstance(config_repo_id, str) or not config_repo_id:
        raise ValueError(f"Training config {args.config_name!r} has no concrete LeRobot repo_id")
    expected_root = (args.hf_lerobot_home.resolve() / config_repo_id).resolve()
    if dataset_root != expected_root:
        raise ValueError(
            f"dataset_root ({dataset_root}) does not match "
            f"hf_lerobot_home / repo_id ({expected_root}). "
            "Inference data and metadata must come from the same dataset."
        )

    checkpoint_root = (args.checkpoint_base / args.config_name / args.exp_name).resolve()

    # P1-2 + P1-A: Bind to the preregistered eval checkpoint (no bypass).
    eval_binding = _verify_eval_checkpoint_binding(
        checkpoint_root,
        args.checkpoint_step,
    )

    checkpoint_dir = _resolve_single_checkpoint(checkpoint_root, args.checkpoint_step)
    checkpoint_step = args.checkpoint_step

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (args.evaluation_base / args.exp_name / f"checkpoint_{checkpoint_step}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir)

    started_at = datetime.now(UTC)
    LOGGER.info("Evaluation output: %s", output_dir)
    LOGGER.info("Dataset: %s", dataset_root)
    LOGGER.info("Checkpoint: %s (step %d)", checkpoint_dir, checkpoint_step)
    LOGGER.info("Threshold: %.2f (fixed; best-F1 is diagnostic only)", DEPLOYMENT_THRESHOLD)
    LOGGER.info("Eval checkpoint binding: %s", eval_binding)

    # --- 1. Run full-frame inference (subprocess) ---
    prediction_file = output_dir / "predictions.npz"
    _run_checkpoint_worker(
        args,
        checkpoint_dir=checkpoint_dir,
        prediction_file=prediction_file,
    )

    # --- 2. Load predictions ---
    preds = _load_predictions(prediction_file)
    ep_idx = preds["episode_index"]
    tk_idx = preds["task_index"]
    fr_idx = preds["frame_index"]
    logits = preds["logit"]
    targets = preds["target"]
    infer_ms = preds["infer_ms"]

    if not np.all(np.logical_or(targets == 0.0, targets == 1.0)):
        raise ValueError("Completion targets must be binary 0/1")

    manifest = train_config.completion
    if manifest.split_manifest_path is None:
        raise ValueError("Boundary evaluation requires completion.split_manifest_path")
    split_manifest = SplitManifest.from_dict(_read_json(Path(manifest.split_manifest_path)))
    test_ids = split_manifest.episode_ids("test")
    _validate_prediction_coverage(
        ep_idx,
        fr_idx,
        arrays=(tk_idx, logits, targets, infer_ms),
        expected_episode_ids=test_ids,
        episode_lengths=_read_episodes_lengths(dataset_root / "meta"),
    )
    LOGGER.info("Test episodes: %d, total frames: %d", len(test_ids), len(logits))

    # --- 3. test_full metrics (all frames) ---
    test_full = _compute_set_metrics(
        ep_idx,
        tk_idx,
        fr_idx,
        logits,
        targets,
        infer_ms,
        fps=fps,
        threshold=DEPLOYMENT_THRESHOLD,
    )

    # --- 4. test_sparse metrics (training-equivalent sampling) ---
    # P1-5: Time-delay metrics are nullified for the sparse view because the
    # compressed array makes array-position-based delays meaningless.
    sparse_sets = _build_sparse_sets(
        dataset_root,
        test_ids,
        chunks_size=chunks_size,
        stride=args.stride,
        forced_first_n=BOUNDARY_COPY_FRAMES,
    )
    smask = _sparse_mask(ep_idx, fr_idx, sparse_sets)
    LOGGER.info(
        "Sparse frames: %d / %d (%.1f%%)", int(np.sum(smask)), len(smask), 100 * np.sum(smask) / max(len(smask), 1)
    )
    test_sparse = _compute_set_metrics(
        ep_idx[smask],
        tk_idx[smask],
        fr_idx[smask],
        logits[smask],
        targets[smask],
        infer_ms[smask],
        fps=fps,
        threshold=DEPLOYMENT_THRESHOLD,
        report_time_metrics=False,
    )

    # --- 5. Summary ---
    finished_at = datetime.now(UTC)
    summary = {
        "result": "success",
        "config_name": args.config_name,
        "experiment_name": args.exp_name,
        "dataset_root": str(dataset_root),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_step": checkpoint_step,
        "eval_checkpoint_binding": eval_binding,
        "threshold": DEPLOYMENT_THRESHOLD,
        "threshold_note": "fixed at 0.5 for deployment; best-threshold is diagnostic only",
        "test_episode_count": len(test_ids),
        "stride": args.stride,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": (finished_at - started_at).total_seconds(),
        "test_full": {
            "frame_count": len(logits),
            "overall": test_full["overall"],
            "per_task": test_full["per_task"],
        },
        "test_sparse": {
            "frame_count": int(np.sum(smask)),
            "overall": test_sparse["overall"],
            "per_task": test_sparse["per_task"],
        },
        "diagnostic": {
            "test_full_best_f1": test_full["overall"]["best_f1"],
            "test_full_best_threshold": test_full["overall"]["best_threshold"],
            "test_full_best_precision": test_full["overall"]["best_precision"],
            "test_full_best_recall": test_full["overall"]["best_recall"],
            "test_sparse_best_f1": test_sparse["overall"]["best_f1"],
            "test_sparse_best_threshold": test_sparse["overall"]["best_threshold"],
            "note": "best-F1/threshold are diagnostics only; deployment threshold is fixed at 0.5",
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "test_full_episodes.json", test_full["episodes"])
    _write_json(output_dir / "test_sparse_episodes.json", test_sparse["episodes"])

    LOGGER.info(
        "test_full:  BCE=%.4f F1@0.5=%.4f AUC=%.4f  |  "
        "test_sparse: BCE=%.4f F1@0.5=%.4f AUC=%.4f  |  "
        "diagnostic best_f1=%.4f@%.4f",
        test_full["overall"]["bce"],
        test_full["overall"]["f1_at_0.5"],
        test_full["overall"]["auc"],
        test_sparse["overall"]["bce"],
        test_sparse["overall"]["f1_at_0.5"],
        test_sparse["overall"]["auc"],
        test_full["overall"]["best_f1"],
        test_full["overall"]["best_threshold"],
    )
    LOGGER.info("Evaluation complete: %s", output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--checkpoint-base", type=Path, default=DEFAULT_CHECKPOINT_BASE)
    parser.add_argument("--evaluation-base", type=Path, default=DEFAULT_EVALUATION_BASE)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--hf-lerobot-home", type=Path, default=DEFAULT_HF_LEROBOT_HOME)
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        required=True,
        help="Single explicit checkpoint step to evaluate (no 'latest', no multi-checkpoint). "
        "Must match the step preregistered in eval_checkpoint.json during training.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=15, help="Negative sampling stride for test_sparse.")
    # Hidden args for the subprocess worker.
    parser.add_argument("--worker-checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.stride < 1:
        parser.error("--stride must be at least 1")
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
        LOGGER.exception("Boundary completion evaluation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
