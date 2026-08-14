"""Tests for scripts/evaluate_completion_boundary.py.

Covers: single-checkpoint enforcement, test_full/test_sparse metric keys,
confusion-matrix correctness, sparse-mask correctness, and best-F1 being
diagnostic-only (not used as the deployment threshold).
"""

# ruff: noqa: SLF001  -- tests intentionally access private functions
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pytest

from scripts import evaluate_completion_boundary as ecb

# ---------------------------------------------------------------------------
# _confusion_matrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_all_tp():
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    targets = np.array([1, 1, 1], dtype=np.float32)
    cm = ecb._confusion_matrix(scores, targets, threshold=0.5)
    assert cm == {"tp": 3, "fp": 0, "fn": 0, "tn": 0}


def test_confusion_matrix_all_tn():
    scores = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    targets = np.array([0, 0, 0], dtype=np.float32)
    cm = ecb._confusion_matrix(scores, targets, threshold=0.5)
    assert cm == {"tp": 0, "fp": 0, "fn": 0, "tn": 3}


def test_confusion_matrix_mixed():
    scores = np.array([0.9, 0.1, 0.8, 0.3], dtype=np.float32)
    targets = np.array([1, 1, 0, 0], dtype=np.float32)
    cm = ecb._confusion_matrix(scores, targets, threshold=0.5)
    # TP: score>=0.5 & target==1 → index 0
    # FP: score>=0.5 & target==0 → index 2
    # FN: score<0.5 & target==1 → index 1
    # TN: score<0.5 & target==0 → index 3
    assert cm == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}


# ---------------------------------------------------------------------------
# _compute_set_metrics
# ---------------------------------------------------------------------------


def _make_synthetic_predictions(num_episodes=2, frames_per_ep=20):
    """Creates synthetic prediction arrays for testing metric computation."""
    ep_idx = np.concatenate([np.full(frames_per_ep, eid, dtype=np.int64) for eid in range(num_episodes)])
    tk_idx = np.concatenate([np.full(frames_per_ep, eid % 4, dtype=np.int64) for eid in range(num_episodes)])
    fr_idx = np.concatenate([np.arange(frames_per_ep, dtype=np.int64) for _ in range(num_episodes)])
    # Last 10 frames positive, rest negative (matches boundary scheme).
    targets = np.concatenate(
        [
            np.concatenate([np.zeros(frames_per_ep - 10, dtype=np.float32), np.ones(10, dtype=np.float32)])
            for _ in range(num_episodes)
        ]
    )
    # Good predictions: high logits for positives, low for negatives.
    logits = np.where(targets == 1, 3.0, -3.0).astype(np.float32)
    infer_ms = np.ones(len(logits), dtype=np.float32)
    return ep_idx, tk_idx, fr_idx, logits, targets, infer_ms


def test_compute_set_metrics_has_required_keys():
    ep_idx, tk_idx, fr_idx, logits, targets, infer_ms = _make_synthetic_predictions()
    result = ecb._compute_set_metrics(
        ep_idx,
        tk_idx,
        fr_idx,
        logits,
        targets,
        infer_ms,
        fps=30.0,
        threshold=0.5,
    )

    overall = result["overall"]
    # Core metrics.
    for key in (
        "bce",
        "auc",
        "f1_at_0.5",
        "precision_at_0.5",
        "recall_at_0.5",
        "best_f1",
        "best_threshold",
        "confusion_matrix",
        "mean_infer_ms_per_frame",
    ):
        assert key in overall, f"missing key: {key}"

    # Confusion matrix keys.
    for key in ("tp", "fp", "fn", "tn"):
        assert key in overall["confusion_matrix"]

    # Per-task and per-episode.
    assert isinstance(result["per_task"], dict)
    assert len(result["episodes"]) == 2


def test_compute_set_metrics_perfect_predictions():
    ep_idx, tk_idx, fr_idx, logits, targets, infer_ms = _make_synthetic_predictions()
    result = ecb._compute_set_metrics(
        ep_idx,
        tk_idx,
        fr_idx,
        logits,
        targets,
        infer_ms,
        fps=30.0,
        threshold=0.5,
    )
    overall = result["overall"]
    # Perfect predictions → F1=1.0, confusion has no FP/FN.
    assert overall["f1_at_0.5"] == pytest.approx(1.0)
    assert overall["confusion_matrix"]["fp"] == 0
    assert overall["confusion_matrix"]["fn"] == 0
    assert overall["auc"] == pytest.approx(1.0)


def test_compute_set_metrics_best_f1_is_diagnostic():
    """best_f1/best_threshold are present but separate from the fixed 0.5 threshold."""

    ep_idx, tk_idx, fr_idx, logits, targets, infer_ms = _make_synthetic_predictions()
    # Add a few misclassified to make best_threshold differ from 0.5.
    logits[:3] = 2.0  # false positives
    targets[:3] = 0.0

    result = ecb._compute_set_metrics(
        ep_idx,
        tk_idx,
        fr_idx,
        logits,
        targets,
        infer_ms,
        fps=30.0,
        threshold=0.5,
    )
    overall = result["overall"]
    # best_f1 >= f1_at_0.5 (best_threshold optimizes F1).
    assert overall["best_f1"] >= overall["f1_at_0.5"]
    # best_threshold is reported as a diagnostic, not used for deployment.
    assert "best_threshold" in overall


def test_compute_set_metrics_sparse_nullifies_time_metrics():
    """P1-5: sparse view must nullify time-delay metrics and add a note."""

    ep_idx, tk_idx, fr_idx, logits, targets, infer_ms = _make_synthetic_predictions()
    result = ecb._compute_set_metrics(
        ep_idx,
        tk_idx,
        fr_idx,
        logits,
        targets,
        infer_ms,
        fps=30.0,
        threshold=0.5,
        report_time_metrics=False,
    )
    overall = result["overall"]
    # Time-delay keys must be None.
    for key in ecb._TIME_METRIC_KEYS:
        assert key not in overall or overall[key] is None, f"{key} should be None for sparse view"
    # Must have the explanatory note.
    assert "time_metrics_note" in overall
    # Per-task time metrics also nullified.
    for task_metrics in result["per_task"].values():
        for key in ecb._TIME_METRIC_KEYS:
            assert key not in task_metrics or task_metrics[key] is None


def test_compute_set_metrics_full_keeps_time_metrics():
    """P1-5: full view must keep time-delay metrics (not nullified)."""

    ep_idx, tk_idx, fr_idx, logits, targets, infer_ms = _make_synthetic_predictions()
    result = ecb._compute_set_metrics(
        ep_idx,
        tk_idx,
        fr_idx,
        logits,
        targets,
        infer_ms,
        fps=30.0,
        threshold=0.5,
        report_time_metrics=True,
    )
    overall = result["overall"]
    # Full view should NOT have the nullification note.
    assert "time_metrics_note" not in overall


# ---------------------------------------------------------------------------
# _resolve_single_checkpoint
# ---------------------------------------------------------------------------


def test_resolve_single_checkpoint_succeeds_for_existing_step(tmp_path):
    checkpoint_root = tmp_path / "ckpts"
    step_dir = checkpoint_root / "200"
    (step_dir / "params").mkdir(parents=True)

    resolved = ecb._resolve_single_checkpoint(checkpoint_root, 200)
    assert resolved == step_dir


def test_resolve_single_checkpoint_rejects_missing_step(tmp_path):
    checkpoint_root = tmp_path / "ckpts"
    (checkpoint_root / "100" / "params").mkdir(parents=True)
    (checkpoint_root / "200" / "params").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Checkpoint step 999 not found"):
        ecb._resolve_single_checkpoint(checkpoint_root, 999)


def test_resolve_single_checkpoint_rejects_nonexistent_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint experiment directory not found"):
        ecb._resolve_single_checkpoint(tmp_path / "nonexistent", 100)


def test_resolve_single_checkpoint_rejects_missing_params(tmp_path):
    checkpoint_root = tmp_path / "ckpts"
    (checkpoint_root / "100").mkdir(parents=True)  # no params/ subdir

    with pytest.raises(FileNotFoundError, match="Checkpoint step 100 not found"):
        ecb._resolve_single_checkpoint(checkpoint_root, 100)


def test_resolve_single_checkpoint_no_latest_alias(tmp_path):
    """There is no 'latest' alias — only explicit integer steps are accepted."""

    checkpoint_root = tmp_path / "ckpts"
    (checkpoint_root / "100" / "params").mkdir(parents=True)
    # "latest" is not a valid step directory name (it's not an int directory
    # and _resolve_single_checkpoint looks for str(step) = "999").
    with pytest.raises(FileNotFoundError, match="not found"):
        ecb._resolve_single_checkpoint(checkpoint_root, 999)


def test_resolve_single_checkpoint_falls_back_to_protected_copy(tmp_path):
    """When the managed step dir was cleaned up (max_to_keep=1), fall back to
    eval_checkpoint/ — but only if the _protected_step.json marker matches."""

    checkpoint_root = tmp_path / "ckpts"
    # Simulate: step 200 was deleted by the checkpoint manager, but a protected
    # copy exists at eval_checkpoint/<step>/ with a matching step marker.
    protected = checkpoint_root / "eval_checkpoint" / "200"
    (protected / "params").mkdir(parents=True)
    (protected / "_protected_step.json").write_text(json.dumps({"step": 200}), encoding="utf-8")
    resolved = ecb._resolve_single_checkpoint(checkpoint_root, 200)
    assert resolved == protected


def test_resolve_single_checkpoint_rejects_protected_step_mismatch(tmp_path):
    """P1-A: Protected copy with a different step marker must not masquerade
    as the requested step."""

    checkpoint_root = tmp_path / "ckpts"
    protected = checkpoint_root / "eval_checkpoint" / "300"
    (protected / "params").mkdir(parents=True)
    (protected / "_protected_step.json").write_text(json.dumps({"step": 500}), encoding="utf-8")
    # Requesting step 300, but the protected copy is for step 500.
    with pytest.raises(ValueError, match="Protected eval_checkpoint copy is for step 500, not 300"):
        ecb._resolve_single_checkpoint(checkpoint_root, 300)


def test_resolve_single_checkpoint_rejects_protected_without_marker(tmp_path):
    """P1-A: Protected copy without _protected_step.json cannot be verified."""

    checkpoint_root = tmp_path / "ckpts"
    (checkpoint_root / "eval_checkpoint" / "200" / "params").mkdir(parents=True)
    # No _protected_step.json marker.
    with pytest.raises(FileNotFoundError, match=r"no _protected_step\.json marker"):
        ecb._resolve_single_checkpoint(checkpoint_root, 200)


def test_resolve_single_checkpoint_prefers_managed_over_protected(tmp_path):
    """If both the managed dir and the protected copy exist, prefer the managed dir."""

    checkpoint_root = tmp_path / "ckpts"
    (checkpoint_root / "200" / "params").mkdir(parents=True)
    protected = checkpoint_root / "eval_checkpoint" / "200"
    (protected / "params").mkdir(parents=True)
    (protected / "_protected_step.json").write_text(json.dumps({"step": 200}), encoding="utf-8")
    resolved = ecb._resolve_single_checkpoint(checkpoint_root, 200)
    assert resolved == checkpoint_root / "200"


def test_resolve_single_checkpoint_protected_missing_params(tmp_path):
    """Protected copy without params/ is not a valid fallback."""

    checkpoint_root = tmp_path / "ckpts"
    protected = checkpoint_root / "eval_checkpoint" / "200"
    protected.mkdir(parents=True)  # no params/
    (protected / "_protected_step.json").write_text(json.dumps({"step": 200}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="not found"):
        ecb._resolve_single_checkpoint(checkpoint_root, 200)


# ---------------------------------------------------------------------------
# _verify_eval_checkpoint_binding
# ---------------------------------------------------------------------------


def test_verify_binding_matches_preregistered(tmp_path):
    """When the requested step matches eval_checkpoint.json, binding succeeds."""

    checkpoint_root = tmp_path / "ckpts"
    checkpoint_root.mkdir()
    (checkpoint_root / "eval_checkpoint.json").write_text(json.dumps({"eval_checkpoint_step": 500}), encoding="utf-8")
    result = ecb._verify_eval_checkpoint_binding(checkpoint_root, 500)
    assert result["preregistered"] is True
    assert result["step"] == 500


def test_verify_binding_rejects_mismatch(tmp_path):
    """P1-A: Mismatched step raises ValueError — no override allowed."""

    checkpoint_root = tmp_path / "ckpts"
    checkpoint_root.mkdir()
    (checkpoint_root / "eval_checkpoint.json").write_text(json.dumps({"eval_checkpoint_step": 500}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the preregistered"):
        ecb._verify_eval_checkpoint_binding(checkpoint_root, 300)


def test_verify_binding_rejects_missing_json(tmp_path):
    """P1-A: No eval_checkpoint.json raises FileNotFoundError — no bypass."""

    checkpoint_root = tmp_path / "ckpts"
    checkpoint_root.mkdir()
    with pytest.raises(FileNotFoundError, match=r"No eval_checkpoint\.json found"):
        ecb._verify_eval_checkpoint_binding(checkpoint_root, 500)


# ---------------------------------------------------------------------------
# _sparse_mask
# ---------------------------------------------------------------------------


def test_sparse_mask_selects_correct_frames():
    """Sparse mask selects only frames in the sparse sample sets."""

    ep_idx = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)
    fr_idx = np.array([0, 1, 2, 10, 15, 0, 1, 5, 10, 15], dtype=np.int64)
    sparse_sets = {
        0: np.array([0, 10, 15], dtype=np.int64),
        1: np.array([0, 5, 15], dtype=np.int64),
    }
    mask = ecb._sparse_mask(ep_idx, fr_idx, sparse_sets)
    # Episode 0: frames 0, 10, 15 in sparse → indices 0, 3, 4
    # Episode 1: frames 0, 5, 15 in sparse → indices 5, 7, 9
    expected = np.array([True, False, False, True, True, True, False, True, False, True])
    np.testing.assert_array_equal(mask, expected)


def test_sparse_mask_empty_for_unsampled_episodes():
    ep_idx = np.array([0, 0, 1, 1], dtype=np.int64)
    fr_idx = np.array([0, 1, 0, 1], dtype=np.int64)
    sparse_sets = {0: np.array([0], dtype=np.int64)}  # episode 1 not in sparse
    mask = ecb._sparse_mask(ep_idx, fr_idx, sparse_sets)
    expected = np.array([True, False, False, False])
    np.testing.assert_array_equal(mask, expected)


def test_prediction_coverage_requires_every_frame_exactly_once():
    ep_idx = np.array([4, 4, 5, 5, 5], dtype=np.int64)
    fr_idx = np.array([0, 1, 0, 1, 2], dtype=np.int64)
    payload = tuple(np.zeros(5) for _ in range(4))
    ecb._validate_prediction_coverage(
        ep_idx,
        fr_idx,
        arrays=payload,
        expected_episode_ids=[4, 5],
        episode_lengths={4: 2, 5: 3},
    )


def test_prediction_coverage_rejects_duplicate_and_missing_frame():
    ep_idx = np.array([4, 4], dtype=np.int64)
    fr_idx = np.array([0, 0], dtype=np.int64)
    payload = tuple(np.zeros(2) for _ in range(4))
    with pytest.raises(ValueError, match="every frame exactly once"):
        ecb._validate_prediction_coverage(
            ep_idx,
            fr_idx,
            arrays=payload,
            expected_episode_ids=[4],
            episode_lengths={4: 2},
        )


# ---------------------------------------------------------------------------
# CLI: --checkpoint-step is required
# ---------------------------------------------------------------------------


def test_cli_requires_checkpoint_step(monkeypatch):
    """The CLI must require an explicit --checkpoint-step (no default, no 'latest')."""

    monkeypatch.setattr(sys, "argv", ["evaluate_completion_boundary.py"])
    with pytest.raises(SystemExit):
        ecb._parse_args()


def test_cli_checkpoint_step_must_be_int(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_completion_boundary.py",
            "--checkpoint-step",
            "latest",
        ],
    )
    with pytest.raises(SystemExit):
        ecb._parse_args()


def test_cli_accepts_explicit_int_step(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_completion_boundary.py",
            "--checkpoint-step",
            "500",
        ],
    )
    args = ecb._parse_args()
    assert args.checkpoint_step == 500
    assert not hasattr(args, "config_repo_id")
    assert not hasattr(args, "allow_unregistered_checkpoint")


def test_checkpoint_worker_passes_required_preregistered_step(tmp_path, monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr(ecb.subprocess, "run", fake_run)
    args = argparse.Namespace(
        config_name=ecb.DEFAULT_CONFIG_NAME,
        batch_size=8,
        num_workers=16,
        seed=42,
        checkpoint_step=3000,
        hf_lerobot_home=tmp_path,
    )
    ecb._run_checkpoint_worker(
        args,
        checkpoint_dir=tmp_path / "3000",
        prediction_file=tmp_path / "predictions.npz",
    )

    command = captured["command"]
    step_position = command.index("--checkpoint-step")
    assert command[step_position + 1] == "3000"
    workers_position = command.index("--num-workers")
    assert command[workers_position + 1] == "16"
    assert captured["kwargs"]["check"] is True


# ---------------------------------------------------------------------------
# Summary structure: best-F1 is diagnostic, threshold is fixed
# ---------------------------------------------------------------------------


def test_summary_diagnostic_structure():
    """The summary template separates diagnostic best_f1 from the fixed threshold.

    We verify the key layout by constructing a minimal summary from
    _compute_set_metrics output (same structure _run_evaluation uses).
    """

    ep_idx, tk_idx, fr_idx, logits, targets, infer_ms = _make_synthetic_predictions()
    test_full = ecb._compute_set_metrics(
        ep_idx,
        tk_idx,
        fr_idx,
        logits,
        targets,
        infer_ms,
        fps=30.0,
        threshold=0.5,
    )
    test_sparse = ecb._compute_set_metrics(
        ep_idx,
        tk_idx,
        fr_idx,
        logits,
        targets,
        infer_ms,
        fps=30.0,
        threshold=0.5,
    )

    # Mirror the summary structure from _run_evaluation.
    threshold = 0.5
    summary = {
        "threshold": threshold,
        "threshold_note": "fixed at 0.5 for deployment; best-threshold is diagnostic only",
        "test_full": {"overall": test_full["overall"]},
        "test_sparse": {"overall": test_sparse["overall"]},
        "diagnostic": {
            "test_full_best_f1": test_full["overall"]["best_f1"],
            "test_full_best_threshold": test_full["overall"]["best_threshold"],
            "note": "best-F1/threshold are diagnostics only; deployment threshold is fixed at 0.5",
        },
    }

    # The deployment threshold is the fixed 0.5, NOT best_threshold.
    assert summary["threshold"] == 0.5
    assert summary["threshold_note"] != ""
    # best_f1 lives under "diagnostic", separate from the deployment threshold.
    assert "test_full_best_f1" in summary["diagnostic"]
    assert "test_full_best_threshold" in summary["diagnostic"]
    # The test_full overall also has confusion_matrix at the fixed threshold.
    assert "confusion_matrix" in summary["test_full"]["overall"]
    assert "confusion_matrix" in summary["test_sparse"]["overall"]
