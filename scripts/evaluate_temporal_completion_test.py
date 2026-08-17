"""Focused protocol tests for evaluate_temporal_completion.py."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from openpi.training import temporal_completion_data as temporal_data
from openpi.training import temporal_completion_metrics as temporal_metrics
from scripts import evaluate_temporal_completion as evaluator


def _selection() -> temporal_metrics.ThresholdSelection:
    return temporal_metrics.ThresholdSelection(
        threshold=0.8,
        validation_event_count=1,
        validation_early_trigger_events=0,
        validation_event_recall=1.0,
        validation_event_f1=1.0,
        candidate_count=3,
    )


def _write_selection(
    root: Path,
    *,
    step: int = 200,
    last_validated_step: int | None = None,
    selected_on_split: str = "val",
    temporal_input_mode: str = "history",
) -> None:
    selection = dataclasses.asdict(_selection())
    selection["selected_on_split"] = selected_on_split
    (root / evaluator.SELECTION_FILE_NAME).write_text(
        json.dumps(
            {
                "checkpoint_step": step,
                "last_validated_checkpoint_step": step if last_validated_step is None else last_validated_step,
                "validation_rank": [1.0, 0.9, 0.8, 0.7],
                "threshold_selection": selection,
                "manifest_fingerprint": "a" * 64,
                "feature_cache_checkpoint_fingerprint": "b" * 64,
                "feature_cache_rows_fingerprint": "c" * 64,
                "feature_cache_preprocess_fingerprint": "d" * 64,
                "feature_cache_payload_fingerprint": "e" * 64,
                "temporal_input_mode": temporal_input_mode,
            }
        ),
        encoding="utf-8",
    )


def _event(
    *,
    trajectory_id: str,
    full_episode_id: int,
    split: temporal_data.SplitName,
    scores: tuple[float, float, float],
) -> tuple[tuple[temporal_data.TemporalSampleRow, ...], np.ndarray]:
    rows: list[temporal_data.TemporalSampleRow] = []
    for logical_tick in (30, 45, 60):
        distance = 60 - logical_tick
        label = int(distance == 0)
        sample_kind: temporal_data.SampleKind = "positive" if label else "hard_negative"
        rows.append(
            temporal_data.TemporalSampleRow(
                trajectory_id=trajectory_id,
                full_episode_id=full_episode_id,
                task_index=0,
                split=split,
                logical_tick=logical_tick,
                label=label,
                sample_kind=sample_kind,
                boundary_tick=60,
                prompt_index=0,
                history_logical_ticks=(logical_tick - 30, logical_tick - 15, logical_tick),
                source_episode_ids=(full_episode_id,) * 3,
                source_frame_indices=(logical_tick - 30, logical_tick - 15, logical_tick),
                terminal_hold_flags=(False, False, False),
            )
        )
    return tuple(rows), np.asarray(scores, dtype=np.float64)


def test_validation_artifact_resolves_only_its_exact_retained_checkpoint(tmp_path: Path) -> None:
    _write_selection(tmp_path, step=200)
    selected_params = tmp_path / "200" / "params"
    selected_params.mkdir(parents=True)
    # A later checkpoint exists, but the evaluator must never infer "latest".
    (tmp_path / "999" / "params").mkdir(parents=True)

    artifact = evaluator.load_validation_artifact(
        tmp_path,
        manifest_fingerprint="a" * 64,
        feature_cache_checkpoint_fingerprint="b" * 64,
        feature_cache_rows_fingerprint="c" * 64,
        feature_cache_preprocess_fingerprint="d" * 64,
        feature_cache_payload_fingerprint="e" * 64,
        temporal_input_mode="history",
    )

    assert artifact.checkpoint_step == 200
    assert artifact.last_validated_checkpoint_step == 200
    assert artifact.threshold_selection.selected_on_split == "val"
    assert evaluator.resolve_validation_checkpoint(tmp_path, artifact) == tmp_path.resolve() / "200"

    selected_params.rmdir()
    with pytest.raises(FileNotFoundError, match="validation-selected retained checkpoint"):
        evaluator.resolve_validation_checkpoint(tmp_path, artifact)


def test_validation_artifact_rejects_non_validation_threshold(tmp_path: Path) -> None:
    _write_selection(tmp_path, selected_on_split="test")

    with pytest.raises(ValueError, match="originate from validation"):
        evaluator.load_validation_artifact(
            tmp_path,
            manifest_fingerprint="a" * 64,
            feature_cache_checkpoint_fingerprint="b" * 64,
            feature_cache_rows_fingerprint="c" * 64,
            feature_cache_preprocess_fingerprint="d" * 64,
            feature_cache_payload_fingerprint="e" * 64,
            temporal_input_mode="history",
        )


def test_validation_artifact_rejects_temporal_input_mode_mismatch(tmp_path: Path) -> None:
    _write_selection(tmp_path, temporal_input_mode="history")

    with pytest.raises(ValueError, match="temporal_input_mode does not match"):
        evaluator.load_validation_artifact(
            tmp_path,
            manifest_fingerprint="a" * 64,
            feature_cache_checkpoint_fingerprint="b" * 64,
            feature_cache_rows_fingerprint="c" * 64,
            feature_cache_preprocess_fingerprint="d" * 64,
            feature_cache_payload_fingerprint="e" * 64,
            temporal_input_mode="current_only",
        )


def test_completed_run_guard_rejects_stale_progress_and_missing_final_checkpoint(tmp_path: Path) -> None:
    _write_selection(tmp_path, step=200, last_validated_step=200)
    artifact = evaluator.load_validation_artifact(
        tmp_path,
        manifest_fingerprint="a" * 64,
        feature_cache_checkpoint_fingerprint="b" * 64,
        feature_cache_rows_fingerprint="c" * 64,
        feature_cache_preprocess_fingerprint="d" * 64,
        feature_cache_payload_fingerprint="e" * 64,
        temporal_input_mode="history",
    )

    with pytest.raises(ValueError, match="does not prove a completed training run"):
        evaluator.require_completed_temporal_run(tmp_path, artifact, expected_final_step=400)

    completed_artifact = dataclasses.replace(artifact, last_validated_checkpoint_step=400)
    with pytest.raises(FileNotFoundError, match="missing its final checkpoint params"):
        evaluator.require_completed_temporal_run(tmp_path, completed_artifact, expected_final_step=400)

    final_params = tmp_path / "400" / "params"
    final_params.mkdir(parents=True)
    assert (
        evaluator.require_completed_temporal_run(
            tmp_path,
            completed_artifact,
            expected_final_step=400,
        )
        == tmp_path.resolve() / "400"
    )


def test_oracle_evaluator_uses_fail_closed_test_entry_point_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    validation_rows, validation_scores = _event(
        trajectory_id="validation",
        full_episode_id=1,
        split="val",
        scores=(0.1, 0.4, 0.8),
    )
    test_rows, test_scores = _event(
        trajectory_id="test",
        full_episode_id=2,
        split="test",
        scores=(0.2, 0.3, 0.9),
    )
    selection = temporal_metrics.select_validation_threshold(validation_rows, validation_scores)
    original = temporal_metrics.evaluate_test_with_validation_threshold
    call_count = 0

    def counted_test_evaluation(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(temporal_metrics, "evaluate_test_with_validation_threshold", counted_test_evaluation)
    report = evaluator.evaluate_oracle_prompt_scores(
        validation_rows=validation_rows,
        validation_scores=validation_scores,
        test_rows=test_rows,
        test_scores=test_scores,
        stored_selection=selection,
    )

    assert call_count == 1
    assert report["validation"]["overall"]["natural/sample_count"] == 3.0
    assert report["test"]["per_task"]["0"]["threshold/event_recall"] == 1.0


def test_oracle_evaluator_fails_closed_before_test_for_non_val_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    validation_rows, validation_scores = _event(
        trajectory_id="validation",
        full_episode_id=1,
        split="val",
        scores=(0.1, 0.4, 0.8),
    )
    test_rows, test_scores = _event(
        trajectory_id="test",
        full_episode_id=2,
        split="test",
        scores=(0.2, 0.3, 0.9),
    )
    invalid = dataclasses.replace(
        temporal_metrics.select_validation_threshold(validation_rows, validation_scores),
        selected_on_split="test",  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        temporal_metrics,
        "evaluate_test_with_validation_threshold",
        lambda *_args, **_kwargs: pytest.fail("test must not be touched after an invalid threshold binding"),
    )

    with pytest.raises(ValueError, match="originate from validation"):
        evaluator.evaluate_oracle_prompt_scores(
            validation_rows=validation_rows,
            validation_scores=validation_scores,
            test_rows=test_rows,
            test_scores=test_scores,
            stored_selection=invalid,
        )
