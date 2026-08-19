from __future__ import annotations

import math

import numpy as np
import pytest

from openpi.training import temporal_completion_data as temporal_data
from openpi.training import temporal_completion_metrics as metrics


def _event(
    *,
    trajectory_id: str,
    full_episode_id: int,
    task_index: int,
    split: temporal_data.SplitName,
    boundary_tick: int,
    score_by_tick: dict[int, float],
) -> tuple[tuple[temporal_data.TemporalSampleRow, ...], np.ndarray]:
    rows: list[temporal_data.TemporalSampleRow] = []
    scores: list[float] = []
    for tick, score in sorted(score_by_tick.items()):
        distance = boundary_tick - tick
        if distance == 0:
            label = 1
            sample_kind: temporal_data.SampleKind = "positive"
        elif distance == 15:
            label = 0
            sample_kind = "hard_negative"
        else:
            label = 0
            sample_kind = "ordinary_negative"
        source_episode_id = full_episode_id * 4 + task_index
        rows.append(
            temporal_data.TemporalSampleRow(
                trajectory_id=trajectory_id,
                full_episode_id=full_episode_id,
                task_index=task_index,
                split=split,
                logical_tick=tick,
                label=label,
                sample_kind=sample_kind,
                boundary_tick=boundary_tick,
                prompt_index=task_index,
                history_logical_ticks=(tick - 30, tick - 15, tick),
                source_episode_ids=(source_episode_id,) * 3,
                source_frame_indices=(tick - 30, tick - 15, tick),
                terminal_hold_flags=(False, False, False),
            )
        )
        scores.append(score)
    return tuple(rows), np.asarray(scores, dtype=np.float64)


def _join(*parts: tuple[tuple[temporal_data.TemporalSampleRow, ...], np.ndarray]):
    rows = tuple(row for part_rows, _ in parts for row in part_rows)
    scores = np.concatenate([part_scores for _, part_scores in parts])
    return rows, scores


def test_stable_sigmoid_handles_extreme_logits_without_overflow():
    result = metrics.stable_sigmoid(np.asarray([-np.inf, -1000.0, 0.0, 1000.0, np.inf]))

    np.testing.assert_array_equal(result[[0, 1]], np.asarray([0.0, 0.0]))
    assert result[2] == 0.5
    np.testing.assert_array_equal(result[[3, 4]], np.asarray([1.0, 1.0]))
    with pytest.raises(ValueError, match="NaN"):
        metrics.stable_sigmoid(np.asarray([np.nan]))


def test_binary_ranking_metrics_are_tie_aware_and_handle_missing_class():
    ranking = metrics.binary_ranking_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])

    assert ranking["auprc"] == pytest.approx(5.0 / 6.0)
    assert ranking["roc_auc"] == pytest.approx(0.75)

    tied = metrics.binary_ranking_metrics([0, 1], [0.5, 0.5])
    assert tied["auprc"] == pytest.approx(0.5)
    assert tied["roc_auc"] == pytest.approx(0.5)

    positive_only = metrics.binary_ranking_metrics([1, 1], [0.2, 0.8])
    assert positive_only["auprc"] == 1.0
    assert math.isnan(positive_only["roc_auc"])

    negative_only = metrics.binary_ranking_metrics([0, 0], [0.2, 0.8])
    assert math.isnan(negative_only["auprc"])
    assert math.isnan(negative_only["roc_auc"])


def test_report_quantifies_all_history_hard_local_margins_and_tasks():
    rows, scores = _join(
        _event(
            trajectory_id="trajectory-a",
            full_episode_id=0,
            task_index=0,
            split="val",
            boundary_tick=120,
            score_by_tick={30: 0.9, 45: 0.1, 60: 0.2, 75: 0.3, 90: 0.4, 105: 0.8, 120: 0.85},
        ),
        _event(
            trajectory_id="trajectory-a",
            full_episode_id=0,
            task_index=1,
            split="val",
            boundary_tick=210,
            score_by_tick={180: 0.1, 195: 0.2, 210: 0.7},
        ),
    )

    report = metrics.evaluate_temporal_completion(rows, scores=scores)

    first, second = report.event_margins
    assert first.margin == pytest.approx(-0.05)
    assert first.hard_local_margin == pytest.approx(0.05)
    assert first.boundary_top1 is False
    assert second.margin == pytest.approx(0.5)
    assert second.boundary_top1 is True
    assert report.metrics["boundary_top1_rate"] == 0.5
    assert report.metrics["margin/all_gt_zero_rate"] == 0.5
    assert report.metrics["margin/hard_local_gt_zero_rate"] == 1.0
    assert report.metrics["task_0/event_count"] == 1.0
    assert report.metrics["task_1/event_count"] == 1.0
    assert report.metrics["task_2/event_count"] == 0.0
    assert report.metrics["natural/sample_count"] == 10.0
    assert report.metrics["hard_local/sample_count"] == 4.0
    assert report.metrics["negative_smoothness/pair_count"] == 6.0
    assert report.metrics["negative_smoothness/adjacent_abs_delta_mean"] == pytest.approx(1.6 / 6.0)
    assert report.metrics["negative_smoothness/adjacent_abs_delta_p95"] == pytest.approx(0.7)
    assert report.metrics["task_1/negative_smoothness/pair_count"] == 1.0


def test_validation_threshold_prioritises_zero_early_events_before_recall():
    validation_rows, validation_scores = _join(
        _event(
            trajectory_id="validation-a",
            full_episode_id=0,
            task_index=0,
            split="val",
            boundary_tick=60,
            score_by_tick={30: 0.1, 45: 0.4, 60: 0.8},
        ),
        _event(
            trajectory_id="validation-b",
            full_episode_id=1,
            task_index=1,
            split="val",
            boundary_tick=105,
            score_by_tick={75: 0.2, 90: 0.7, 105: 0.6},
        ),
    )

    selection = metrics.select_validation_threshold(validation_rows, validation_scores)

    assert selection.threshold == 0.5
    assert selection.validation_early_trigger_events == 1
    assert selection.validation_event_recall == 0.5
    assert selection.validation_event_f1 == pytest.approx(0.5)
    assert selection.selected_on_split == "val"
    assert "fixed threshold=0.5" in selection.rule


def test_frozen_validation_threshold_reports_operational_test_metrics():
    validation_rows, validation_scores = _event(
        trajectory_id="validation-a",
        full_episode_id=0,
        task_index=0,
        split="val",
        boundary_tick=60,
        score_by_tick={30: 0.1, 45: 0.4, 60: 0.8},
    )
    selection = metrics.select_validation_threshold(validation_rows, validation_scores)
    test_rows, test_scores = _join(
        _event(
            trajectory_id="test-a",
            full_episode_id=10,
            task_index=0,
            split="test",
            boundary_tick=60,
            score_by_tick={30: 0.9, 45: 0.1, 60: 0.95},
        ),
        _event(
            trajectory_id="test-a",
            full_episode_id=10,
            task_index=3,
            split="test",
            boundary_tick=120,
            score_by_tick={90: 0.1, 105: 0.2, 120: 0.85},
        ),
    )

    report = metrics.evaluate_test_with_validation_threshold(test_rows, selection, scores=test_scores)
    applied = report.metrics

    assert applied["threshold/early_trigger_event_rate"] == 0.5
    assert applied["threshold/early_trigger_trajectory_rate"] == 1.0
    assert applied["threshold/event_recall"] == 0.5
    assert applied["threshold/event_f1"] == 0.5
    assert applied["threshold/timing_error_mean_frames"] == -15.0
    assert applied["threshold/timing_error_mean_seconds"] == -0.5
    assert applied["threshold/false_positives_per_minute"] == 30.0
    assert applied["threshold/final_done_recall"] == 1.0
    assert applied["task_3/threshold/event_recall"] == 1.0


def test_test_rows_cannot_be_used_for_threshold_selection():
    test_rows, test_scores = _event(
        trajectory_id="test-a",
        full_episode_id=10,
        task_index=0,
        split="test",
        boundary_tick=60,
        score_by_tick={30: 0.1, 45: 0.2, 60: 0.9},
    )

    with pytest.raises(ValueError, match="only validation"):
        metrics.select_validation_threshold(test_rows, test_scores)
    with pytest.raises(ValueError, match="pure test"):
        metrics.evaluate_test_with_validation_threshold(
            (),
            metrics.ThresholdSelection(0.5, 1, 0, 1.0, 1.0, 2),
            scores=np.asarray([]),
        )


def test_no_negative_event_and_tied_boundary_are_explicit_edge_cases():
    positive_only_rows, positive_only_scores = _event(
        trajectory_id="short",
        full_episode_id=0,
        task_index=0,
        split="val",
        boundary_tick=30,
        score_by_tick={30: 0.9},
    )
    positive_report = metrics.evaluate_temporal_completion(positive_only_rows, scores=positive_only_scores)

    assert positive_report.event_margins[0].margin is None
    assert positive_report.event_margins[0].boundary_top1 is True
    assert math.isnan(positive_report.metrics["margin/all_mean"])
    assert math.isnan(positive_report.metrics["natural/roc_auc"])
    assert positive_report.metrics["hard_local/sample_count"] == 0.0
    assert positive_report.metrics["hard_local/event_count"] == 0.0
    assert math.isnan(positive_report.metrics["hard_local/auprc"])
    selection = metrics.select_validation_threshold(positive_only_rows, positive_only_scores)
    assert selection.threshold == 0.5
    applied = metrics.threshold_application_metrics(positive_only_rows, positive_only_scores, threshold=0.9)
    assert math.isnan(applied["false_positives_per_minute"])

    tied_rows, tied_scores = _event(
        trajectory_id="tie",
        full_episode_id=1,
        task_index=0,
        split="val",
        boundary_tick=45,
        score_by_tick={30: 0.5, 45: 0.5},
    )
    tied_report = metrics.evaluate_temporal_completion(tied_rows, scores=tied_scores)
    assert tied_report.event_margins[0].margin == 0.0
    assert tied_report.event_margins[0].boundary_top1 is False
    tied_selection = metrics.select_validation_threshold(tied_rows, tied_scores)
    assert tied_selection.threshold == 0.5
    assert tied_selection.validation_event_recall == 0.0


def test_hard_local_auprc_excludes_free_positive_without_local_negative():
    short_rows, short_scores = _event(
        trajectory_id="short",
        full_episode_id=0,
        task_index=0,
        split="val",
        boundary_tick=30,
        score_by_tick={30: 0.99},
    )
    paired_rows, paired_scores = _event(
        trajectory_id="paired",
        full_episode_id=1,
        task_index=0,
        split="val",
        boundary_tick=60,
        score_by_tick={30: 0.8, 45: 0.7, 60: 0.6},
    )
    rows, scores = _join((short_rows, short_scores), (paired_rows, paired_scores))

    report = metrics.evaluate_temporal_completion(rows, scores=scores)

    assert report.metrics["natural/positive_count"] == 2.0
    assert report.metrics["hard_local/event_count"] == 1.0
    assert report.metrics["hard_local/positive_count"] == 1.0
    assert report.metrics["hard_local/negative_count"] == 1.0


def test_evaluator_rejects_incomplete_events_duplicates_and_ambiguous_inputs():
    rows, scores = _event(
        trajectory_id="validation-a",
        full_episode_id=0,
        task_index=0,
        split="val",
        boundary_tick=60,
        score_by_tick={30: 0.1, 45: 0.2, 60: 0.9},
    )

    with pytest.raises(ValueError, match="exactly one boundary positive"):
        metrics.event_margins(rows[:-1], scores[:-1])
    with pytest.raises(ValueError, match="duplicate natural sample"):
        metrics.event_margins((*rows, rows[0]), np.append(scores, scores[0]))
    with pytest.raises(ValueError, match="exactly one of logits or scores"):
        metrics.evaluate_temporal_completion(rows)
    with pytest.raises(ValueError, match="exactly one of logits or scores"):
        metrics.evaluate_temporal_completion(rows, logits=np.zeros(3), scores=scores)
    with pytest.raises(ValueError, match="probabilities"):
        metrics.evaluate_temporal_completion(rows, scores=np.asarray([0.1, 0.2, 1.1]))
