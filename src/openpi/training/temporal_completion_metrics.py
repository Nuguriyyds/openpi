"""Pure NumPy metrics for the temporal completion detector.

The evaluator consumes the *natural*, non-resampled candidate rows produced by
``temporal_completion_data``.  An event is one ``(trajectory_id, task_index)``
pair and must contain exactly one positive row at its boundary tick.

Threshold selection is intentionally exposed only through
:func:`select_validation_threshold`, which rejects anything other than a pure
validation split.  The temporal training path does not use threshold metrics
for checkpoint selection; it uses natural AUPRC, hard-only AUPRC, and paired
hard margin.  A fixed 0.5 threshold is retained only for compatibility with
the report/evaluator API.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import dataclasses
import itertools
import math
from typing import Literal

import numpy as np

from openpi.training import temporal_completion_data as _temporal_data

THRESHOLD_SELECTION_RULE = (
    "validation-only fixed threshold=0.5 for reporting; checkpoint selection is threshold-free"
)


@dataclasses.dataclass(frozen=True)
class EventMargin:
    """Boundary separation for one trajectory/task event.

    ``margin`` uses every eligible pre-boundary row. ``hard_local_margin`` uses
    only the fixed preceding 0.5-second hard tick.  A margin is ``None``
    when that event has no eligible negative in the relevant window.
    """

    trajectory_id: str
    task_index: int
    positive_score: float
    negative_count: int
    hard_local_negative_count: int
    margin: float | None
    hard_local_margin: float | None
    boundary_top1: bool


@dataclasses.dataclass(frozen=True)
class ThresholdSelection:
    """A frozen threshold selected exclusively from validation data."""

    threshold: float
    validation_event_count: int
    validation_early_trigger_events: int
    validation_event_recall: float
    validation_event_f1: float
    candidate_count: int
    selected_on_split: Literal["val"] = "val"
    rule: str = THRESHOLD_SELECTION_RULE


@dataclasses.dataclass(frozen=True)
class TemporalCompletionReport:
    """Flat scalar metrics plus inspectable per-event margins."""

    metrics: Mapping[str, float]
    event_margins: tuple[EventMargin, ...]


@dataclasses.dataclass(frozen=True)
class _Event:
    trajectory_id: str
    task_index: int
    positive_index: int
    negative_indices: np.ndarray
    hard_local_negative_indices: np.ndarray


def stable_sigmoid(logits: np.ndarray | Sequence[float]) -> np.ndarray:
    """Computes sigmoid without overflowing for large-magnitude logits."""

    values = np.asarray(logits, dtype=np.float64)
    if np.any(np.isnan(values)):
        raise ValueError("logits must not contain NaN")
    flat = values.reshape(-1)
    result = np.empty_like(flat)
    nonnegative = flat >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-flat[nonnegative]))
    negative_exp = np.exp(flat[~nonnegative])
    result[~nonnegative] = negative_exp / (1.0 + negative_exp)
    return result.reshape(values.shape)


def binary_ranking_metrics(
    labels: np.ndarray | Sequence[int], scores: np.ndarray | Sequence[float]
) -> dict[str, float]:
    """Returns tie-aware average precision (AUPRC) and ROC-AUC.

    AUPRC follows the non-interpolated average-precision convention: precision
    is weighted by each increase in recall.  Metrics that require a missing
    class are returned as ``NaN`` rather than inventing a value.
    """

    binary_labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if binary_labels.shape != score_values.shape or binary_labels.size == 0:
        raise ValueError(
            f"ranking labels/scores must be equally shaped and non-empty, got {binary_labels.shape}/{score_values.shape}"
        )
    if not np.all(np.logical_or(binary_labels == 0, binary_labels == 1)):
        raise ValueError("ranking labels must contain only 0/1")
    _validate_scores(score_values)

    order = np.argsort(-score_values, kind="mergesort")
    sorted_scores = score_values[order]
    sorted_labels = binary_labels[order]
    cumulative_true = np.cumsum(sorted_labels, dtype=np.int64)
    cumulative_false = np.cumsum(1 - sorted_labels, dtype=np.int64)
    group_ends = np.concatenate((sorted_scores[1:] != sorted_scores[:-1], np.asarray([True])))
    true_at_threshold = cumulative_true[group_ends].astype(np.float64)
    false_at_threshold = cumulative_false[group_ends].astype(np.float64)

    positive_count = int(np.sum(binary_labels))
    negative_count = int(binary_labels.size - positive_count)
    if positive_count:
        recall = true_at_threshold / positive_count
        precision = true_at_threshold / (true_at_threshold + false_at_threshold)
        recall_increment = np.diff(np.concatenate((np.asarray([0.0]), recall)))
        auprc = float(np.sum(recall_increment * precision))
    else:
        auprc = math.nan

    if positive_count and negative_count:
        true_positive_rate = np.concatenate((np.asarray([0.0]), true_at_threshold / positive_count))
        false_positive_rate = np.concatenate((np.asarray([0.0]), false_at_threshold / negative_count))
        widths = np.diff(false_positive_rate)
        heights = (true_positive_rate[:-1] + true_positive_rate[1:]) / 2.0
        roc_auc = float(np.sum(widths * heights))
    else:
        roc_auc = math.nan

    return {
        "sample_count": float(binary_labels.size),
        "positive_count": float(positive_count),
        "negative_count": float(negative_count),
        "positive_rate": float(positive_count / binary_labels.size),
        "auprc": auprc,
        "roc_auc": roc_auc,
    }


def event_margins(
    rows: Sequence[_temporal_data.TemporalSampleRow], scores: np.ndarray | Sequence[float]
) -> tuple[EventMargin, ...]:
    """Computes all-history and fixed E-15 hard margins for every event."""

    row_tuple, score_values, events = _prepare_rows_and_scores(rows, scores)
    del row_tuple
    margins: list[EventMargin] = []
    for event in events:
        positive_score = float(score_values[event.positive_index])
        if event.negative_indices.size:
            margin: float | None = positive_score - float(np.max(score_values[event.negative_indices]))
            boundary_top1 = margin > 0.0
        else:
            margin = None
            # The boundary is the only candidate and is therefore top-1; this
            # event is excluded from numerical margin summaries below.
            boundary_top1 = True
        hard_local_margin = (
            positive_score - float(np.max(score_values[event.hard_local_negative_indices]))
            if event.hard_local_negative_indices.size
            else None
        )
        margins.append(
            EventMargin(
                trajectory_id=event.trajectory_id,
                task_index=event.task_index,
                positive_score=positive_score,
                negative_count=int(event.negative_indices.size),
                hard_local_negative_count=int(event.hard_local_negative_indices.size),
                margin=margin,
                hard_local_margin=hard_local_margin,
                boundary_top1=boundary_top1,
            )
        )
    return tuple(margins)


def threshold_application_metrics(
    rows: Sequence[_temporal_data.TemporalSampleRow],
    scores: np.ndarray | Sequence[float],
    *,
    threshold: float,
) -> dict[str, float]:
    """Applies a frozen threshold to natural 2 Hz rows.

    The first crossing in an event is the operational trigger.  Canonical rows
    stop at the supervised boundary, so a trigger is either early or on-time;
    a boundary miss has no synthetic late-recovery row.  Signed timing error is
    ``first_crossing_tick - boundary_tick`` and excludes missed events.

    ``false_positives_per_minute`` is tick-level: every pre-boundary 2 Hz row
    above threshold divided by negative exposure time (negative_rows / 120
    minutes).  ``false_trigger_events_per_minute`` separately reports latched
    first-crossing events over the same exposure.
    """

    row_tuple, score_values, events = _prepare_rows_and_scores(rows, scores)
    threshold_value = _validate_threshold(threshold)
    return _threshold_metrics_from_prepared(row_tuple, score_values, events, threshold_value)


def select_validation_threshold(
    rows: Sequence[_temporal_data.TemporalSampleRow],
    scores: np.ndarray | Sequence[float],
) -> ThresholdSelection:
    """Returns the fixed reporting threshold after validating a pure val split."""

    row_tuple, score_values, events = _prepare_rows_and_scores(rows, scores)
    splits = {row.split for row in row_tuple}
    if splits != {"val"}:
        raise ValueError(f"threshold selection requires only validation rows; got splits={sorted(splits)}")

    fixed_threshold = 0.5
    report_metrics = _threshold_metrics_from_prepared(row_tuple, score_values, events, fixed_threshold)
    return ThresholdSelection(
        threshold=fixed_threshold,
        validation_event_count=len(events),
        validation_early_trigger_events=int(report_metrics["early_trigger_event_count"]),
        validation_event_recall=report_metrics["event_recall"],
        validation_event_f1=report_metrics["event_f1"],
        candidate_count=1,
    )


def evaluate_temporal_completion(
    rows: Sequence[_temporal_data.TemporalSampleRow],
    *,
    logits: np.ndarray | Sequence[float] | None = None,
    scores: np.ndarray | Sequence[float] | None = None,
    threshold: float | ThresholdSelection | None = None,
) -> TemporalCompletionReport:
    """Evaluates one natural split without ever searching that split's threshold.

    Pass exactly one of ``logits`` or ``scores``.  A test call should pass a
    :class:`ThresholdSelection` returned by :func:`select_validation_threshold`;
    this function performs no implicit threshold optimisation.
    """

    score_values = _scores_from_logits_or_scores(logits=logits, scores=scores)
    row_tuple, score_values, _ = _prepare_rows_and_scores(rows, score_values)
    split_names = {row.split for row in row_tuple}
    if len(split_names) != 1:
        raise ValueError(f"evaluation rows must belong to one split, got {sorted(split_names)}")
    if isinstance(threshold, ThresholdSelection) and threshold.selected_on_split != "val":
        raise ValueError("a frozen ThresholdSelection must originate from validation")
    threshold_value = threshold.threshold if isinstance(threshold, ThresholdSelection) else threshold

    metrics: dict[str, float] = {}
    margins = event_margins(row_tuple, score_values)
    _add_stratified_metrics(metrics, row_tuple, score_values, margins, prefix="")
    for task_index in range(_temporal_data.TASKS_PER_TRAJECTORY):
        indices = [index for index, row in enumerate(row_tuple) if row.task_index == task_index]
        if indices:
            task_rows = tuple(row_tuple[index] for index in indices)
            task_scores = score_values[indices]
            task_margins = tuple(margin for margin in margins if margin.task_index == task_index)
            _add_stratified_metrics(
                metrics,
                task_rows,
                task_scores,
                task_margins,
                prefix=f"task_{task_index}/",
            )
        else:
            _add_empty_stratum(metrics, prefix=f"task_{task_index}/")

    if threshold_value is not None:
        threshold_value = _validate_threshold(threshold_value)
        _merge_prefixed(
            metrics,
            threshold_application_metrics(row_tuple, score_values, threshold=threshold_value),
            prefix="threshold/",
        )
        for task_index in range(_temporal_data.TASKS_PER_TRAJECTORY):
            indices = [index for index, row in enumerate(row_tuple) if row.task_index == task_index]
            if not indices:
                continue
            task_rows = tuple(row_tuple[index] for index in indices)
            task_scores = score_values[indices]
            _merge_prefixed(
                metrics,
                threshold_application_metrics(task_rows, task_scores, threshold=threshold_value),
                prefix=f"task_{task_index}/threshold/",
            )
    return TemporalCompletionReport(metrics=metrics, event_margins=margins)


def evaluate_test_with_validation_threshold(
    rows: Sequence[_temporal_data.TemporalSampleRow],
    selection: ThresholdSelection,
    *,
    logits: np.ndarray | Sequence[float] | None = None,
    scores: np.ndarray | Sequence[float] | None = None,
) -> TemporalCompletionReport:
    """Fail-closed test entry point that requires an already-frozen val threshold."""

    if not rows or {row.split for row in rows} != {"test"}:
        raise ValueError("test evaluation requires a non-empty pure test split")
    if selection.selected_on_split != "val":
        raise ValueError("test evaluation threshold must originate from validation")
    return evaluate_temporal_completion(rows, logits=logits, scores=scores, threshold=selection)


def _scores_from_logits_or_scores(
    *,
    logits: np.ndarray | Sequence[float] | None,
    scores: np.ndarray | Sequence[float] | None,
) -> np.ndarray:
    if (logits is None) == (scores is None):
        raise ValueError("pass exactly one of logits or scores")
    if logits is not None:
        return stable_sigmoid(logits).reshape(-1)
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    _validate_scores(score_values)
    return score_values


def _validate_scores(scores: np.ndarray) -> None:
    if np.any(~np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("scores must be finite probabilities in [0, 1]")


def _validate_threshold(threshold: float) -> float:
    threshold_value = float(threshold)
    # The selector can return nextafter(1, +inf) as an explicit abstention
    # sentinel when a sigmoid score rounded to exactly one.
    maximum = float(np.nextafter(1.0, np.inf))
    if not math.isfinite(threshold_value) or threshold_value < 0.0 or threshold_value > maximum:
        raise ValueError(f"threshold must be in [0, nextafter(1,+inf)], got {threshold_value}")
    return threshold_value


def _prepare_rows_and_scores(
    rows: Sequence[_temporal_data.TemporalSampleRow], scores: np.ndarray | Sequence[float]
) -> tuple[tuple[_temporal_data.TemporalSampleRow, ...], np.ndarray, tuple[_Event, ...]]:
    row_tuple = tuple(rows)
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not row_tuple or len(row_tuple) != score_values.size:
        raise ValueError(f"rows/scores must be equally sized and non-empty, got {len(row_tuple)}/{score_values.size}")
    _validate_scores(score_values)

    grouped_indices: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    seen_ticks: set[tuple[str, int, int]] = set()
    for index, row in enumerate(row_tuple):
        tick_key = (row.trajectory_id, row.task_index, row.logical_tick)
        if tick_key in seen_ticks:
            raise ValueError(f"duplicate natural sample row {tick_key}")
        seen_ticks.add(tick_key)
        grouped_indices[(row.trajectory_id, row.task_index)].append(index)

    events: list[_Event] = []
    for (trajectory_id, task_index), indices in sorted(grouped_indices.items()):
        event_rows = [row_tuple[index] for index in indices]
        if len({row.split for row in event_rows}) != 1:
            raise ValueError(f"event {(trajectory_id, task_index)} crosses dataset splits")
        if len({row.full_episode_id for row in event_rows}) != 1:
            raise ValueError(f"event {(trajectory_id, task_index)} crosses full episodes")
        positive_indices = [index for index in indices if row_tuple[index].label == 1]
        if len(positive_indices) != 1:
            raise ValueError(
                f"event {(trajectory_id, task_index)} must contain exactly one boundary positive, got {len(positive_indices)}"
            )
        positive_index = positive_indices[0]
        boundary_tick = row_tuple[positive_index].boundary_tick
        negative_indices = np.asarray(
            sorted(
                (index for index in indices if row_tuple[index].label == 0),
                key=lambda index: row_tuple[index].logical_tick,
            ),
            dtype=np.int64,
        )
        if any(row_tuple[index].logical_tick >= boundary_tick for index in negative_indices):
            raise ValueError(f"event {(trajectory_id, task_index)} contains a non-pre-boundary negative")
        hard_local_indices = np.asarray(
            [index for index in negative_indices if row_tuple[index].sample_kind == "hard_negative"],
            dtype=np.int64,
        )
        events.append(
            _Event(
                trajectory_id=trajectory_id,
                task_index=task_index,
                positive_index=positive_index,
                negative_indices=negative_indices,
                hard_local_negative_indices=hard_local_indices,
            )
        )
    return row_tuple, score_values, tuple(events)


def _threshold_metrics_from_prepared(
    rows: tuple[_temporal_data.TemporalSampleRow, ...],
    scores: np.ndarray,
    events: tuple[_Event, ...],
    threshold: float,
) -> dict[str, float]:
    early_count = 0
    on_time_count = 0
    missed_count = 0
    triggered_count = 0
    early_trajectories: set[str] = set()
    timing_errors_frames: list[float] = []
    task3_on_time = 0
    task3_count = 0

    for event in events:
        positive_row = rows[event.positive_index]
        if event.task_index == _temporal_data.TASKS_PER_TRAJECTORY - 1:
            task3_count += 1
        crossing_negative_indices = [index for index in event.negative_indices if scores[index] >= threshold]
        if crossing_negative_indices:
            first_index = min(crossing_negative_indices, key=lambda index: rows[index].logical_tick)
            early_count += 1
            triggered_count += 1
            early_trajectories.add(event.trajectory_id)
            timing_errors_frames.append(float(rows[first_index].logical_tick - positive_row.boundary_tick))
        elif scores[event.positive_index] >= threshold:
            on_time_count += 1
            triggered_count += 1
            timing_errors_frames.append(0.0)
            if event.task_index == _temporal_data.TASKS_PER_TRAJECTORY - 1:
                task3_on_time += 1
        else:
            missed_count += 1

    event_count = len(events)
    event_precision = on_time_count / triggered_count if triggered_count else 0.0
    event_recall = on_time_count / event_count
    event_f1 = (
        2.0 * event_precision * event_recall / (event_precision + event_recall)
        if event_precision + event_recall
        else 0.0
    )
    trajectory_count = len({event.trajectory_id for event in events})
    negative_indices = np.asarray([index for index, row in enumerate(rows) if row.label == 0], dtype=np.int64)
    false_positive_tick_count = int(np.sum(scores[negative_indices] >= threshold)) if negative_indices.size else 0
    negative_exposure_minutes = negative_indices.size / (2.0 * 60.0)
    if negative_exposure_minutes:
        false_positives_per_minute = false_positive_tick_count / negative_exposure_minutes
        false_triggers_per_minute = early_count / negative_exposure_minutes
    else:
        false_positives_per_minute = math.nan
        false_triggers_per_minute = math.nan
    if timing_errors_frames:
        timing_array = np.asarray(timing_errors_frames, dtype=np.float64)
        timing_mean_frames = float(np.mean(timing_array))
        timing_absolute_mean_frames = float(np.mean(np.abs(timing_array)))
    else:
        timing_mean_frames = math.nan
        timing_absolute_mean_frames = math.nan

    return {
        "threshold": float(threshold),
        "event_count": float(event_count),
        "triggered_event_count": float(triggered_count),
        "early_trigger_event_count": float(early_count),
        "missed_event_count": float(missed_count),
        "early_trigger_event_rate": float(early_count / event_count),
        "early_trigger_trajectory_rate": float(len(early_trajectories) / trajectory_count),
        "event_precision": float(event_precision),
        "event_recall": float(event_recall),
        "event_f1": float(event_f1),
        "timing_error_mean_frames": timing_mean_frames,
        "timing_error_mean_seconds": timing_mean_frames / _temporal_data.FPS,
        "timing_absolute_error_mean_frames": timing_absolute_mean_frames,
        "timing_absolute_error_mean_seconds": timing_absolute_mean_frames / _temporal_data.FPS,
        "timing_error_event_count": float(len(timing_errors_frames)),
        "negative_exposure_minutes": float(negative_exposure_minutes),
        "false_positive_tick_count": float(false_positive_tick_count),
        "false_positives_per_minute": float(false_positives_per_minute),
        "false_trigger_events_per_minute": float(false_triggers_per_minute),
        "final_done_event_count": float(task3_count),
        "final_done_recall": float(task3_on_time / task3_count) if task3_count else math.nan,
    }


def _add_stratified_metrics(
    output: dict[str, float],
    rows: tuple[_temporal_data.TemporalSampleRow, ...],
    scores: np.ndarray,
    margins: tuple[EventMargin, ...],
    *,
    prefix: str,
) -> None:
    labels = np.asarray([row.label for row in rows], dtype=np.int64)
    _merge_prefixed(output, binary_ranking_metrics(labels, scores), prefix=f"{prefix}natural/")
    hard_local_events = {
        (margin.trajectory_id, margin.task_index) for margin in margins if margin.hard_local_negative_count > 0
    }
    hard_local_mask = np.asarray(
        [
            (row.trajectory_id, row.task_index) in hard_local_events
            and row.sample_kind in ("positive", "hard_negative")
            for row in rows
        ]
    )
    if np.any(hard_local_mask):
        _merge_prefixed(
            output,
            binary_ranking_metrics(labels[hard_local_mask], scores[hard_local_mask]),
            prefix=f"{prefix}hard_local/",
        )
    else:
        output.update(
            {
                f"{prefix}hard_local/sample_count": 0.0,
                f"{prefix}hard_local/positive_count": 0.0,
                f"{prefix}hard_local/negative_count": 0.0,
                f"{prefix}hard_local/positive_rate": math.nan,
                f"{prefix}hard_local/auprc": math.nan,
                f"{prefix}hard_local/roc_auc": math.nan,
            }
        )
    output[f"{prefix}hard_local/event_count"] = float(len(hard_local_events))
    output[f"{prefix}event_count"] = float(len(margins))
    output[f"{prefix}boundary_top1_rate"] = float(np.mean([margin.boundary_top1 for margin in margins]))
    _add_margin_summary(output, margins, attribute="margin", prefix=f"{prefix}margin/all_")
    _add_margin_summary(output, margins, attribute="hard_local_margin", prefix=f"{prefix}margin/hard_local_")
    hard_margins = np.asarray(
        [margin.hard_local_margin for margin in margins if margin.hard_local_margin is not None], dtype=np.float64
    )
    output[f"{prefix}paired_hard/ordering_accuracy"] = (
        float(np.mean(hard_margins > 0.0)) if hard_margins.size else math.nan
    )
    output[f"{prefix}paired_hard/margin_mean"] = float(np.mean(hard_margins)) if hard_margins.size else math.nan
    output[f"{prefix}paired_hard/margin_median"] = (
        float(np.median(hard_margins)) if hard_margins.size else math.nan
    )
    output[f"{prefix}paired_hard/margin_p25"] = (
        float(np.percentile(hard_margins, 25)) if hard_margins.size else math.nan
    )
    output[f"{prefix}paired_hard/margin_p75"] = (
        float(np.percentile(hard_margins, 75)) if hard_margins.size else math.nan
    )
    ordinary_scores = scores[np.asarray([row.sample_kind == "ordinary_negative" for row in rows])]
    output[f"{prefix}ordinary/score_mean"] = float(np.mean(ordinary_scores)) if ordinary_scores.size else math.nan
    output[f"{prefix}ordinary/score_p95"] = (
        float(np.percentile(ordinary_scores, 95)) if ordinary_scores.size else math.nan
    )
    _add_negative_smoothness(output, rows, scores, prefix=f"{prefix}negative_smoothness/")


def _add_empty_stratum(output: dict[str, float], *, prefix: str) -> None:
    output[f"{prefix}natural/sample_count"] = 0.0
    output[f"{prefix}natural/positive_count"] = 0.0
    output[f"{prefix}natural/negative_count"] = 0.0
    output[f"{prefix}natural/positive_rate"] = math.nan
    output[f"{prefix}natural/auprc"] = math.nan
    output[f"{prefix}natural/roc_auc"] = math.nan
    output[f"{prefix}hard_local/sample_count"] = 0.0
    output[f"{prefix}hard_local/positive_count"] = 0.0
    output[f"{prefix}hard_local/negative_count"] = 0.0
    output[f"{prefix}hard_local/positive_rate"] = math.nan
    output[f"{prefix}hard_local/auprc"] = math.nan
    output[f"{prefix}hard_local/roc_auc"] = math.nan
    output[f"{prefix}hard_local/event_count"] = 0.0
    output[f"{prefix}event_count"] = 0.0
    output[f"{prefix}boundary_top1_rate"] = math.nan
    for margin_prefix in ("margin/all_", "margin/hard_local_"):
        output[f"{prefix}{margin_prefix}count"] = 0.0
        output[f"{prefix}{margin_prefix}mean"] = math.nan
        output[f"{prefix}{margin_prefix}median"] = math.nan
        output[f"{prefix}{margin_prefix}p25"] = math.nan
        output[f"{prefix}{margin_prefix}p75"] = math.nan
        output[f"{prefix}{margin_prefix}gt_zero_rate"] = math.nan
    for name in ("ordering_accuracy", "margin_mean", "margin_median", "margin_p25", "margin_p75"):
        output[f"{prefix}paired_hard/{name}"] = math.nan
    output[f"{prefix}negative_smoothness/pair_count"] = 0.0
    output[f"{prefix}negative_smoothness/adjacent_abs_delta_mean"] = math.nan
    output[f"{prefix}negative_smoothness/adjacent_abs_delta_p95"] = math.nan
    output[f"{prefix}ordinary/score_mean"] = math.nan
    output[f"{prefix}ordinary/score_p95"] = math.nan


def _add_negative_smoothness(
    output: dict[str, float],
    rows: tuple[_temporal_data.TemporalSampleRow, ...],
    scores: np.ndarray,
    *,
    prefix: str,
) -> None:
    """Summarises score jitter between exact adjacent 2 Hz negative ticks."""

    grouped: defaultdict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.label == 0:
            grouped[(row.trajectory_id, row.task_index)].append(index)
    deltas: list[float] = []
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda index: rows[index].logical_tick)
        for left, right in itertools.pairwise(ordered):
            if rows[right].logical_tick - rows[left].logical_tick == _temporal_data.TICK_STRIDE_FRAMES:
                deltas.append(abs(float(scores[right]) - float(scores[left])))
    values = np.asarray(deltas, dtype=np.float64)
    output[f"{prefix}pair_count"] = float(values.size)
    output[f"{prefix}adjacent_abs_delta_mean"] = float(np.mean(values)) if values.size else math.nan
    output[f"{prefix}adjacent_abs_delta_p95"] = float(np.percentile(values, 95)) if values.size else math.nan


def _add_margin_summary(
    output: dict[str, float],
    margins: tuple[EventMargin, ...],
    *,
    attribute: Literal["margin", "hard_local_margin"],
    prefix: str,
) -> None:
    values = np.asarray(
        [value for margin in margins if (value := getattr(margin, attribute)) is not None],
        dtype=np.float64,
    )
    output[f"{prefix}count"] = float(values.size)
    output[f"{prefix}mean"] = float(np.mean(values)) if values.size else math.nan
    output[f"{prefix}median"] = float(np.median(values)) if values.size else math.nan
    output[f"{prefix}p25"] = float(np.percentile(values, 25)) if values.size else math.nan
    output[f"{prefix}p75"] = float(np.percentile(values, 75)) if values.size else math.nan
    output[f"{prefix}gt_zero_rate"] = float(np.mean(values > 0.0)) if values.size else math.nan


def _merge_prefixed(output: dict[str, float], values: Mapping[str, float], *, prefix: str) -> None:
    output.update({f"{prefix}{key}": value for key, value in values.items()})
