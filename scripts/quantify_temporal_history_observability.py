"""Measure whether a complete subtask prefix history adds completion signal.

This is a CPU-only analysis of the already extracted frozen prefix cache.  It
reconstructs each subtask-local 2 Hz prefix sequence from the cache's source
frame references, then compares linear logistic probes over current, recent,
full-history, elapsed-time, and current-plus-elapsed representations.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from openpi.training import temporal_completion_data as temporal_data
from openpi.training import temporal_completion_features as temporal_features

try:
    # The script directory is on sys.path when this file is run directly.
    from quantify_clean_completion_features import _average_precision, _sigmoid
except ModuleNotFoundError:  # pragma: no cover - useful for module-style execution.
    from scripts.quantify_clean_completion_features import _average_precision, _sigmoid


DEFAULT_MANIFEST = Path(
    "/mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v4.json"
)
DEFAULT_FEATURE_CACHE = Path(
    "/mnt/data/models/wyt/evaluations/temporal_completion_prefix_features_v5/features.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/data/models/wyt/evaluations/temporal_completion_history_observability_v1/seed42"
)
DEFAULT_SEED = 42
DEFAULT_ALIAS_MIN_SECONDS = 2.0
DEFAULT_ELAPSED_MATCH_TOLERANCE_SECONDS = 0.5
DEFAULT_L2_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
REPRESENTATIONS = ("current", "recent3", "full_history", "elapsed", "current_elapsed")
SPLITS = ("train", "val", "test")
TASKS = (0, 1, 2, 3)


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One canonical candidate, after the common history-length filter."""

    episode_key: tuple[str, int, int, int]
    split: str
    task_id: int
    trajectory_id: str
    full_episode_id: int
    subtask_id: int
    frame: int
    label: int
    sample_kind: str
    sequence_index: int

    @property
    def uid(self) -> tuple[str, int, int, int, int]:
        return (*self.episode_key, self.frame)


@dataclasses.dataclass(frozen=True)
class RawCandidate:
    frame: int
    label: int
    sample_kind: str
    boundary_frame: int


@dataclasses.dataclass
class EpisodeData:
    """All deduplicated prefix points and canonical rows for one subtask."""

    key: tuple[str, int, int, int]
    split: str
    task_id: int
    trajectory_id: str
    full_episode_id: int
    subtask_id: int
    end_frame: int
    raw_candidates: list[RawCandidate] = dataclasses.field(default_factory=list)
    points: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    candidates: list[Candidate] = dataclasses.field(default_factory=list)
    frame_to_index: dict[int, int] = dataclasses.field(default_factory=dict)
    frames: tuple[int, ...] = ()
    sequence_features: np.ndarray | None = None
    history_too_short_count: int = 0

    @property
    def start_frame(self) -> int:
        if not self.frames:
            raise ValueError(f"episode {self.key} has no deduplicated prefix points")
        return self.frames[0]

    @property
    def candidates_by_frame(self) -> dict[int, Candidate]:
        return {candidate.frame: candidate for candidate in self.candidates}


@dataclasses.dataclass(frozen=True)
class PairSpec:
    positive: Candidate
    adjacent: Candidate | None
    alias: Candidate | None
    alias_cosine: float | None


def _episode_key_from_row(row: temporal_data.TemporalSampleRow) -> tuple[str, int, int, int]:
    return (
        row.trajectory_id,
        int(row.full_episode_id),
        int(row.task_index),
        int(row.subtask_episode_id),
    )


def _episode_key_from_record(
    record: temporal_data.ManifestTrajectoryRecord,
    task_id: int,
) -> tuple[str, int, int, int]:
    full_episode_id = record.full_episode_id
    if full_episode_id is None:
        if record.group_id is None:
            raise ValueError(f"manifest trajectory {record.trajectory_id} has no episode identity")
        full_episode_id = record.group_id
    return (
        record.trajectory_id,
        int(full_episode_id),
        int(task_id),
        int(record.source_episode_ids[task_id]),
    )


def _make_manifest_episodes(manifest: temporal_data.TemporalCompletionManifest) -> dict[
    tuple[str, int, int, int], EpisodeData
]:
    episodes: dict[tuple[str, int, int, int], EpisodeData] = {}
    for record in manifest.trajectories:
        if record.split not in SPLITS:
            continue
        group = record.as_group()
        full_episode_id = record.full_episode_id
        if full_episode_id is None:
            if record.group_id is None:
                raise ValueError(f"manifest trajectory {record.trajectory_id} has no full episode id")
            full_episode_id = record.group_id
        for task_id in TASKS:
            key = _episode_key_from_record(record, task_id)
            if key in episodes:
                raise ValueError(f"manifest repeats subtask episode key {key}")
            episodes[key] = EpisodeData(
                key=key,
                split=str(record.split),
                task_id=task_id,
                trajectory_id=record.trajectory_id,
                full_episode_id=int(full_episode_id),
                subtask_id=int(group.source_episode_ids[task_id]),
                end_frame=int(group.lengths[task_id] - 1),
            )
    return episodes


def _build_episode_sequences(
    manifest: temporal_data.TemporalCompletionManifest,
    cache: temporal_features.TemporalFeatureCache,
) -> tuple[dict[tuple[str, int, int, int], EpisodeData], dict[str, int]]:
    """Deduplicates source-frame features and builds common valid candidates."""

    episodes = _make_manifest_episodes(manifest)
    feature_dim = int(cache.metadata.feature_dim)
    raw_candidate_counts = {split: 0 for split in SPLITS}

    for row_index, (row, history_value) in enumerate(zip(cache.rows, cache.prefix_history, strict=True)):
        if row.sample_kind == "transition_negative":
            raise ValueError("subtask-local observability analysis cannot consume transition-negative rows")
        key = _episode_key_from_row(row)
        episode = episodes.get(key)
        if episode is None:
            raise ValueError(f"cache row {row_index} refers to unknown manifest subtask episode {key}")
        if row.split != episode.split or row.boundary_tick != episode.end_frame:
            raise ValueError(f"cache row {row_index} split/boundary disagrees with manifest for {key}")
        history = np.asarray(history_value, dtype=np.float32)
        if history.shape != (temporal_data.TEMPORAL_HISTORY_STEPS, feature_dim):
            raise ValueError(
                f"cache row {row_index} has history shape {history.shape}, "
                f"expected {(temporal_data.TEMPORAL_HISTORY_STEPS, feature_dim)}"
            )
        if row.logical_tick != row.source_frame_indices[-1]:
            raise ValueError(f"cache row {row_index} current frame metadata is inconsistent")
        if row.subtask_episode_id != episode.subtask_id:
            raise ValueError(f"cache row {row_index} subtask episode id disagrees with its key")
        episode.raw_candidates.append(
            RawCandidate(
                frame=int(row.logical_tick),
                label=int(row.label),
                sample_kind=row.sample_kind,
                boundary_frame=int(row.boundary_tick),
            )
        )
        raw_candidate_counts[row.split] += 1

        for source_episode_id, frame, feature in zip(
            row.source_episode_ids,
            row.source_frame_indices,
            history,
            strict=True,
        ):
            if source_episode_id != episode.subtask_id:
                raise ValueError(f"cache row {row_index} crosses subtask episodes")
            frame = int(frame)
            if frame > episode.end_frame:
                raise ValueError(f"cache row {row_index} contains a feature after its endpoint")
            if frame not in episode.points:
                episode.points[frame] = np.asarray(feature, dtype=np.float32).copy()

    short_candidate_counts = {split: 0 for split in SPLITS}
    for episode in episodes.values():
        if not episode.points:
            continue
        frames = tuple(sorted(episode.points))
        sequence_features = np.stack([episode.points[frame] for frame in frames], axis=0).astype(
            np.float32, copy=False
        )
        if sequence_features.shape != (len(frames), feature_dim):
            raise ValueError(f"deduplicated sequence for {episode.key} has unexpected shape {sequence_features.shape}")
        episode.frames = frames
        episode.sequence_features = sequence_features
        episode.frame_to_index = {frame: index for index, frame in enumerate(frames)}

        seen_candidate_frames: set[int] = set()
        for raw in episode.raw_candidates:
            if raw.frame in seen_candidate_frames:
                raise ValueError(f"manifest/cache repeats canonical candidate frame {episode.key}:{raw.frame}")
            seen_candidate_frames.add(raw.frame)
            sequence_index = episode.frame_to_index.get(raw.frame)
            if sequence_index is None:
                raise ValueError(f"candidate {episode.key}:{raw.frame} has no deduplicated current feature")
            if sequence_index + 1 < temporal_data.TEMPORAL_HISTORY_STEPS:
                episode.history_too_short_count += 1
                short_candidate_counts[episode.split] += 1
                continue
            episode.candidates.append(
                Candidate(
                    episode_key=episode.key,
                    split=episode.split,
                    task_id=episode.task_id,
                    trajectory_id=episode.trajectory_id,
                    full_episode_id=episode.full_episode_id,
                    subtask_id=episode.subtask_id,
                    frame=raw.frame,
                    label=raw.label,
                    sample_kind=raw.sample_kind,
                    sequence_index=sequence_index,
                )
            )
        episode.candidates.sort(key=lambda candidate: candidate.frame)

    return episodes, {
        "raw_candidate_count": int(sum(raw_candidate_counts.values())),
        "history_too_short_candidate_count": int(sum(short_candidate_counts.values())),
        "raw_candidate_count_by_split": raw_candidate_counts,
        "history_too_short_candidate_count_by_split": short_candidate_counts,
    }


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _stable_candidate_key(candidate: Candidate) -> tuple[int, int, int, str]:
    return (candidate.full_episode_id, candidate.subtask_id, candidate.frame, candidate.trajectory_id)


def _build_within_episode_pairs(
    episodes: list[EpisodeData],
    *,
    alias_min_frames: int,
) -> tuple[dict[tuple[str, int, int, int], PairSpec], dict[str, int]]:
    pairs: dict[tuple[str, int, int, int], PairSpec] = {}
    counts = {
        "episodes_missing_alias_negative": 0,
        "episodes_without_positive": 0,
        "episodes_without_adjacent": 0,
    }
    for episode in episodes:
        by_kind = {candidate.sample_kind: candidate for candidate in episode.candidates}
        positive = by_kind.get("positive")
        if positive is None:
            counts["episodes_without_positive"] += 1
            continue
        adjacent = by_kind.get("hard_negative")
        if adjacent is None:
            counts["episodes_without_adjacent"] += 1

        ordinary = [candidate for candidate in episode.candidates if candidate.sample_kind == "ordinary_negative"]
        eligible = [candidate for candidate in ordinary if candidate.frame <= episode.end_frame - alias_min_frames]
        alias: Candidate | None = None
        alias_cosine: float | None = None
        if eligible:
            endpoint_feature = _candidate_current_feature(episode, positive)
            best_candidate: Candidate | None = None
            best_key: tuple[float, int] | None = None
            for candidate in eligible:
                similarity = _cosine_similarity(endpoint_feature, _candidate_current_feature(episode, candidate))
                distance = episode.end_frame - candidate.frame
                key = (similarity, -distance)
                if best_key is None or key > best_key:
                    best_key = key
                    best_candidate = candidate
                    alias_cosine = similarity
            alias = best_candidate
        else:
            counts["episodes_missing_alias_negative"] += 1
        pairs[episode.key] = PairSpec(
            positive=positive,
            adjacent=adjacent,
            alias=alias,
            alias_cosine=alias_cosine,
        )
    return pairs, counts


def _build_elapsed_matched_cross_episode_pairs(
    episodes: list[EpisodeData],
    *,
    alias_min_frames: int,
    fps: float,
    tolerance_seconds: float,
) -> tuple[dict[tuple[str, int, int, int], PairSpec], dict[str, int]]:
    """Greedily assigns one elapsed-matched ordinary negative per positive."""

    pairs: dict[tuple[str, int, int, int], PairSpec] = {}
    counts = {
        "episodes_missing_alias_negative": 0,
        "episodes_without_positive": 0,
        "episodes_without_adjacent": 0,
        "elapsed_matched_candidates_rejected_as_used": 0,
    }
    episode_by_key = {episode.key: episode for episode in episodes}
    by_split_task: dict[tuple[str, int], list[EpisodeData]] = {}
    for episode in episodes:
        by_split_task.setdefault((episode.split, episode.task_id), []).append(episode)

    for split_task, grouped_episodes in sorted(by_split_task.items()):
        split, task_id = split_task
        positive_episodes: list[tuple[EpisodeData, Candidate, Candidate | None]] = []
        negative_pool = [
            candidate
            for episode in grouped_episodes
            for candidate in episode.candidates
            if candidate.sample_kind == "ordinary_negative"
            and episode.end_frame - candidate.frame >= alias_min_frames
        ]
        for episode in grouped_episodes:
            by_kind = {candidate.sample_kind: candidate for candidate in episode.candidates}
            positive = by_kind.get("positive")
            if positive is None:
                counts["episodes_without_positive"] += 1
                continue
            adjacent = by_kind.get("hard_negative")
            if adjacent is None:
                counts["episodes_without_adjacent"] += 1
            positive_episodes.append((episode, positive, adjacent))

        candidates_by_positive: dict[tuple[str, int, int, int], list[Candidate]] = {}
        for episode, positive, _adjacent in positive_episodes:
            positive_elapsed = (positive.frame - episode.start_frame) / fps
            feasible = [
                candidate
                for candidate in negative_pool
                if candidate.episode_key != episode.key
                and abs(
                    (candidate.frame - episode_by_key[candidate.episode_key].start_frame) / fps
                    - positive_elapsed
                )
                <= tolerance_seconds
            ]
            candidates_by_positive[episode.key] = feasible

        # The candidate-count ordering is fixed before any negative is consumed.
        matching_order = sorted(
            positive_episodes,
            key=lambda item: (
                len(candidates_by_positive[item[0].key]),
                item[0].full_episode_id,
                item[0].subtask_id,
                item[0].trajectory_id,
            ),
        )
        used_negative_uids: set[tuple[str, int, int, int, int]] = set()
        for episode, positive, adjacent in matching_order:
            feasible = candidates_by_positive[episode.key]
            if not feasible:
                counts["episodes_missing_alias_negative"] += 1
                pairs[episode.key] = PairSpec(
                    positive=positive,
                    adjacent=adjacent,
                    alias=None,
                    alias_cosine=None,
                )
                continue
            available = [candidate for candidate in feasible if candidate.uid not in used_negative_uids]
            if not available:
                counts["episodes_missing_alias_negative"] += 1
                counts["elapsed_matched_candidates_rejected_as_used"] += 1
                pairs[episode.key] = PairSpec(
                    positive=positive,
                    adjacent=adjacent,
                    alias=None,
                    alias_cosine=None,
                )
                continue

            positive_feature = _candidate_current_feature(episode, positive)
            positive_elapsed = (positive.frame - episode.start_frame) / fps
            differences = {
                candidate.uid: abs(
                    (candidate.frame - episode_by_key[candidate.episode_key].start_frame) / fps
                    - positive_elapsed
                )
                for candidate in available
            }
            minimum_difference = min(differences.values())
            near_time = [
                candidate
                for candidate in available
                if differences[candidate.uid] <= minimum_difference + 1.0 / fps
            ]
            negative = sorted(
                near_time,
                key=lambda candidate: (
                    -_cosine_similarity(
                        positive_feature,
                        _candidate_current_feature(episode_by_key[candidate.episode_key], candidate),
                    ),
                    *_stable_candidate_key(candidate),
                ),
            )[0]
            used_negative_uids.add(negative.uid)
            negative_episode = episode_by_key[negative.episode_key]
            pairs[episode.key] = PairSpec(
                positive=positive,
                adjacent=adjacent,
                alias=negative,
                alias_cosine=_cosine_similarity(
                    positive_feature,
                    _candidate_current_feature(negative_episode, negative),
                ),
            )

    return pairs, counts


def _build_pairs(
    episodes: Iterable[EpisodeData],
    *,
    alias_mode: str,
    alias_min_frames: int,
    fps: float,
    tolerance_seconds: float,
) -> tuple[dict[tuple[str, int, int, int], PairSpec], dict[str, int]]:
    episode_list = list(episodes)
    if alias_mode == "within_episode":
        return _build_within_episode_pairs(episode_list, alias_min_frames=alias_min_frames)
    if alias_mode == "elapsed_matched_cross_episode":
        return _build_elapsed_matched_cross_episode_pairs(
            episode_list,
            alias_min_frames=alias_min_frames,
            fps=fps,
            tolerance_seconds=tolerance_seconds,
        )
    raise ValueError(f"unknown alias mode {alias_mode!r}")


def _candidate_current_feature(episode: EpisodeData, candidate: Candidate) -> np.ndarray:
    if episode.sequence_features is None:
        raise ValueError(f"episode {episode.key} has no sequence features")
    return episode.sequence_features[candidate.sequence_index]


def _candidate_vector(episode: EpisodeData, candidate: Candidate, representation: str, fps: float) -> np.ndarray:
    if episode.sequence_features is None:
        raise ValueError(f"episode {episode.key} has no sequence features")
    sequence = episode.sequence_features
    index = candidate.sequence_index
    current = sequence[index]
    if representation == "current":
        return current.astype(np.float32, copy=False)
    if representation == "recent3":
        return sequence[index - 2 : index + 1].reshape(-1).astype(np.float32, copy=False)
    if representation == "full_history":
        history = sequence[: index + 1]
        thirds = np.array_split(history, 3, axis=0)
        if any(len(third) == 0 for third in thirds):
            raise ValueError(f"candidate {candidate.uid} has fewer than three history points")
        return np.concatenate([third.mean(axis=0, dtype=np.float64) for third in thirds] + [current], axis=0).astype(
            np.float32
        )
    elapsed = np.asarray([(candidate.frame - episode.start_frame) / fps], dtype=np.float32)
    if representation == "elapsed":
        return elapsed
    if representation == "current_elapsed":
        return np.concatenate([current, elapsed], axis=0).astype(np.float32, copy=False)
    raise ValueError(f"unknown representation {representation!r}")


def _candidate_sort_key(candidate: Candidate) -> tuple[int, str, int]:
    return (candidate.task_id, candidate.trajectory_id, candidate.frame)


def _episodes_for_split_task(
    episodes: Iterable[EpisodeData],
    split: str,
    task_id: int,
) -> list[EpisodeData]:
    return sorted(
        [episode for episode in episodes if episode.split == split and episode.task_id == task_id],
        key=lambda episode: (episode.trajectory_id, episode.subtask_id),
    )


def _natural_candidates(episodes: Iterable[EpisodeData], split: str) -> list[Candidate]:
    candidates = [candidate for episode in episodes if episode.split == split for candidate in episode.candidates]
    return sorted(candidates, key=_candidate_sort_key)


def _probe_samples(
    episodes: Iterable[EpisodeData],
    pairs: dict[tuple[str, int, int, int], PairSpec],
    split: str,
) -> list[Candidate]:
    samples: list[Candidate] = []
    for episode in sorted(
        [item for item in episodes if item.split == split],
        key=lambda item: (item.task_id, item.trajectory_id, item.subtask_id),
    ):
        pair = pairs.get(episode.key)
        if pair is None:
            continue
        samples.append(pair.positive)
        if pair.adjacent is not None:
            samples.append(pair.adjacent)
        if pair.alias is not None:
            samples.append(pair.alias)
    if len({candidate.uid for candidate in samples}) != len(samples):
        raise ValueError(f"probe sample pool for {split} contains duplicate candidates")
    return samples


def _task_balanced_weights(samples: list[Candidate]) -> np.ndarray:
    if not samples:
        raise ValueError("cannot weight an empty probe sample pool")
    counts = {task_id: sum(candidate.task_id == task_id for candidate in samples) for task_id in TASKS}
    active = [task_id for task_id in TASKS if counts[task_id] > 0]
    if not active:
        raise ValueError("probe sample pool has no task")
    weights = np.asarray([1.0 / counts[candidate.task_id] for candidate in samples], dtype=np.float64)
    weights *= len(samples) / float(np.sum(weights))
    if not np.isfinite(weights).all() or not np.isclose(float(np.mean(weights)), 1.0):
        raise ValueError("task-balanced sample weights are not finite or mean-normalized")
    return weights


def _standardize_train(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = np.mean(train_x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(train_x, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-5, 1.0, std)
    return tuple(((values.astype(np.float32) - mean) / std).astype(np.float32) for values in (train_x, *others))


def _fit_weighted_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    regularization: float,
    max_iter: int,
) -> tuple[np.ndarray, float]:
    from scipy import optimize  # noqa: PLC0415

    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if train_x.ndim != 2 or len(train_x) != len(train_y) or len(train_y) != len(sample_weight):
        raise ValueError("weighted logistic inputs have inconsistent shapes")
    denominator = float(np.sum(sample_weight))
    parameters = np.zeros(train_x.shape[1] + 1, dtype=np.float64)

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        weights = values[:-1]
        bias = values[-1]
        logits = train_x @ weights + bias
        residual = _sigmoid(logits) - train_y
        data_loss = np.sum(sample_weight * (np.logaddexp(0.0, logits) - train_y * logits)) / denominator
        loss = data_loss + 0.5 * regularization * np.sum(weights**2)
        gradient_w = train_x.T @ (sample_weight * residual) / denominator + regularization * weights
        gradient_b = float(np.sum(sample_weight * residual) / denominator)
        return float(loss), np.concatenate([gradient_w, [gradient_b]])

    fitted = optimize.minimize(
        objective,
        parameters,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": 1e-9},
    )
    if not np.isfinite(fitted.x).all():
        raise ValueError(f"logistic probe optimization returned non-finite parameters: {fitted.message}")
    return fitted.x[:-1].astype(np.float64, copy=True), float(fitted.x[-1])


def _matrix(
    candidates: list[Candidate],
    episodes_by_key: dict[tuple[str, int, int, int], EpisodeData],
    representation: str,
    fps: float,
) -> tuple[np.ndarray, dict[tuple[str, int, int, int, int], int]]:
    if not candidates:
        raise ValueError(f"no candidates available for representation {representation}")
    values = [
        _candidate_vector(episodes_by_key[candidate.episode_key], candidate, representation, fps)
        for candidate in candidates
    ]
    matrix = np.stack(values, axis=0).astype(np.float32, copy=False)
    return matrix, {candidate.uid: index for index, candidate in enumerate(candidates)}


def _score_lookup(
    candidates: list[Candidate],
    logits: np.ndarray,
) -> dict[tuple[str, int, int, int, int], tuple[float, float]]:
    probabilities = _sigmoid(logits)
    return {
        candidate.uid: (float(logit), float(probability))
        for candidate, logit, probability in zip(candidates, logits, probabilities, strict=True)
    }


def _macro(values: list[float]) -> float | None:
    if not values or not all(np.isfinite(value) for value in values):
        return None
    return float(np.mean(values))


def _pair_metrics(
    episodes: Iterable[EpisodeData],
    pairs: dict[tuple[str, int, int, int], PairSpec],
    scores: dict[tuple[str, int, int, int, int], tuple[float, float]],
    *,
    split: str,
    pair_type: str,
) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    for task_id in TASKS:
        correct: list[float] = []
        margins: list[float] = []
        selected_episodes = _episodes_for_split_task(episodes, split, task_id)
        for episode in selected_episodes:
            pair = pairs.get(episode.key)
            if pair is None:
                continue
            negative = pair.alias if pair_type == "alias" else pair.adjacent
            if negative is None:
                continue
            positive_score = scores[pair.positive.uid]
            negative_score = scores[negative.uid]
            correct.append(1.0 if positive_score[0] > negative_score[0] else 0.0)
            margins.append(positive_score[1] - negative_score[1])
        by_task[str(task_id)] = {
            "pair_count": len(correct),
            "accuracy": float(np.mean(correct)) if correct else None,
            "mean_probability_margin": float(np.mean(margins)) if margins else None,
            "median_probability_margin": float(np.median(margins)) if margins else None,
        }
    accuracies = [item["accuracy"] for item in by_task.values() if item["accuracy"] is not None]
    margins = [item["mean_probability_margin"] for item in by_task.values() if item["mean_probability_margin"] is not None]
    return {
        "by_task": by_task,
        "macro": {
            "pair_count": int(sum(item["pair_count"] for item in by_task.values())),
            "accuracy": _macro(accuracies),
            "mean_probability_margin": _macro(margins),
            "median_probability_margin": _macro(
                [item["median_probability_margin"] for item in by_task.values() if item["median_probability_margin"] is not None]
            ),
        },
    }


def _natural_metrics(
    episodes: Iterable[EpisodeData],
    split: str,
    scores: dict[tuple[str, int, int, int, int], tuple[float, float]],
) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    for task_id in TASKS:
        candidates = [candidate for candidate in _natural_candidates(episodes, split) if candidate.task_id == task_id]
        target = np.asarray([candidate.label for candidate in candidates], dtype=np.int8)
        score = np.asarray([scores[candidate.uid][1] for candidate in candidates], dtype=np.float64)
        value = _average_precision(target, score) if len(candidates) else float("nan")
        by_task[str(task_id)] = {
            "candidate_count": len(candidates),
            "positive_count": int(np.sum(target)),
            "negative_count": int(len(target) - np.sum(target)),
            "auprc": float(value) if np.isfinite(value) else None,
        }
    values = [item["auprc"] for item in by_task.values() if item["auprc"] is not None]
    return {"by_task": by_task, "macro_auprc": _macro(values)}


def _fit_and_score(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    eval_x: np.ndarray,
    *,
    regularization: float,
    max_iter: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    weights, bias = _fit_weighted_logistic(
        train_x,
        train_y,
        train_weight,
        regularization=regularization,
        max_iter=max_iter,
    )
    return weights, bias, eval_x @ weights + bias


def _evaluate_with_logits(
    episodes: Iterable[EpisodeData],
    pairs: dict[tuple[str, int, int, int], PairSpec],
    split: str,
    candidates: list[Candidate],
    logits: np.ndarray,
) -> tuple[dict[str, Any], dict[tuple[str, int, int, int, int], tuple[float, float]]]:
    split_episodes = [episode for episode in episodes if episode.split == split]
    scores = _score_lookup(candidates, logits)
    alias = _pair_metrics(split_episodes, pairs, scores, split=split, pair_type="alias")
    adjacent = _pair_metrics(split_episodes, pairs, scores, split=split, pair_type="adjacent")
    natural = _natural_metrics(split_episodes, split, scores)
    return {
        "alias_pair_ordering": alias,
        "adjacent_pair_ordering": adjacent,
        "natural_candidate_auprc": natural,
    }, scores


def _selection_score(metrics: dict[str, Any]) -> float | None:
    alias = metrics["alias_pair_ordering"]["macro"]["accuracy"]
    adjacent = metrics["adjacent_pair_ordering"]["macro"]["accuracy"]
    if alias is None or adjacent is None:
        return None
    return 0.5 * (float(alias) + float(adjacent))


def _fit_representation(
    episodes: dict[tuple[str, int, int, int], EpisodeData],
    pairs: dict[tuple[str, int, int, int], PairSpec],
    representation: str,
    *,
    fps: float,
    l2_grid: tuple[float, ...],
    max_iter: int,
    alias_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    train_samples = _probe_samples(episodes.values(), pairs, "train")
    val_candidates = _natural_candidates(episodes.values(), "val")
    if not train_samples or not val_candidates:
        raise ValueError(f"representation {representation} lacks train or validation candidates")
    train_x_raw, _ = _matrix(train_samples, episodes, representation, fps)
    val_x_raw, _ = _matrix(val_candidates, episodes, representation, fps)
    train_x, val_x = _standardize_train(train_x_raw, val_x_raw)
    train_y = np.asarray([candidate.label for candidate in train_samples], dtype=np.int8)
    train_weight = _task_balanced_weights(train_samples)
    val_candidates = list(val_candidates)

    best: dict[str, Any] | None = None
    validation_grid: list[dict[str, Any]] = []
    for regularization in l2_grid:
        weights, bias, val_logits = _fit_and_score(
            train_x,
            train_y,
            train_weight,
            val_x,
            regularization=regularization,
            max_iter=max_iter,
        )
        val_metrics, val_scores = _evaluate_with_logits(
            episodes.values(), pairs, "val", val_candidates, val_logits
        )
        score = _selection_score(val_metrics)
        natural_macro = val_metrics["natural_candidate_auprc"]["macro_auprc"]
        alias_accuracy = val_metrics["alias_pair_ordering"]["macro"]["accuracy"]
        adjacent_accuracy = val_metrics["adjacent_pair_ordering"]["macro"]["accuracy"]
        if alias_mode == "within_episode":
            selection_key = (
                None
                if score is None or natural_macro is None
                else (float(score), float(natural_macro), float(regularization))
            )
            selection_priority = "pair_selection_score, natural_candidate_auprc, stronger_lambda"
        else:
            selection_key = (
                None
                if natural_macro is None or alias_accuracy is None or adjacent_accuracy is None
                else (float(natural_macro), float(alias_accuracy), float(adjacent_accuracy), float(regularization))
            )
            selection_priority = "natural_candidate_auprc, matched_alias_accuracy, adjacent_accuracy, stronger_lambda"
        validation_grid.append(
            {
                "lambda": float(regularization),
                "selection_score": score,
                "selection_key": selection_key,
                "macro_alias_pair_accuracy": alias_accuracy,
                "macro_adjacent_pair_accuracy": adjacent_accuracy,
                "macro_natural_candidate_auprc": natural_macro,
            }
        )
        if selection_key is None:
            continue
        if best is None or selection_key > best["selection_key"]:
            best = {
                "selection_key": selection_key,
                "lambda": float(regularization),
                "weights": weights,
                "bias": bias,
                "metrics": val_metrics,
                "scores": val_scores,
            }
    if best is None:
        raise ValueError(f"validation cannot select a regularization value for {representation}")

    selected_lambda = best["lambda"]
    selected_val_metrics = best["metrics"]
    selected_val_scores = best["scores"]

    train_val_samples = _probe_samples(episodes.values(), pairs, "train") + _probe_samples(episodes.values(), pairs, "val")
    train_val_x_raw, _ = _matrix(train_val_samples, episodes, representation, fps)
    test_candidates = _natural_candidates(episodes.values(), "test")
    if not test_candidates:
        raise ValueError("test split has no natural candidates")
    test_x_raw, _ = _matrix(test_candidates, episodes, representation, fps)
    train_val_x, test_x = _standardize_train(train_val_x_raw, test_x_raw)
    train_val_y = np.asarray([candidate.label for candidate in train_val_samples], dtype=np.int8)
    train_val_weight = _task_balanced_weights(train_val_samples)
    final_weights, final_bias, test_logits = _fit_and_score(
        train_val_x,
        train_val_y,
        train_val_weight,
        test_x,
        regularization=selected_lambda,
        max_iter=max_iter,
    )
    test_metrics, test_scores = _evaluate_with_logits(
        episodes.values(), pairs, "test", test_candidates, test_logits
    )

    val_prediction_rows = _natural_prediction_rows(
        episodes,
        "val",
        representation,
        val_candidates,
        selected_val_scores,
        fps=fps,
    )
    test_prediction_rows = _natural_prediction_rows(
        episodes,
        "test",
        representation,
        test_candidates,
        test_scores,
        fps=fps,
    )

    output = {
        "selected_lambda": selected_lambda,
        "validation": {
            "selection_score": _selection_score(selected_val_metrics),
            "selection_priority": selection_priority,
            "selection_key": best["selection_key"],
            "macro_alias_pair_accuracy": selected_val_metrics["alias_pair_ordering"]["macro"]["accuracy"],
            "macro_adjacent_pair_accuracy": selected_val_metrics["adjacent_pair_ordering"]["macro"]["accuracy"],
            "macro_natural_candidate_auprc": selected_val_metrics["natural_candidate_auprc"]["macro_auprc"],
            "metrics": selected_val_metrics,
            "lambda_grid": validation_grid,
        },
        "test": test_metrics,
        "train_pool": {
            "sample_count": len(train_samples),
            "task_sample_counts": {str(task_id): sum(item.task_id == task_id for item in train_samples) for task_id in TASKS},
        },
        "train_val_pool": {
            "sample_count": len(train_val_samples),
            "task_sample_counts": {
                str(task_id): sum(item.task_id == task_id for item in train_val_samples) for task_id in TASKS
            },
        },
    }
    prediction_rows = []
    prediction_rows.extend(
        _pair_prediction_rows(
            episodes,
            pairs,
            "val",
            representation,
            selected_val_scores,
            fps=fps,
            alias_mode=alias_mode,
            episodes_by_key=episodes,
        )
    )
    prediction_rows.extend(
        _pair_prediction_rows(
            episodes,
            pairs,
            "test",
            representation,
            test_scores,
            fps=fps,
            alias_mode=alias_mode,
            episodes_by_key=episodes,
        )
    )
    return output, prediction_rows, val_prediction_rows + test_prediction_rows


def _natural_prediction_rows(
    episodes: dict[tuple[str, int, int, int], EpisodeData],
    split: str,
    representation: str,
    candidates: list[Candidate],
    scores: dict[tuple[str, int, int, int, int], tuple[float, float]],
    *,
    fps: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in TASKS:
        task_candidates = [candidate for candidate in candidates if candidate.task_id == task_id]
        task_candidates.sort(
            key=lambda candidate: (
                -scores[candidate.uid][1],
                *_stable_candidate_key(candidate),
            )
        )
        for rank, candidate in enumerate(task_candidates, start=1):
            episode = episodes[candidate.episode_key]
            logit, probability = scores[candidate.uid]
            rows.append(
                {
                    "split": split,
                    "task_id": task_id,
                    "episode_or_group_id": candidate.trajectory_id,
                    "subtask_id": candidate.subtask_id,
                    "frame": candidate.frame,
                    "end_frame": episode.end_frame,
                    "elapsed_seconds": (candidate.frame - episode.start_frame) / fps,
                    "remaining_seconds": (episode.end_frame - candidate.frame) / fps,
                    "sample_kind": candidate.sample_kind,
                    "label": candidate.label,
                    "representation": representation,
                    "logit": logit,
                    "probability": probability,
                    "rank_within_task": rank,
                }
            )
    return rows


def _pair_prediction_rows(
    episodes: dict[tuple[str, int, int, int], EpisodeData],
    pairs: dict[tuple[str, int, int, int], PairSpec],
    split: str,
    representation: str,
    scores: dict[tuple[str, int, int, int, int], tuple[float, float]],
    *,
    fps: float,
    alias_mode: str,
    episodes_by_key: dict[tuple[str, int, int, int], EpisodeData],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in sorted(
        [item for item in episodes.values() if item.split == split],
        key=lambda item: (item.task_id, item.trajectory_id, item.subtask_id),
    ):
        pair = pairs.get(episode.key)
        if pair is None:
            continue
        alias_pair_type = "elapsed_matched_alias" if alias_mode == "elapsed_matched_cross_episode" else "alias"
        for pair_type, negative in (("adjacent", pair.adjacent), (alias_pair_type, pair.alias)):
            if negative is None:
                continue
            positive_logit, positive_probability = scores[pair.positive.uid]
            negative_logit, negative_probability = scores[negative.uid]
            positive_episode = episodes_by_key[pair.positive.episode_key]
            negative_episode = episodes_by_key[negative.episode_key]
            positive_elapsed = (pair.positive.frame - positive_episode.start_frame) / fps
            negative_elapsed = (negative.frame - negative_episode.start_frame) / fps
            rows.append(
                {
                    "split": split,
                    "task_id": episode.task_id,
                    "episode_or_group_id": episode.trajectory_id,
                    "subtask_id": episode.subtask_id,
                    "pair_type": pair_type,
                    "positive_frame": pair.positive.frame,
                    "negative_frame": negative.frame,
                    "negative_distance_seconds": (pair.positive.frame - negative.frame) / fps,
                    "positive_episode": pair.positive.trajectory_id,
                    "negative_episode": negative.trajectory_id,
                    "positive_elapsed_seconds": positive_elapsed,
                    "negative_elapsed_seconds": negative_elapsed,
                    "absolute_elapsed_difference": abs(positive_elapsed - negative_elapsed),
                    "negative_distance_from_own_endpoint_seconds": (negative_episode.end_frame - negative.frame) / fps,
                    "endpoint_alias_cosine": pair.alias_cosine if pair_type == alias_pair_type else "",
                    "representation": representation,
                    "positive_logit": positive_logit,
                    "negative_logit": negative_logit,
                    "positive_probability": positive_probability,
                    "negative_probability": negative_probability,
                    "probability_margin": positive_probability - negative_probability,
                    "ordering_correct": int(positive_logit > negative_logit),
                }
            )
    return rows


def _sample_counts(
    episodes: Iterable[EpisodeData],
    pairs: dict[tuple[str, int, int, int], PairSpec],
    split: str,
    task_id: int,
) -> dict[str, Any]:
    selected = _episodes_for_split_task(episodes, split, task_id)
    natural = [candidate for episode in selected for candidate in episode.candidates]
    positives = [candidate for candidate in natural if candidate.sample_kind == "positive"]
    adjacent = [candidate for candidate in natural if candidate.sample_kind == "hard_negative"]
    ordinary = [candidate for candidate in natural if candidate.sample_kind == "ordinary_negative"]
    alias_count = sum(1 for episode in selected if pairs.get(episode.key) is not None and pairs[episode.key].alias is not None)
    missing_alias = sum(
        1
        for episode in selected
        if pairs.get(episode.key) is not None
        and pairs[episode.key].positive is not None
        and pairs[episode.key].alias is None
    )
    return {
        "episode_count": len(selected),
        "episodes_with_valid_positive": len(positives),
        "positive_count": len(positives),
        "adjacent_count": len(adjacent),
        "alias_count": alias_count,
        "ordinary_negative_count": len(ordinary),
        "natural_candidate_count": len(natural),
        "raw_candidate_count": sum(len(episode.raw_candidates) for episode in selected),
        "history_too_short_candidate_count": sum(episode.history_too_short_count for episode in selected),
        "episodes_missing_alias_negative": missing_alias,
    }


def _matching_quality(
    episodes: Iterable[EpisodeData],
    pairs: dict[tuple[str, int, int, int], PairSpec],
    *,
    fps: float,
) -> dict[str, Any]:
    """Summarize alias coverage and elapsed/cosine quality per split/task."""

    episode_by_key = {episode.key: episode for episode in episodes}
    result: dict[str, Any] = {}
    for split in SPLITS:
        result[split] = {}
        task_blocks: list[dict[str, Any]] = []
        for task_id in TASKS:
            selected = _episodes_for_split_task(episode_by_key.values(), split, task_id)
            positive_count = sum(1 for episode in selected if episode.key in pairs)
            matched = [
                pairs[episode.key]
                for episode in selected
                if episode.key in pairs and pairs[episode.key].alias is not None
            ]
            elapsed_differences: list[float] = []
            cosines: list[float] = []
            for pair in matched:
                positive_episode = episode_by_key[pair.positive.episode_key]
                negative = pair.alias
                assert negative is not None
                negative_episode = episode_by_key[negative.episode_key]
                positive_elapsed = (pair.positive.frame - positive_episode.start_frame) / fps
                negative_elapsed = (negative.frame - negative_episode.start_frame) / fps
                elapsed_differences.append(abs(positive_elapsed - negative_elapsed))
                if pair.alias_cosine is not None:
                    cosines.append(float(pair.alias_cosine))
            block = {
                "positive_count": positive_count,
                "matched_alias_count": len(matched),
                "missing_match_count": positive_count - len(matched),
                "match_coverage": len(matched) / positive_count if positive_count else None,
                "mean_absolute_elapsed_difference": float(np.mean(elapsed_differences))
                if elapsed_differences
                else None,
                "median_absolute_elapsed_difference": float(np.median(elapsed_differences))
                if elapsed_differences
                else None,
                "max_absolute_elapsed_difference": float(np.max(elapsed_differences))
                if elapsed_differences
                else None,
                "mean_endpoint_negative_cosine": float(np.mean(cosines)) if cosines else None,
                "median_endpoint_negative_cosine": float(np.median(cosines)) if cosines else None,
            }
            result[split][str(task_id)] = block
            task_blocks.append(block)
        result[split]["macro"] = {
            "positive_count": int(sum(block["positive_count"] for block in task_blocks)),
            "matched_alias_count": int(sum(block["matched_alias_count"] for block in task_blocks)),
            "missing_match_count": int(sum(block["missing_match_count"] for block in task_blocks)),
            "match_coverage": _macro(
                [block["match_coverage"] for block in task_blocks if block["match_coverage"] is not None]
            ),
            "mean_absolute_elapsed_difference": _macro(
                [
                    block["mean_absolute_elapsed_difference"]
                    for block in task_blocks
                    if block["mean_absolute_elapsed_difference"] is not None
                ]
            ),
            "median_absolute_elapsed_difference": _macro(
                [
                    block["median_absolute_elapsed_difference"]
                    for block in task_blocks
                    if block["median_absolute_elapsed_difference"] is not None
                ]
            ),
            "max_absolute_elapsed_difference": _macro(
                [
                    block["max_absolute_elapsed_difference"]
                    for block in task_blocks
                    if block["max_absolute_elapsed_difference"] is not None
                ]
            ),
            "mean_endpoint_negative_cosine": _macro(
                [block["mean_endpoint_negative_cosine"] for block in task_blocks if block["mean_endpoint_negative_cosine"] is not None]
            ),
            "median_endpoint_negative_cosine": _macro(
                [
                    block["median_endpoint_negative_cosine"]
                    for block in task_blocks
                    if block["median_endpoint_negative_cosine"] is not None
                ]
            ),
        }
    return result


def _delta_block(left: dict[str, Any], right: dict[str, Any], *, metric_name: str) -> dict[str, Any]:
    def value(block: dict[str, Any], task: str) -> float | None:
        raw = block["by_task"][task][metric_name]
        return None if raw is None else float(raw)

    by_task: dict[str, float | None] = {}
    for task_id in TASKS:
        task = str(task_id)
        left_value = value(left, task)
        right_value = value(right, task)
        by_task[task] = None if left_value is None or right_value is None else left_value - right_value
    left_macro = left["macro"][metric_name]
    right_macro = right["macro"][metric_name]
    macro = None if left_macro is None or right_macro is None else float(left_macro - right_macro)
    return {"by_task": by_task, "macro": macro}


def _write_pair_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "task_id",
        "episode_or_group_id",
        "subtask_id",
        "pair_type",
        "positive_frame",
        "negative_frame",
        "negative_distance_seconds",
        "positive_episode",
        "negative_episode",
        "positive_elapsed_seconds",
        "negative_elapsed_seconds",
        "absolute_elapsed_difference",
        "negative_distance_from_own_endpoint_seconds",
        "endpoint_alias_cosine",
        "representation",
        "positive_logit",
        "negative_logit",
        "positive_probability",
        "negative_probability",
        "probability_margin",
        "ordering_correct",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_natural_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "task_id",
        "episode_or_group_id",
        "subtask_id",
        "frame",
        "end_frame",
        "elapsed_seconds",
        "remaining_seconds",
        "sample_kind",
        "label",
        "representation",
        "logit",
        "probability",
        "rank_within_task",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _false_positive_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return top negative candidates and simple top-10 repair comparisons."""

    top: dict[str, dict[str, list[dict[str, Any]]]] = {}
    top_sets: dict[tuple[str, int], set[tuple[str, int, int]]] = {}
    for representation in REPRESENTATIONS:
        top[representation] = {}
        for task_id in TASKS:
            negatives = [
                row
                for row in rows
                if row["split"] == "test"
                and row["representation"] == representation
                and row["task_id"] == task_id
                and row["label"] == 0
            ]
            negatives.sort(
                key=lambda row: (
                    -float(row["probability"]),
                    str(row["episode_or_group_id"]),
                    int(row["subtask_id"]),
                    int(row["frame"]),
                )
            )
            selected = negatives[:10]
            top[representation][str(task_id)] = [
                {
                    "episode_or_group_id": row["episode_or_group_id"],
                    "subtask_id": row["subtask_id"],
                    "frame": row["frame"],
                    "remaining_seconds": row["remaining_seconds"],
                    "probability": row["probability"],
                    "sample_kind": row["sample_kind"],
                }
                for row in selected
            ]
            top_sets[(representation, task_id)] = {
                (str(row["episode_or_group_id"]), int(row["subtask_id"]), int(row["frame"]))
                for row in selected
            }

    comparisons: dict[str, dict[str, int]] = {}
    for task_id in TASKS:
        current = top_sets[("current", task_id)]
        recent3 = top_sets[("recent3", task_id)]
        full_history = top_sets[("full_history", task_id)]
        comparisons[str(task_id)] = {
            "current_top10_removed_by_recent3": len(current - recent3),
            "recent3_top10_removed_by_full_history": len(recent3 - full_history),
            "full_history_top10_new_vs_recent3": len(full_history - recent3),
        }
    return {"top10_high_scoring_negatives": top, "top10_set_comparisons": comparisons}


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_summary(path: Path, results: dict[str, Any]) -> None:
    representations = results["representations"]
    lines = [
        "# Temporal history observability",
        "",
        f"Conclusion: **{results['automatic_conclusion']}**",
        "",
        "The probe is a CPU linear Logistic Regression over frozen prefix features. "
        "Validation selects L2 regularization; test is evaluated with the selected model fit on train+validation.",
        "",
        "## Test macro metrics",
        "",
        "| representation | lambda | alias accuracy | alias margin | adjacent accuracy | adjacent margin | natural AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in REPRESENTATIONS:
        result = representations[name]
        test = result["test"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _format_value(result["selected_lambda"]),
                    _format_value(test["alias_pair_ordering"]["macro"]["accuracy"]),
                    _format_value(test["alias_pair_ordering"]["macro"]["mean_probability_margin"]),
                    _format_value(test["adjacent_pair_ordering"]["macro"]["accuracy"]),
                    _format_value(test["adjacent_pair_ordering"]["macro"]["mean_probability_margin"]),
                    _format_value(test["natural_candidate_auprc"]["macro_auprc"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Test alias pair accuracy by task", "", "| representation | task0 | task1 | task2 | task3 | macro |", "|---|---:|---:|---:|---:|---:|"])
    for name in REPRESENTATIONS:
        block = representations[name]["test"]["alias_pair_ordering"]
        lines.append("| " + " | ".join([name] + [_format_value(block["by_task"][str(task)]["accuracy"]) for task in TASKS] + [_format_value(block["macro"]["accuracy"])]) + " |")

    lines.extend(["", "## Test adjacent pair accuracy by task", "", "| representation | task0 | task1 | task2 | task3 | macro |", "|---|---:|---:|---:|---:|---:|"])
    for name in REPRESENTATIONS:
        block = representations[name]["test"]["adjacent_pair_ordering"]
        lines.append("| " + " | ".join([name] + [_format_value(block["by_task"][str(task)]["accuracy"]) for task in TASKS] + [_format_value(block["macro"]["accuracy"])]) + " |")

    lines.extend(["", "## Test natural-candidate AUPRC by task", "", "| representation | task0 | task1 | task2 | task3 | macro |", "|---|---:|---:|---:|---:|---:|"])
    for name in REPRESENTATIONS:
        block = representations[name]["test"]["natural_candidate_auprc"]
        lines.append("| " + " | ".join([name] + [_format_value(block["by_task"][str(task)]["auprc"]) for task in TASKS] + [_format_value(block["macro_auprc"])]) + " |")

    delta = results["deltas"]["full_history_vs_recent3"]["alias_pair_accuracy"]
    lines.extend(["", "## Full-history minus Recent-3 alias accuracy", "", "| task0 | task1 | task2 | task3 | macro |", "|---:|---:|---:|---:|---:|"])
    lines.append("| " + " | ".join([_format_value(delta["by_task"][str(task)]) for task in TASKS] + [_format_value(delta["macro"])]) + " |")

    full = representations["full_history"]["test"]
    current_elapsed = representations["current_elapsed"]["test"]
    lines.extend(
        [
            "",
            "## Full-history versus Current + elapsed",
            "",
            "| metric | full-history | current+elapsed | full-history minus current+elapsed |",
            "|---|---:|---:|---:|",
            "| alias pair accuracy | "
            + " | ".join(
                [
                    _format_value(full["alias_pair_ordering"]["macro"]["accuracy"]),
                    _format_value(current_elapsed["alias_pair_ordering"]["macro"]["accuracy"]),
                    _format_value(results["deltas"]["full_history_vs_current_elapsed"]["alias_pair_accuracy"]["macro"]),
                ]
            )
            + " |",
            "| natural AUPRC | "
            + " | ".join(
                [
                    _format_value(full["natural_candidate_auprc"]["macro_auprc"]),
                    _format_value(current_elapsed["natural_candidate_auprc"]["macro_auprc"]),
                    _format_value(results["deltas"]["full_history_vs_current_elapsed"]["natural_candidate_auprc"]["macro"]),
                ]
            )
            + " |",
        ]
    )

    lines.extend(["", "## Sample counts and exclusions", "", "| split | task | episodes | positives | adjacent | alias | natural candidates | short-history candidates | missing-alias episodes |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for split in SPLITS:
        for task_id in TASKS:
            item = results["sample_counts"][split][str(task_id)]
            lines.append(
                "| "
                + " | ".join(
                    [
                        split,
                        str(task_id),
                        str(item["episode_count"]),
                        str(item["positive_count"]),
                        str(item["adjacent_count"]),
                        str(item["alias_count"]),
                        str(item["natural_candidate_count"]),
                        str(item["history_too_short_candidate_count"]),
                        str(item["episodes_missing_alias_negative"]),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Automatic conclusion details",
            "",
            f"- alias_gain (full-history minus Recent-3): {_format_value(results['conclusion_details']['alias_gain'])}",
            f"- tasks with alias improvement: {results['conclusion_details']['tasks_with_alias_improvement']} / 4",
            f"- full-history minus current+elapsed alias accuracy: {_format_value(results['conclusion_details']['full_history_vs_current_elapsed_alias_gap'])}",
            f"- full-history minus Recent-3 natural AUPRC: {_format_value(results['conclusion_details']['full_history_vs_recent3_natural_auprc_delta'])}",
            "",
            results["conclusion_details"]["explanation"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path: Path, results: dict[str, Any]) -> None:
    """Write the v2-oriented report while retaining the v1 core tables."""

    representations = results["representations"]
    alias_label = (
        "elapsed-matched alias"
        if results["alias_mode"] == "elapsed_matched_cross_episode"
        else "within-episode alias"
    )
    lines = [
        "# Temporal history observability",
        "",
        f"Conclusion: **{results['automatic_conclusion']}**",
        "",
        f"Alias mode: `{results['alias_mode']}`; elapsed-match tolerance: "
        f"{_format_value(results['elapsed_match_tolerance_seconds'])} seconds; "
        f"minimum own-episode negative distance: {results['alias_min_seconds']} seconds.",
        "",
        "The probe is a CPU linear Logistic Regression over frozen prefix features. "
        "Natural-candidate AUPRC is the primary validation-selection and test metric in the cross-episode mode.",
        "",
        "## Test macro metrics",
        "",
        f"| representation | lambda | natural AUPRC | {alias_label} accuracy | mean margin | median margin | adjacent accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in REPRESENTATIONS:
        result = representations[name]
        test = result["test"]
        alias = test["alias_pair_ordering"]["macro"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _format_value(result["selected_lambda"]),
                    _format_value(test["natural_candidate_auprc"]["macro_auprc"]),
                    _format_value(alias["accuracy"]),
                    _format_value(alias["mean_probability_margin"]),
                    _format_value(alias["median_probability_margin"]),
                    _format_value(test["adjacent_pair_ordering"]["macro"]["accuracy"]),
                ]
            )
            + " |"
        )

    lines.extend([
        "",
        "## Natural-candidate AUPRC by task (test)",
        "",
        "| representation | task0 | task1 | task2 | task3 | macro |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in REPRESENTATIONS:
        block = representations[name]["test"]["natural_candidate_auprc"]
        lines.append(
            "| "
            + " | ".join(
                [name]
                + [_format_value(block["by_task"][str(task)]["auprc"]) for task in TASKS]
                + [_format_value(block["macro_auprc"])]
            )
            + " |"
        )

    natural_deltas = results["deltas"]["natural_candidate_auprc"]
    lines.extend([
        "",
        "## Natural-candidate AUPRC deltas (test)",
        "",
        "| comparison | task0 | task1 | task2 | task3 | macro |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for comparison in ("recent3_minus_current", "full_history_minus_recent3", "full_history_minus_current"):
        block = natural_deltas[comparison]
        lines.append(
            "| "
            + " | ".join(
                [comparison]
                + [_format_value(block["by_task"][str(task)]) for task in TASKS]
                + [_format_value(block["macro"])]
            )
            + " |"
        )

    lines.extend([
        "",
        f"## {alias_label.title()} ordering accuracy (test)",
        "",
        "| representation | task0 | task1 | task2 | task3 | macro |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in REPRESENTATIONS:
        block = representations[name]["test"]["alias_pair_ordering"]
        lines.append(
            "| "
            + " | ".join(
                [name]
                + [_format_value(block["by_task"][str(task)]["accuracy"]) for task in TASKS]
                + [_format_value(block["macro"]["accuracy"])]
            )
            + " |"
        )

    lines.extend([
        "",
        "## Adjacent ordering accuracy (test)",
        "",
        "| representation | task0 | task1 | task2 | task3 | macro |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in REPRESENTATIONS:
        block = representations[name]["test"]["adjacent_pair_ordering"]
        lines.append(
            "| "
            + " | ".join(
                [name]
                + [_format_value(block["by_task"][str(task)]["accuracy"]) for task in TASKS]
                + [_format_value(block["macro"]["accuracy"])]
            )
            + " |"
        )

    matching = results["matching_quality"]
    lines.extend([
        "",
        "## Matching quality",
        "",
        "| split | task | positives | matched | missing | coverage | mean | median | max elapsed diff (s) | mean cosine | median cosine |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for split in SPLITS:
        for task_id in TASKS:
            block = matching[split][str(task_id)]
            lines.append(
                "| "
                + " | ".join(
                    [
                        split,
                        str(task_id),
                        str(block["positive_count"]),
                        str(block["matched_alias_count"]),
                        str(block["missing_match_count"]),
                        _format_value(block["match_coverage"]),
                        _format_value(block["mean_absolute_elapsed_difference"]),
                        _format_value(block["median_absolute_elapsed_difference"]),
                        _format_value(block["max_absolute_elapsed_difference"]),
                        _format_value(block["mean_endpoint_negative_cosine"]),
                        _format_value(block["median_endpoint_negative_cosine"]),
                    ]
                )
                + " |"
            )

    elapsed_only = representations["elapsed"]["test"]["alias_pair_ordering"]["macro"]
    lines.extend([
        "",
        "## Elapsed-only leakage check",
        "",
        f"Elapsed-only {alias_label} accuracy: **{_format_value(elapsed_only['accuracy'])}**; "
        f"mean probability margin: **{_format_value(elapsed_only['mean_probability_margin'])}**; "
        f"pair count: **{elapsed_only['pair_count']}**.",
        "Values clearly above 0.6 indicate that elapsed matching still leaves a time confound; matched-pair claims are then treated cautiously.",
    ])

    fp = results["false_positive_analysis"]
    lines.extend([
        "",
        "## Top-10 highest-scoring natural negatives (test)",
        "",
        "| representation | task | episode | subtask | frame | remaining seconds | probability |",
        "|---|---:|---|---:|---:|---:|---:|",
    ])
    for representation in REPRESENTATIONS:
        for task_id in TASKS:
            for row in fp["top10_high_scoring_negatives"][representation][str(task_id)]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            representation,
                            str(task_id),
                            str(row["episode_or_group_id"]),
                            str(row["subtask_id"]),
                            str(row["frame"]),
                            _format_value(row["remaining_seconds"]),
                            _format_value(row["probability"]),
                        ]
                    )
                    + " |"
                )

    lines.extend([
        "",
        "## Top-10 false-positive set comparisons",
        "",
        "| task | current top-10 removed by recent3 | recent3 top-10 removed by full-history | full-history top-10 new vs recent3 |",
        "|---:|---:|---:|---:|",
    ])
    for task_id in TASKS:
        block = fp["top10_set_comparisons"][str(task_id)]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(task_id),
                    str(block["current_top10_removed_by_recent3"]),
                    str(block["recent3_top10_removed_by_full_history"]),
                    str(block["full_history_top10_new_vs_recent3"]),
                ]
            )
            + " |"
        )

    lines.extend([
        "",
        "## Sample counts",
        "",
        "| split | task | episodes | positives | adjacent | alias | natural candidates | short-history | missing alias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for split in SPLITS:
        for task_id in TASKS:
            item = results["sample_counts"][split][str(task_id)]
            lines.append(
                "| "
                + " | ".join(
                    [
                        split,
                        str(task_id),
                        str(item["episode_count"]),
                        str(item["positive_count"]),
                        str(item["adjacent_count"]),
                        str(item["alias_count"]),
                        str(item["natural_candidate_count"]),
                        str(item["history_too_short_candidate_count"]),
                        str(item["episodes_missing_alias_negative"]),
                    ]
                )
                + " |"
            )

    lines.extend([
        "",
        "## Conclusion details",
        "",
        results["conclusion_details"]["explanation"],
        f"- full-history minus Recent-3 natural AUPRC: {_format_value(results['conclusion_details']['full_history_minus_recent3_natural_auprc'])}",
        f"- full-history minus Recent-3 {alias_label} accuracy: {_format_value(results['conclusion_details']['full_history_minus_recent3_alias_accuracy'])}",
        f"- tasks with positive natural-AUPRC delta: {results['conclusion_details']['tasks_with_positive_natural_delta']} / 4",
        f"- tasks with positive {alias_label} accuracy delta: {results['conclusion_details']['tasks_with_positive_alias_delta']} / 4",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--feature-cache", type=Path, default=DEFAULT_FEATURE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--alias-mode",
        choices=("within_episode", "elapsed_matched_cross_episode"),
        default="within_episode",
    )
    parser.add_argument("--alias-min-seconds", type=float, default=DEFAULT_ALIAS_MIN_SECONDS)
    parser.add_argument(
        "--elapsed-match-tolerance-seconds",
        type=float,
        default=DEFAULT_ELAPSED_MATCH_TOLERANCE_SECONDS,
    )
    parser.add_argument("--max-iter", type=int, default=120)
    args = parser.parse_args()
    if args.alias_min_seconds <= 0.0:
        parser.error("--alias-min-seconds must be positive")
    if args.elapsed_match_tolerance_seconds <= 0.0:
        parser.error("--elapsed-match-tolerance-seconds must be positive")
    if args.max_iter <= 0:
        parser.error("--max-iter must be positive")
    return args


def _run(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(
        (args.output_dir / filename).exists() for filename in ("results.json", "summary.md", "pair_predictions.csv")
    ):
        raise FileExistsError(f"refusing to overwrite existing observability outputs: {args.output_dir}")
    manifest = temporal_data.load_temporal_manifest(args.manifest)
    cache = temporal_features.load_temporal_feature_cache(
        args.feature_cache,
        manifest=manifest,
        sampling_protocol="subtask_local",
    )
    fps = float(manifest.fps)
    alias_min_frames = int(np.ceil(float(args.alias_min_seconds) * fps))
    if alias_min_frames <= 0:
        raise ValueError("alias minimum frame distance must be positive")
    episodes, reconstruction_counts = _build_episode_sequences(manifest, cache)
    pairs, pair_counts = _build_pairs(episodes.values(), alias_min_frames=alias_min_frames)
    sample_counts = {
        split: {
            str(task_id): _sample_counts(episodes.values(), pairs, split, task_id)
            for task_id in TASKS
        }
        for split in SPLITS
    }

    representations: dict[str, Any] = {}
    pair_prediction_rows: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        print(f"Fitting {representation} probe on CPU")
        result, prediction_rows = _fit_representation(
            episodes,
            pairs,
            representation,
            fps=fps,
            l2_grid=DEFAULT_L2_GRID,
            max_iter=args.max_iter,
        )
        representations[representation] = result
        pair_prediction_rows.extend(prediction_rows)

    full_test = representations["full_history"]["test"]
    recent_test = representations["recent3"]["test"]
    current_test = representations["current"]["test"]
    current_elapsed_test = representations["current_elapsed"]["test"]
    full_alias = float(full_test["alias_pair_ordering"]["macro"]["accuracy"])
    recent_alias = float(recent_test["alias_pair_ordering"]["macro"]["accuracy"])
    current_elapsed_alias = float(current_elapsed_test["alias_pair_ordering"]["macro"]["accuracy"])
    alias_gain = full_alias - recent_alias
    task_alias_delta = {
        str(task_id): float(
            full_test["alias_pair_ordering"]["by_task"][str(task_id)]["accuracy"]
            - recent_test["alias_pair_ordering"]["by_task"][str(task_id)]["accuracy"]
        )
        for task_id in TASKS
    }
    tasks_with_alias_improvement = sum(delta > 0.0 for delta in task_alias_delta.values())
    full_recent_natural_delta = float(
        full_test["natural_candidate_auprc"]["macro_auprc"]
        - recent_test["natural_candidate_auprc"]["macro_auprc"]
    )
    full_elapsed_gap = full_alias - current_elapsed_alias
    if (
        alias_gain >= 0.05
        and tasks_with_alias_improvement >= 3
        and full_recent_natural_delta < -0.02
    ):
        conclusion = "HISTORY_GAIN_WITH_AUPRC_REGRESSION"
        explanation = "Alias ordering improved substantially, but natural-candidate AUPRC regressed by more than 0.02 versus Recent-3."
    elif alias_gain >= 0.05 and tasks_with_alias_improvement >= 3 and full_elapsed_gap < 0.02:
        conclusion = "TIME_CONFOUNDED"
        explanation = "Full-history improved over Recent-3, but its alias accuracy is within 0.02 of the current-plus-elapsed baseline."
    elif (
        alias_gain >= 0.05
        and tasks_with_alias_improvement >= 3
        and full_elapsed_gap >= 0.02
        and full_recent_natural_delta >= -0.02
    ):
        conclusion = "PASS_HISTORY_SIGNAL"
        explanation = "Full-history improves alias ordering across at least three tasks, beats current-plus-elapsed by at least 0.02, and does not materially regress natural AUPRC."
    else:
        conclusion = "NO_CLEAR_HISTORY_GAIN"
        explanation = "Full-history did not satisfy all required alias-gain, per-task improvement, time-baseline, and natural-AUPRC conditions."

    deltas = {
        "full_history_vs_recent3": {
            "alias_pair_accuracy": _delta_block(
                full_test["alias_pair_ordering"], recent_test["alias_pair_ordering"], metric_name="accuracy"
            ),
            "alias_mean_probability_margin": _delta_block(
                full_test["alias_pair_ordering"], recent_test["alias_pair_ordering"], metric_name="mean_probability_margin"
            ),
            "adjacent_pair_accuracy": _delta_block(
                full_test["adjacent_pair_ordering"], recent_test["adjacent_pair_ordering"], metric_name="accuracy"
            ),
            "natural_candidate_auprc": {
                "by_task": {
                    str(task_id): float(
                        full_test["natural_candidate_auprc"]["by_task"][str(task_id)]["auprc"]
                        - recent_test["natural_candidate_auprc"]["by_task"][str(task_id)]["auprc"]
                    )
                    for task_id in TASKS
                },
                "macro": full_recent_natural_delta,
            },
        },
        "full_history_vs_current_elapsed": {
            "alias_pair_accuracy": _delta_block(
                full_test["alias_pair_ordering"], current_elapsed_test["alias_pair_ordering"], metric_name="accuracy"
            ),
            "natural_candidate_auprc": {
                "by_task": {
                    str(task_id): float(
                        full_test["natural_candidate_auprc"]["by_task"][str(task_id)]["auprc"]
                        - current_elapsed_test["natural_candidate_auprc"]["by_task"][str(task_id)]["auprc"]
                    )
                    for task_id in TASKS
                },
                "macro": float(
                    full_test["natural_candidate_auprc"]["macro_auprc"]
                    - current_elapsed_test["natural_candidate_auprc"]["macro_auprc"]
                ),
            },
        },
        "current_elapsed_vs_current": {
            "alias_pair_accuracy": _delta_block(
                current_elapsed_test["alias_pair_ordering"], current_test["alias_pair_ordering"], metric_name="accuracy"
            ),
            "natural_candidate_auprc": {
                "by_task": {
                    str(task_id): float(
                        current_elapsed_test["natural_candidate_auprc"]["by_task"][str(task_id)]["auprc"]
                        - current_test["natural_candidate_auprc"]["by_task"][str(task_id)]["auprc"]
                    )
                    for task_id in TASKS
                },
                "macro": float(
                    current_elapsed_test["natural_candidate_auprc"]["macro_auprc"]
                    - current_test["natural_candidate_auprc"]["macro_auprc"]
                ),
            },
        },
    }

    results: dict[str, Any] = {
        "experiment": "temporal_history_observability",
        "manifest": str(args.manifest),
        "feature_cache": str(args.feature_cache),
        "seed": int(args.seed),
        "alias_min_seconds": float(args.alias_min_seconds),
        "alias_min_frame_distance": alias_min_frames,
        "fps": fps,
        "tick_stride_frames": int(manifest.tick_stride_frames),
        "cache_metadata": dataclasses.asdict(cache.metadata),
        "reconstruction_counts": reconstruction_counts,
        "pair_counts": pair_counts,
        "sample_counts": sample_counts,
        "representations": representations,
        "deltas": deltas,
        "automatic_conclusion": conclusion,
        "conclusion_details": {
            "alias_gain": alias_gain,
            "tasks_with_alias_improvement": tasks_with_alias_improvement,
            "full_history_vs_current_elapsed_alias_gap": full_elapsed_gap,
            "full_history_vs_recent3_natural_auprc_delta": full_recent_natural_delta,
            "task_alias_accuracy_delta": task_alias_delta,
            "explanation": explanation,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_pair_predictions(args.output_dir / "pair_predictions.csv", pair_prediction_rows)
    _write_summary(args.output_dir / "summary.md", results)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved results to {args.output_dir}")
    print(f"Automatic conclusion: {conclusion}")


def _natural_delta_block(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    by_task = {
        str(task_id): float(
            left["by_task"][str(task_id)]["auprc"] - right["by_task"][str(task_id)]["auprc"]
        )
        for task_id in TASKS
    }
    return {
        "by_task": by_task,
        "macro": float(left["macro_auprc"] - right["macro_auprc"]),
    }


def _run(args: argparse.Namespace) -> None:
    output_files = ("results.json", "summary.md", "pair_predictions.csv", "natural_predictions.csv")
    if args.output_dir.exists() and any((args.output_dir / filename).exists() for filename in output_files):
        raise FileExistsError(f"refusing to overwrite existing observability outputs: {args.output_dir}")
    manifest = temporal_data.load_temporal_manifest(args.manifest)
    cache = temporal_features.load_temporal_feature_cache(
        args.feature_cache,
        manifest=manifest,
        sampling_protocol="subtask_local",
    )
    fps = float(manifest.fps)
    alias_min_frames = int(np.ceil(float(args.alias_min_seconds) * fps))
    if alias_min_frames <= 0:
        raise ValueError("alias minimum frame distance must be positive")
    episodes, reconstruction_counts = _build_episode_sequences(manifest, cache)
    pairs, pair_counts = _build_pairs(
        episodes.values(),
        alias_mode=args.alias_mode,
        alias_min_frames=alias_min_frames,
        fps=fps,
        tolerance_seconds=float(args.elapsed_match_tolerance_seconds),
    )
    sample_counts = {
        split: {
            str(task_id): _sample_counts(episodes.values(), pairs, split, task_id)
            for task_id in TASKS
        }
        for split in SPLITS
    }
    matching_quality = _matching_quality(episodes.values(), pairs, fps=fps)

    representations: dict[str, Any] = {}
    pair_prediction_rows: list[dict[str, Any]] = []
    natural_prediction_rows: list[dict[str, Any]] = []
    for representation in REPRESENTATIONS:
        print(f"Fitting {representation} probe on CPU")
        result, pair_rows, natural_rows = _fit_representation(
            episodes,
            pairs,
            representation,
            fps=fps,
            l2_grid=DEFAULT_L2_GRID,
            max_iter=args.max_iter,
            alias_mode=args.alias_mode,
        )
        representations[representation] = result
        pair_prediction_rows.extend(pair_rows)
        natural_prediction_rows.extend(natural_rows)

    test = {name: representations[name]["test"] for name in REPRESENTATIONS}
    natural_deltas = {
        "recent3_minus_current": _natural_delta_block(
            test["recent3"]["natural_candidate_auprc"], test["current"]["natural_candidate_auprc"]
        ),
        "full_history_minus_recent3": _natural_delta_block(
            test["full_history"]["natural_candidate_auprc"], test["recent3"]["natural_candidate_auprc"]
        ),
        "full_history_minus_current": _natural_delta_block(
            test["full_history"]["natural_candidate_auprc"], test["current"]["natural_candidate_auprc"]
        ),
    }
    alias_deltas = {
        "recent3_minus_current": _delta_block(
            test["recent3"]["alias_pair_ordering"], test["current"]["alias_pair_ordering"], metric_name="accuracy"
        ),
        "full_history_minus_recent3": _delta_block(
            test["full_history"]["alias_pair_ordering"], test["recent3"]["alias_pair_ordering"], metric_name="accuracy"
        ),
        "full_history_minus_current": _delta_block(
            test["full_history"]["alias_pair_ordering"], test["current"]["alias_pair_ordering"], metric_name="accuracy"
        ),
    }
    full_natural_delta = natural_deltas["full_history_minus_recent3"]["macro"]
    full_alias_delta = alias_deltas["full_history_minus_recent3"]["macro"]
    tasks_with_positive_natural_delta = sum(
        value > 0.0 for value in natural_deltas["full_history_minus_recent3"]["by_task"].values()
    )
    tasks_with_positive_alias_delta = sum(
        value > 0.0 for value in alias_deltas["full_history_minus_recent3"]["by_task"].values()
    )
    elapsed_only_alias_accuracy = float(test["elapsed"]["alias_pair_ordering"]["macro"]["accuracy"])
    if elapsed_only_alias_accuracy > 0.6:
        conclusion = "MATCHING_STILL_TIME_CONFOUNDED"
        explanation = (
            f"Elapsed-only matched alias accuracy is {elapsed_only_alias_accuracy:.6f}, above 0.6; "
            "the elapsed matching still leaves a substantial time confound, so matched-pair history claims are not decisive."
        )
    elif full_natural_delta > 0.0 and full_alias_delta > 0.0 and tasks_with_positive_natural_delta >= 3 and tasks_with_positive_alias_delta >= 3:
        conclusion = "HISTORY_SIGNAL_SUPPORTED"
        explanation = (
            "Full-history improves both natural-candidate AUPRC and elapsed-matched alias ordering over Recent-3 "
            "across at least three tasks, while elapsed-only matching is near chance."
        )
    elif full_natural_delta <= 0.0 and full_alias_delta <= 0.0:
        conclusion = "HISTORY_SIGNAL_NOT_SUPPORTED"
        explanation = (
            "Full-history does not improve either the primary natural-candidate AUPRC or the elapsed-matched alias "
            "ordering over Recent-3."
        )
    else:
        conclusion = "MIXED_RESULT"
        explanation = (
            "Natural-candidate AUPRC and elapsed-matched alias ordering move in different directions; inspect the "
            "task-level metrics and natural_predictions.csv before drawing a history conclusion."
        )

    deltas = {
        "natural_candidate_auprc": natural_deltas,
        "alias_pair_accuracy": alias_deltas,
        "full_history_vs_recent3": {
            "alias_pair_accuracy": alias_deltas["full_history_minus_recent3"],
            "alias_mean_probability_margin": _delta_block(
                test["full_history"]["alias_pair_ordering"],
                test["recent3"]["alias_pair_ordering"],
                metric_name="mean_probability_margin",
            ),
            "alias_median_probability_margin": _delta_block(
                test["full_history"]["alias_pair_ordering"],
                test["recent3"]["alias_pair_ordering"],
                metric_name="median_probability_margin",
            ),
            "adjacent_pair_accuracy": _delta_block(
                test["full_history"]["adjacent_pair_ordering"],
                test["recent3"]["adjacent_pair_ordering"],
                metric_name="accuracy",
            ),
            "natural_candidate_auprc": natural_deltas["full_history_minus_recent3"],
        },
        "full_history_vs_current": {
            "alias_pair_accuracy": alias_deltas["full_history_minus_current"],
            "natural_candidate_auprc": natural_deltas["full_history_minus_current"],
        },
    }
    false_positive_analysis = _false_positive_analysis(natural_prediction_rows)
    results: dict[str, Any] = {
        "experiment": "temporal_history_observability",
        "manifest": str(args.manifest),
        "feature_cache": str(args.feature_cache),
        "seed": int(args.seed),
        "alias_mode": args.alias_mode,
        "alias_min_seconds": float(args.alias_min_seconds),
        "alias_min_frame_distance": alias_min_frames,
        "elapsed_match_tolerance_seconds": float(args.elapsed_match_tolerance_seconds),
        "fps": fps,
        "tick_stride_frames": int(manifest.tick_stride_frames),
        "cache_metadata": dataclasses.asdict(cache.metadata),
        "reconstruction_counts": reconstruction_counts,
        "pair_counts": pair_counts,
        "matching_quality": matching_quality,
        "sample_counts": sample_counts,
        "representations": representations,
        "deltas": deltas,
        "false_positive_analysis": false_positive_analysis,
        "natural_prediction_count": len(natural_prediction_rows),
        "automatic_conclusion": conclusion,
        "conclusion_details": {
            "elapsed_only_matched_alias_accuracy": elapsed_only_alias_accuracy,
            "full_history_minus_recent3_natural_auprc": full_natural_delta,
            "full_history_minus_recent3_alias_accuracy": full_alias_delta,
            "tasks_with_positive_natural_delta": tasks_with_positive_natural_delta,
            "tasks_with_positive_alias_delta": tasks_with_positive_alias_delta,
            "explanation": explanation,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_pair_predictions(args.output_dir / "pair_predictions.csv", pair_prediction_rows)
    _write_natural_predictions(args.output_dir / "natural_predictions.csv", natural_prediction_rows)
    _write_summary(args.output_dir / "summary.md", results)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved results to {args.output_dir}")
    print(f"Automatic conclusion: {conclusion}")


def main() -> int:
    _run(_parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
