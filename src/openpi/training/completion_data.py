"""Completion label audit and leak-free episode split manifests."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
import dataclasses
import json
import os
import pathlib
import random
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SplitName = Literal["train", "val", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")
MANIFEST_VERSION = 1


@dataclasses.dataclass(frozen=True)
class TaskGroup:
    group_id: int
    episode_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"group_id": self.group_id, "episode_ids": list(self.episode_ids)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskGroup:
        return cls(
            group_id=int(value["group_id"]),
            episode_ids=tuple(int(episode_id) for episode_id in value["episode_ids"]),
        )


@dataclasses.dataclass(frozen=True)
class SplitManifest:
    repo_id: str
    seed: int
    episodes_per_group: int
    all_episode_ids: tuple[int, ...]
    splits: Mapping[SplitName, tuple[TaskGroup, ...]]
    version: int = MANIFEST_VERSION

    def episode_ids(self, split: SplitName) -> tuple[int, ...]:
        return tuple(episode_id for group in self.splits[split] for episode_id in group.episode_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "repo_id": self.repo_id,
            "seed": self.seed,
            "episodes_per_group": self.episodes_per_group,
            "all_episode_ids": list(self.all_episode_ids),
            "splits": {split: [group.to_dict() for group in self.splits[split]] for split in SPLIT_NAMES},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SplitManifest:
        raw_splits = value["splits"]
        return cls(
            version=int(value["version"]),
            repo_id=str(value["repo_id"]),
            seed=int(value["seed"]),
            episodes_per_group=int(value["episodes_per_group"]),
            all_episode_ids=tuple(int(episode_id) for episode_id in value["all_episode_ids"]),
            splits={split: tuple(TaskGroup.from_dict(group) for group in raw_splits[split]) for split in SPLIT_NAMES},
        )


@dataclasses.dataclass(frozen=True)
class EpisodeAudit:
    episode_id: int
    frame_count: int
    positive_count: int
    negative_count: int
    # Progress labels additionally audit this value. Keeping it optional
    # preserves the binary audit's public shape and old manifests/tests.
    task_index: int | None = None


@dataclasses.dataclass(frozen=True)
class CompletionDataInfo:
    manifest: SplitManifest
    episode_audits: Mapping[int, EpisodeAudit]
    train_positive_count: int | None
    train_negative_count: int | None
    pos_weight: float | None
    # Boundary-scheme audits (per-episode index sets for deterministic training
    # sampling). Present only for the ``boundary`` label scheme.
    boundary_episode_audits: Mapping[int, BoundaryEpisodeAudit] | None = None


# Boundary completion label scheme constants. See ``label_completion_boundary``
# for the generation script and ``audit_boundary_completion_episode_parquet``
# for the per-episode audit. Episodes are grouped into ``BOUNDARY_GROUP`` (4)
# contiguous subtasks; subtasks 1/2/3 (group positions 0/1/2) append
# ``BOUNDARY_COPY_FRAMES`` frames copied from the next subtask, while subtask 4
# (group position 3) does not cross the group boundary.
BOUNDARY_GROUP: int = 4
BOUNDARY_COPY_FRAMES: int = 5
BOUNDARY_POSITIVE_TAIL: int = 5
BOUNDARY_SUBTASK4_TAIL: int = 10
# Every episode ends with exactly this many positive frames: subtasks 1/2/3 have
# BOUNDARY_POSITIVE_TAIL original + BOUNDARY_COPY_FRAMES copied; subtask 4 has
# BOUNDARY_SUBTASK4_TAIL original.
BOUNDARY_TOTAL_POSITIVES: int = 10
BOUNDARY_LABEL_SCHEME: str = "boundary"


@dataclasses.dataclass(frozen=True)
class BoundaryEpisodeAudit:
    """Per-episode audit result for the boundary completion scheme.

    Carries both the audited counts and the raw per-frame arrays needed to build
    the deterministic training sample index set (positive + ordinary negative +
    forced first-frame negative). ``group_position`` is the 0-3 index of this
    episode within its 4-episode task group (subtask 1/2/3/4).
    """

    episode_id: int
    group_position: int
    task_index: int
    frame_count: int
    positive_count: int
    negative_count: int
    boundary_copy_count: int
    is_subtask4: bool
    # Per-frame arrays, length ``frame_count``, in parquet row order.
    completion: np.ndarray
    source_episode_indices: np.ndarray
    source_frame_indices: np.ndarray
    is_boundary_copy: np.ndarray


def build_task_groups(
    episode_ids: Sequence[int],
    *,
    episodes_per_group: int = 4,
    minimum_groups: int = 11,
) -> tuple[TaskGroup, ...]:
    """Validates sorted-contiguous episode IDs and forms whole task groups."""

    if episodes_per_group <= 0:
        raise ValueError("episodes_per_group must be positive")
    sorted_ids = tuple(sorted(int(episode_id) for episode_id in episode_ids))
    if len(set(sorted_ids)) != len(sorted_ids):
        raise ValueError("episode IDs contain duplicates")
    if not sorted_ids:
        raise ValueError("dataset contains no episodes")
    expected_ids = tuple(range(sorted_ids[0], sorted_ids[0] + len(sorted_ids)))
    if sorted_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(sorted_ids))
        raise ValueError(
            "episode IDs must be continuous before grouping; "
            f"got range {sorted_ids[0]}..{sorted_ids[-1]} with missing IDs {missing}"
        )
    if len(sorted_ids) % episodes_per_group:
        raise ValueError(
            f"episode count {len(sorted_ids)} is not divisible by {episodes_per_group}; "
            "cannot form complete breakfast task groups"
        )
    group_count = len(sorted_ids) // episodes_per_group
    if group_count < minimum_groups:
        raise ValueError(f"at least {minimum_groups} complete groups are required, found {group_count}")
    return tuple(
        TaskGroup(group_id=index, episode_ids=sorted_ids[start : start + episodes_per_group])
        for index, start in enumerate(range(0, len(sorted_ids), episodes_per_group))
    )


def create_split_manifest(
    episode_ids: Sequence[int],
    *,
    repo_id: str,
    seed: int = 42,
    episodes_per_group: int = 4,
    val_groups: int = 5,
    test_groups: int = 5,
) -> SplitManifest:
    """Shuffles whole task groups deterministically, never individual episodes."""

    if val_groups < 0:
        raise ValueError("val_groups must be non-negative (0 disables the val split)")
    if test_groups < 0:
        raise ValueError("test_groups must be non-negative")
    minimum_groups = val_groups + test_groups + 1
    groups = list(
        build_task_groups(
            episode_ids,
            episodes_per_group=episodes_per_group,
            minimum_groups=minimum_groups,
        )
    )
    random.Random(seed).shuffle(groups)
    val = tuple(groups[:val_groups])
    test = tuple(groups[val_groups : val_groups + test_groups])
    train = tuple(groups[val_groups + test_groups :])
    manifest = SplitManifest(
        repo_id=repo_id,
        seed=seed,
        episodes_per_group=episodes_per_group,
        all_episode_ids=tuple(sorted(int(episode_id) for episode_id in episode_ids)),
        splits={"train": train, "val": val, "test": test},
    )
    validate_split_manifest(
        manifest,
        episode_ids=episode_ids,
        repo_id=repo_id,
        seed=seed,
        episodes_per_group=episodes_per_group,
        val_groups=val_groups,
        test_groups=test_groups,
    )
    return manifest


def validate_split_manifest(
    manifest: SplitManifest,
    *,
    episode_ids: Sequence[int],
    repo_id: str,
    seed: int,
    episodes_per_group: int,
    val_groups: int,
    test_groups: int,
) -> None:
    """Rejects stale, malformed, or leaking manifests with actionable errors."""

    if val_groups < 0:
        raise ValueError("val_groups must be non-negative (0 disables the val split)")
    if test_groups < 0:
        raise ValueError("test_groups must be non-negative")
    if manifest.version != MANIFEST_VERSION:
        raise ValueError(f"split manifest version {manifest.version} is unsupported; expected {MANIFEST_VERSION}")
    if manifest.repo_id != repo_id:
        raise ValueError(f"split manifest repo_id {manifest.repo_id!r} does not match configured {repo_id!r}")
    if manifest.seed != seed:
        raise ValueError(f"split manifest seed {manifest.seed} does not match configured seed {seed}")
    if manifest.episodes_per_group != episodes_per_group:
        raise ValueError(
            "split manifest episodes_per_group "
            f"{manifest.episodes_per_group} does not match configured {episodes_per_group}"
        )
    expected_ids = tuple(sorted(int(episode_id) for episode_id in episode_ids))
    if manifest.all_episode_ids != expected_ids:
        raise ValueError(
            "split manifest episode IDs do not match the dataset; "
            f"manifest={manifest.all_episode_ids}, dataset={expected_ids}"
        )
    if len(manifest.splits["val"]) != val_groups:
        raise ValueError(f"split manifest must contain {val_groups} val groups")
    if len(manifest.splits["test"]) != test_groups:
        raise ValueError(f"split manifest must contain {test_groups} test groups")
    if not manifest.splits["train"]:
        raise ValueError("split manifest must contain at least one train group")

    canonical_groups = {
        group.group_id: group
        for group in build_task_groups(
            expected_ids,
            episodes_per_group=episodes_per_group,
            minimum_groups=val_groups + test_groups + 1,
        )
    }
    seen_groups: dict[int, SplitName] = {}
    seen_episodes: dict[int, SplitName] = {}
    for split in SPLIT_NAMES:
        for group in manifest.splits[split]:
            expected_group = canonical_groups.get(group.group_id)
            if expected_group is None or group.episode_ids != expected_group.episode_ids:
                raise ValueError(f"split {split} contains malformed group {group.group_id}: {group.episode_ids}")
            if group.group_id in seen_groups:
                raise ValueError(f"task group {group.group_id} leaks across {seen_groups[group.group_id]} and {split}")
            seen_groups[group.group_id] = split
            for episode_id in group.episode_ids:
                if episode_id in seen_episodes:
                    raise ValueError(f"episode {episode_id} leaks across {seen_episodes[episode_id]} and {split}")
                seen_episodes[episode_id] = split
    if set(seen_episodes) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(seen_episodes))
        raise ValueError(f"split manifest does not assign all episodes; missing {missing}")


def load_or_create_split_manifest(
    manifest_path: str | os.PathLike[str],
    episode_ids: Sequence[int],
    *,
    repo_id: str,
    seed: int = 42,
    episodes_per_group: int = 4,
    val_groups: int = 5,
    test_groups: int = 5,
) -> SplitManifest:
    """Reuses an existing manifest, otherwise writes one atomically."""

    path = pathlib.Path(manifest_path)
    if path.exists():
        manifest = SplitManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
        validate_split_manifest(
            manifest,
            episode_ids=episode_ids,
            repo_id=repo_id,
            seed=seed,
            episodes_per_group=episodes_per_group,
            val_groups=val_groups,
            test_groups=test_groups,
        )
        return manifest

    manifest = create_split_manifest(
        episode_ids,
        repo_id=repo_id,
        seed=seed,
        episodes_per_group=episodes_per_group,
        val_groups=val_groups,
        test_groups=test_groups,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)
    return manifest


def _as_scalar_array(column: Any, *, column_name: str, episode_id: int) -> np.ndarray:
    values = column.combine_chunks().to_pylist()
    scalars = []
    for frame_index, value in enumerate(values):
        scalar = value
        if isinstance(scalar, list | tuple):
            if len(scalar) != 1:
                raise ValueError(
                    f"episode {episode_id} field {column_name!r} is not scalar at frame {frame_index}: {scalar!r}"
                )
            scalar = scalar[0]
        scalars.append(scalar)
    return np.asarray(scalars)


def make_progress_targets(frame_count: int) -> np.ndarray:
    """Returns exact float32 linear within-episode progress targets.

    A single-frame subtask is conventionally complete at its only frame. For
    every longer episode the explicit assignments keep both endpoints bitwise
    exact after float32 arithmetic.
    """

    if frame_count <= 0:
        raise ValueError(f"progress labels require a positive frame count, got {frame_count}")
    if frame_count == 1:
        return np.ones((1,), dtype=np.float32)
    targets = np.arange(frame_count, dtype=np.float32) / np.float32(frame_count - 1)
    targets[0] = np.float32(0.0)
    targets[-1] = np.float32(1.0)
    return targets


def make_window_completion_targets(frame_count: int, window_frames: int) -> np.ndarray:
    """Returns exact float32 labels: ``1.0`` over the trailing ``window_frames``.

    Every frame before the trailing window is ``0.0``. An episode shorter
    than the window is exempt from the window rule and labeled all-``0.0``
    instead, mirroring the legacy ``frame_count < 2`` exemption of
    ``audit_episode_parquet`` — the window doesn't fully fit inside such a
    short episode's own timeline, and padding it with frames borrowed from an
    adjacent episode would mean labeling frames already annotated as the next
    subtask as if they still belonged to this one.
    """

    if window_frames <= 0:
        raise ValueError(f"window_frames must be positive, got {window_frames}")
    targets = np.zeros(frame_count, dtype=np.float32)
    if frame_count >= window_frames:
        targets[-window_frames:] = np.float32(1.0)
    return targets


def make_window_progress_targets(frame_count: int, window_frames: int, ramp_start: float) -> np.ndarray:
    """Returns exact float32 labels: ``0.0`` before the trailing window, then a
    linear ramp from ``ramp_start`` to ``1.0`` across the trailing ``window_frames``.

    See ``make_window_completion_targets`` for why ``frame_count < window_frames``
    is exempt (labeled all-``0.0``) rather than padded from another episode.
    """

    if window_frames <= 0:
        raise ValueError(f"window_frames must be positive, got {window_frames}")
    if not 0.0 <= ramp_start < 1.0:
        raise ValueError(f"ramp_start must be in [0, 1), got {ramp_start}")
    targets = np.zeros(frame_count, dtype=np.float32)
    if frame_count < window_frames:
        return targets
    if window_frames == 1:
        ramp = np.ones(1, dtype=np.float32)
    else:
        fraction = np.arange(window_frames, dtype=np.float32) / np.float32(window_frames - 1)
        ramp = fraction * np.float32(1.0 - ramp_start) + np.float32(ramp_start)
        ramp[0] = np.float32(ramp_start)
        ramp[-1] = np.float32(1.0)
    targets[-window_frames:] = ramp
    return targets


def _validate_episode_and_frame_indices(
    episode_values: np.ndarray,
    frame_values: np.ndarray,
    *,
    episode_id: int,
    expected_length: int,
) -> None:
    if not np.all(episode_values == episode_id):
        bad = np.flatnonzero(episode_values != episode_id).tolist()
        raise ValueError(f"episode {episode_id} parquet contains mismatched episode_index at rows {bad}")
    try:
        integer_frames = frame_values.astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"episode {episode_id} has non-integer frame_index values") from error
    if not np.array_equal(frame_values, integer_frames):
        raise ValueError(f"episode {episode_id} has non-integer frame_index values: {frame_values.tolist()}")
    if not np.array_equal(integer_frames, np.arange(expected_length, dtype=np.int64)):
        raise ValueError(
            f"episode {episode_id} frame_index must be exactly 0..{expected_length - 1}, got {integer_frames.tolist()}"
        )


def audit_episode_parquet(
    parquet_path: str | os.PathLike[str],
    *,
    episode_id: int,
    expected_length: int,
    label_key: str = "completion",
    window_frames: int = 2,
) -> EpisodeAudit:
    """Reads only scalar audit columns and enforces the exact last-N label rule.

    ``window_frames`` defaults to ``2`` (the legacy last-two-frames rule).
    Episodes shorter than ``window_frames`` are exempt from the last-N rule;
    they must be labeled all-0 instead.
    """

    if window_frames <= 0:
        raise ValueError(f"window_frames must be positive, got {window_frames}")
    path = pathlib.Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"episode {episode_id} parquet file is missing: {path}")
    parquet_file = pq.ParquetFile(path)
    available_columns = set(parquet_file.schema_arrow.names)
    required_columns = {"episode_index", "frame_index", label_key}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise ValueError(f"episode {episode_id} is missing parquet field(s) {missing_columns} in {path}")
    table = parquet_file.read(columns=["episode_index", "frame_index", label_key])
    episode_values = _as_scalar_array(table["episode_index"], column_name="episode_index", episode_id=episode_id)
    frame_values = _as_scalar_array(table["frame_index"], column_name="frame_index", episode_id=episode_id)
    labels = _as_scalar_array(table[label_key], column_name=label_key, episode_id=episode_id)

    if len(labels) != expected_length:
        raise ValueError(
            f"episode {episode_id} has {len(labels)} parquet rows but metadata length is {expected_length}"
        )
    _validate_episode_and_frame_indices(
        episode_values,
        frame_values,
        episode_id=episode_id,
        expected_length=expected_length,
    )

    numeric_labels = np.asarray(labels)
    valid_label_mask = np.logical_or(numeric_labels == 0, numeric_labels == 1)
    if not np.all(valid_label_mask):
        bad_rows = np.flatnonzero(~valid_label_mask)
        details = [
            (int(row), labels[int(row)].item() if hasattr(labels[int(row)], "item") else labels[int(row)])
            for row in bad_rows
        ]
        raise ValueError(f"episode {episode_id} field {label_key!r} contains labels outside 0/1: {details}")
    numeric_labels = numeric_labels.astype(np.int8)

    # Episodes shorter than window_frames cannot satisfy the last-N-frames
    # rule, so they are labeled all-0 and exempt from that part of the audit.
    if expected_length < window_frames:
        positive_rows = np.flatnonzero(numeric_labels != 0).tolist()
        if positive_rows:
            raise ValueError(
                f"episode {episode_id} has only {expected_length} frame(s); all labels must be 0, "
                f"but found positive labels at frames {positive_rows}"
            )
        positive_count = 0
        return EpisodeAudit(
            episode_id=episode_id,
            frame_count=expected_length,
            positive_count=positive_count,
            negative_count=expected_length - positive_count,
        )

    early_positive_rows = np.flatnonzero(numeric_labels[:-window_frames] != 0).tolist()
    if early_positive_rows:
        raise ValueError(
            f"episode {episode_id} must have {label_key}=0 before its last {window_frames} frames; "
            f"offending frames {early_positive_rows}"
        )
    final_bad_rows = (np.flatnonzero(numeric_labels[-window_frames:] != 1) + expected_length - window_frames).tolist()
    if final_bad_rows:
        raise ValueError(
            f"episode {episode_id} must have {label_key}=1 on exactly its last {window_frames} frames; "
            f"offending frames {final_bad_rows}"
        )
    positive_count = int(np.sum(numeric_labels))
    return EpisodeAudit(
        episode_id=episode_id,
        frame_count=expected_length,
        positive_count=positive_count,
        negative_count=expected_length - positive_count,
    )


def audit_progress_episode_parquet(
    parquet_path: str | os.PathLike[str],
    *,
    episode_id: int,
    expected_length: int,
    label_key: str = "progress",
) -> EpisodeAudit:
    """Audits one subtask's exact float32 linear progress labels.

    Progress data is deliberately stricter than the legacy completion audit:
    every episode must contain one task index, contiguous local frame indices,
    and the exact target implied by its own length. This prevents an episode
    boundary or subtask-ID bug from silently becoming a regression target.
    """

    path = pathlib.Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"episode {episode_id} parquet file is missing: {path}")
    parquet_file = pq.ParquetFile(path)
    available_columns = set(parquet_file.schema_arrow.names)
    required_columns = {"episode_index", "frame_index", "task_index", label_key}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise ValueError(f"episode {episode_id} is missing parquet field(s) {missing_columns} in {path}")
    progress_field = parquet_file.schema_arrow.field(label_key)
    if not pa.types.is_float32(progress_field.type):
        raise ValueError(f"episode {episode_id} field {label_key!r} must be parquet float32, got {progress_field.type}")
    table = parquet_file.read(columns=["episode_index", "frame_index", "task_index", label_key])
    episode_values = _as_scalar_array(table["episode_index"], column_name="episode_index", episode_id=episode_id)
    frame_values = _as_scalar_array(table["frame_index"], column_name="frame_index", episode_id=episode_id)
    task_values = _as_scalar_array(table["task_index"], column_name="task_index", episode_id=episode_id)
    labels = _as_scalar_array(table[label_key], column_name=label_key, episode_id=episode_id)

    if len(labels) != expected_length:
        raise ValueError(
            f"episode {episode_id} has {len(labels)} parquet rows but metadata length is {expected_length}"
        )
    _validate_episode_and_frame_indices(
        episode_values,
        frame_values,
        episode_id=episode_id,
        expected_length=expected_length,
    )
    try:
        integer_tasks = task_values.astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"episode {episode_id} has non-integer task_index values") from error
    if not np.array_equal(task_values, integer_tasks):
        raise ValueError(f"episode {episode_id} has non-integer task_index values: {task_values.tolist()}")
    unique_tasks = np.unique(integer_tasks)
    if len(unique_tasks) != 1:
        raise ValueError(f"episode {episode_id} must contain exactly one task_index, got {unique_tasks.tolist()}")

    try:
        progress = np.asarray(labels, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"episode {episode_id} field {label_key!r} contains non-numeric progress labels") from error
    if not np.all(np.isfinite(progress)):
        bad_rows = np.flatnonzero(~np.isfinite(progress)).tolist()
        raise ValueError(f"episode {episode_id} field {label_key!r} contains non-finite labels at rows {bad_rows}")
    if np.any((progress < 0.0) | (progress > 1.0)):
        bad_rows = np.flatnonzero((progress < 0.0) | (progress > 1.0)).tolist()
        raise ValueError(f"episode {episode_id} field {label_key!r} contains labels outside [0, 1] at rows {bad_rows}")
    expected_progress = make_progress_targets(expected_length)
    if expected_length > 1 and progress[0] != np.float32(0.0):
        raise ValueError(f"episode {episode_id} progress first frame must be exactly 0.0")
    if progress[-1] != np.float32(1.0):
        raise ValueError(f"episode {episode_id} progress last frame must be exactly 1.0")
    if np.any(np.diff(progress) < 0.0):
        bad_rows = (np.flatnonzero(np.diff(progress) < 0.0) + 1).tolist()
        raise ValueError(f"episode {episode_id} progress must be monotonic non-decreasing; offending rows {bad_rows}")
    if not np.array_equal(progress, expected_progress):
        raise ValueError(
            f"episode {episode_id} field {label_key!r} must equal local_frame_index / "
            f"(episode_length - 1): got {progress.tolist()}, expected {expected_progress.tolist()}"
        )
    return EpisodeAudit(
        episode_id=episode_id,
        frame_count=expected_length,
        positive_count=0,
        negative_count=0,
        task_index=int(unique_tasks[0]),
    )


def audit_window_progress_episode_parquet(
    parquet_path: str | os.PathLike[str],
    *,
    episode_id: int,
    expected_length: int,
    window_frames: int,
    ramp_start: float,
    label_key: str = "progress",
) -> EpisodeAudit:
    """Audits one subtask's exact float32 tail-window ramp labels.

    Same structural checks as ``audit_progress_episode_parquet`` (single
    ``task_index``, contiguous frame indices, finite values in ``[0, 1]``),
    but compares against ``make_window_progress_targets`` instead of the
    full-episode linear ramp. An episode shorter than ``window_frames`` is
    exempt (must be all-``0.0``, matching ``make_window_progress_targets``'s
    own exemption) rather than required to end at exactly ``1.0``.
    """

    path = pathlib.Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"episode {episode_id} parquet file is missing: {path}")
    parquet_file = pq.ParquetFile(path)
    available_columns = set(parquet_file.schema_arrow.names)
    required_columns = {"episode_index", "frame_index", "task_index", label_key}
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise ValueError(f"episode {episode_id} is missing parquet field(s) {missing_columns} in {path}")
    progress_field = parquet_file.schema_arrow.field(label_key)
    if not pa.types.is_float32(progress_field.type):
        raise ValueError(f"episode {episode_id} field {label_key!r} must be parquet float32, got {progress_field.type}")
    table = parquet_file.read(columns=["episode_index", "frame_index", "task_index", label_key])
    episode_values = _as_scalar_array(table["episode_index"], column_name="episode_index", episode_id=episode_id)
    frame_values = _as_scalar_array(table["frame_index"], column_name="frame_index", episode_id=episode_id)
    task_values = _as_scalar_array(table["task_index"], column_name="task_index", episode_id=episode_id)
    labels = _as_scalar_array(table[label_key], column_name=label_key, episode_id=episode_id)

    if len(labels) != expected_length:
        raise ValueError(
            f"episode {episode_id} has {len(labels)} parquet rows but metadata length is {expected_length}"
        )
    _validate_episode_and_frame_indices(
        episode_values,
        frame_values,
        episode_id=episode_id,
        expected_length=expected_length,
    )
    try:
        integer_tasks = task_values.astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"episode {episode_id} has non-integer task_index values") from error
    if not np.array_equal(task_values, integer_tasks):
        raise ValueError(f"episode {episode_id} has non-integer task_index values: {task_values.tolist()}")
    unique_tasks = np.unique(integer_tasks)
    if len(unique_tasks) != 1:
        raise ValueError(f"episode {episode_id} must contain exactly one task_index, got {unique_tasks.tolist()}")

    try:
        progress = np.asarray(labels, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"episode {episode_id} field {label_key!r} contains non-numeric progress labels") from error
    if not np.all(np.isfinite(progress)):
        bad_rows = np.flatnonzero(~np.isfinite(progress)).tolist()
        raise ValueError(f"episode {episode_id} field {label_key!r} contains non-finite labels at rows {bad_rows}")
    if np.any((progress < 0.0) | (progress > 1.0)):
        bad_rows = np.flatnonzero((progress < 0.0) | (progress > 1.0)).tolist()
        raise ValueError(f"episode {episode_id} field {label_key!r} contains labels outside [0, 1] at rows {bad_rows}")
    if expected_length < window_frames:
        positive_rows = np.flatnonzero(progress != 0.0).tolist()
        if positive_rows:
            raise ValueError(
                f"episode {episode_id} has only {expected_length} frame(s), shorter than "
                f"window_frames={window_frames}; all labels must be 0.0, but found nonzero "
                f"labels at frames {positive_rows}"
            )
    else:
        if progress[-1] != np.float32(1.0):
            raise ValueError(f"episode {episode_id} progress last frame must be exactly 1.0")
        if np.any(np.diff(progress) < 0.0):
            bad_rows = (np.flatnonzero(np.diff(progress) < 0.0) + 1).tolist()
            raise ValueError(
                f"episode {episode_id} progress must be monotonic non-decreasing; offending rows {bad_rows}"
            )
    expected_progress = make_window_progress_targets(expected_length, window_frames, ramp_start)
    if not np.array_equal(progress, expected_progress):
        raise ValueError(
            f"episode {episode_id} field {label_key!r} must equal the tail-window ramp "
            f"(window_frames={window_frames}, ramp_start={ramp_start}): "
            f"got {progress.tolist()}, expected {expected_progress.tolist()}"
        )
    return EpisodeAudit(
        episode_id=episode_id,
        frame_count=expected_length,
        positive_count=0,
        negative_count=0,
        task_index=int(unique_tasks[0]),
    )


def _read_label_scheme(dataset_root: pathlib.Path) -> str | None:
    """Reads the optional ``label_scheme`` field from ``meta/info.json``.

    ``"boundary"`` selects the boundary completion audit; absence keeps the
    legacy window/progress audits. Keeping the scheme a property of the dataset
    itself (rather than a training-config value) prevents drift between what the
    dataset was generated with and what the trainer audits.
    """

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    scheme = info.get("label_scheme")
    if scheme is None:
        return None
    if not isinstance(scheme, str) or not scheme:
        raise ValueError(f"invalid label_scheme in {info_path}: {scheme!r}")
    return scheme


def _read_boundary_excluded_episode_ids(dataset_root: pathlib.Path) -> tuple[int, ...]:
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    raw = info.get("boundary_excluded_episode_indices", [])
    if not isinstance(raw, list) or any(not isinstance(value, int) for value in raw):
        raise ValueError(f"invalid boundary_excluded_episode_indices in {info_path}: {raw!r}")
    if len(set(raw)) != len(raw):
        raise ValueError(f"duplicate boundary_excluded_episode_indices in {info_path}: {raw!r}")
    return tuple(raw)


def audit_boundary_completion_episode_parquet(
    parquet_path: str | os.PathLike[str],
    *,
    episode_id: int,
    expected_length: int,
    group_episode_ids: Sequence[int],
    group_position: int,
    label_key: str = "completion",
    excluded_episode_ids: Collection[int] = (),
) -> BoundaryEpisodeAudit:
    """Audits one episode of the boundary completion scheme.

    Enforces the strict label rule (last ``BOUNDARY_TOTAL_POSITIVES`` frames are
    1, everything before is 0), the boundary-copy provenance (subtasks 1/2/3
    append ``BOUNDARY_COPY_FRAMES`` frames copied from the next subtask with the
    current subtask's ``task_index``), and that no copy crosses a task-group
    boundary. Returns the per-frame arrays needed to build the deterministic
    training sample set.
    """

    if group_position < 0 or group_position >= BOUNDARY_GROUP:
        raise ValueError(f"group_position must be in 0..{BOUNDARY_GROUP - 1}, got {group_position}")
    if len(group_episode_ids) != BOUNDARY_GROUP:
        raise ValueError(f"boundary group must contain exactly {BOUNDARY_GROUP} episodes, got {len(group_episode_ids)}")
    if group_episode_ids[group_position] != episode_id:
        raise ValueError(
            f"group_episode_ids[{group_position}]={group_episode_ids[group_position]} != episode_id {episode_id}"
        )
    path = pathlib.Path(parquet_path)
    if not path.is_file():
        raise FileNotFoundError(f"episode {episode_id} parquet file is missing: {path}")
    parquet_file = pq.ParquetFile(path)
    available_columns = set(parquet_file.schema_arrow.names)
    required_columns = {
        "episode_index",
        "frame_index",
        "task_index",
        label_key,
        "source_episode_index",
        "source_frame_index",
        "is_boundary_copy",
    }
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise ValueError(f"episode {episode_id} is missing parquet field(s) {missing_columns} in {path}")
    table = parquet_file.read(columns=sorted(required_columns))
    episode_values = _as_scalar_array(table["episode_index"], column_name="episode_index", episode_id=episode_id)
    frame_values = _as_scalar_array(table["frame_index"], column_name="frame_index", episode_id=episode_id)
    task_values = _as_scalar_array(table["task_index"], column_name="task_index", episode_id=episode_id)
    labels = _as_scalar_array(table[label_key], column_name=label_key, episode_id=episode_id)
    source_episode = _as_scalar_array(
        table["source_episode_index"], column_name="source_episode_index", episode_id=episode_id
    )
    source_frame = _as_scalar_array(
        table["source_frame_index"], column_name="source_frame_index", episode_id=episode_id
    )
    is_copy = _as_scalar_array(table["is_boundary_copy"], column_name="is_boundary_copy", episode_id=episode_id)

    if len(labels) != expected_length:
        raise ValueError(
            f"episode {episode_id} has {len(labels)} parquet rows but metadata length is {expected_length}"
        )
    _validate_episode_and_frame_indices(
        episode_values,
        frame_values,
        episode_id=episode_id,
        expected_length=expected_length,
    )

    try:
        integer_tasks = task_values.astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"episode {episode_id} has non-integer task_index values") from error
    if not np.array_equal(task_values, integer_tasks):
        raise ValueError(f"episode {episode_id} has non-integer task_index values: {task_values.tolist()}")
    unique_tasks = np.unique(integer_tasks)
    if len(unique_tasks) != 1:
        raise ValueError(f"episode {episode_id} must contain exactly one task_index, got {unique_tasks.tolist()}")
    if int(unique_tasks[0]) != group_position:
        raise ValueError(
            f"episode {episode_id} task_index {int(unique_tasks[0])} does not match group_position {group_position}"
        )

    numeric_labels = np.asarray(labels)
    if not np.all(np.logical_or(numeric_labels == 0, numeric_labels == 1)):
        bad_rows = np.flatnonzero(~np.logical_or(numeric_labels == 0, numeric_labels == 1)).tolist()
        raise ValueError(f"episode {episode_id} field {label_key!r} contains labels outside 0/1: rows {bad_rows}")
    numeric_labels = numeric_labels.astype(np.int8)

    copy_values = np.asarray(is_copy)
    if not np.all(np.logical_or(copy_values == 0, copy_values == 1)):
        raise ValueError(f"episode {episode_id} is_boundary_copy contains values outside 0/1")
    copy_mask = copy_values.astype(bool)

    excluded = {int(value) for value in excluded_episode_ids}
    is_excluded = episode_id in excluded
    is_subtask4 = group_position == BOUNDARY_GROUP - 1
    copy_source_episode = next(
        (int(candidate) for candidate in group_episode_ids[group_position + 1 :] if int(candidate) not in excluded),
        None,
    )
    expected_copy_count = 0 if is_subtask4 or is_excluded or copy_source_episode is None else BOUNDARY_COPY_FRAMES
    expected_positive_count = (
        0 if is_excluded else (BOUNDARY_SUBTASK4_TAIL if is_subtask4 else BOUNDARY_POSITIVE_TAIL + expected_copy_count)
    )
    original_length = expected_length - expected_copy_count
    required_tail = BOUNDARY_SUBTASK4_TAIL if is_subtask4 else BOUNDARY_POSITIVE_TAIL
    if is_excluded and original_length >= required_tail:
        raise ValueError(
            f"episode {episode_id} is marked excluded but has {original_length} source frames, enough for tail {required_tail}"
        )
    if not is_excluded and original_length < required_tail:
        raise ValueError(
            f"episode {episode_id} has only {original_length} source frames and must be listed in "
            "boundary_excluded_episode_indices"
        )

    prefix_length = expected_length - expected_positive_count
    early_positive_rows = np.flatnonzero(numeric_labels[:prefix_length] != 0).tolist()
    if early_positive_rows:
        raise ValueError(
            f"episode {episode_id} must have {label_key}=0 before its last {expected_positive_count} frames; "
            f"offending frames {early_positive_rows}"
        )
    final_bad_rows = (
        np.flatnonzero(numeric_labels[prefix_length:] != 1) + prefix_length
        if expected_positive_count
        else np.empty(0, dtype=np.int64)
    ).tolist()
    if final_bad_rows:
        raise ValueError(
            f"episode {episode_id} must have {label_key}=1 on exactly its last {expected_positive_count} frames; "
            f"offending frames {final_bad_rows}"
        )
    positive_count = int(np.sum(numeric_labels))
    if positive_count != expected_positive_count:
        raise ValueError(
            f"episode {episode_id} must have exactly {expected_positive_count} positives, got {positive_count}"
        )

    source_episode = source_episode.astype(np.int64)
    source_frame = source_frame.astype(np.int64)
    if expected_copy_count == 0:
        if np.any(copy_mask):
            reason = "subtask 4" if is_subtask4 else "excluded boundary"
            raise ValueError(f"episode {episode_id} ({reason}) must contain no boundary copies")
        boundary_copy_count = 0
        if not np.all(source_episode == episode_id):
            bad = np.flatnonzero(source_episode != episode_id).tolist()
            raise ValueError(f"episode {episode_id} source_episode_index must be self at rows {bad}")
        if not np.array_equal(source_frame, frame_values.astype(np.int64)):
            raise ValueError(f"episode {episode_id} source_frame_index must equal frame_index")
    else:
        assert copy_source_episode is not None
        if int(np.sum(copy_mask)) != expected_copy_count:
            raise ValueError(
                f"episode {episode_id} (subtask {group_position + 1}) must have exactly "
                f"{expected_copy_count} boundary copies, got {int(np.sum(copy_mask))}"
            )
        if not np.array_equal(copy_mask[-expected_copy_count:], np.ones(expected_copy_count, dtype=bool)):
            raise ValueError(
                f"episode {episode_id} boundary copies must occupy exactly the last {BOUNDARY_COPY_FRAMES} rows"
            )
        if np.any(copy_mask[:-expected_copy_count]):
            raise ValueError(f"episode {episode_id} boundary copies must be confined to the trailing rows")
        if not np.all(source_episode[copy_mask] == copy_source_episode):
            raise ValueError(
                f"episode {episode_id} boundary copies must source from next valid episode {copy_source_episode}"
            )
        if not np.array_equal(source_frame[copy_mask], np.arange(expected_copy_count, dtype=np.int64)):
            raise ValueError(
                f"episode {episode_id} boundary copies must source frame indices 0..{expected_copy_count - 1}"
            )
        # Copy frames keep the current subtask's task_index (already verified single task_index).
        non_copy = ~copy_mask
        if not np.all(source_episode[non_copy] == episode_id):
            raise ValueError(f"episode {episode_id} non-copy frames must source from self")
        if not np.array_equal(source_frame[non_copy], np.arange(original_length, dtype=np.int64)):
            raise ValueError(f"episode {episode_id} non-copy source_frame_index must be 0..{original_length - 1}")
        boundary_copy_count = expected_copy_count

    # No copy may cross a task-group boundary.
    group_id_set = {int(episode) for episode in group_episode_ids}
    out_of_group = np.flatnonzero(~np.isin(source_episode, np.asarray(sorted(group_id_set), dtype=np.int64)))
    if out_of_group.size:
        raise ValueError(
            f"episode {episode_id} has source_episode_index outside its task group at rows {out_of_group.tolist()}"
        )

    return BoundaryEpisodeAudit(
        episode_id=episode_id,
        group_position=group_position,
        task_index=int(unique_tasks[0]),
        frame_count=expected_length,
        positive_count=positive_count,
        negative_count=expected_length - positive_count,
        boundary_copy_count=boundary_copy_count,
        is_subtask4=is_subtask4,
        completion=numeric_labels.astype(np.int8),
        source_episode_indices=source_episode,
        source_frame_indices=source_frame,
        is_boundary_copy=copy_mask.astype(np.int8),
    )


def boundary_train_sample_indices(
    audit: BoundaryEpisodeAudit,
    *,
    stride: int = 15,
    forced_first_n: int = BOUNDARY_COPY_FRAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(positive, ordinary_negative, forced_negative)`` local indices.

    Positive frames are all kept. Ordinary negatives are original (non-copy)
    negative frames whose ``source_frame_index`` falls on the ``stride`` grid.
    Forced negatives are the first ``forced_first_n`` original frames of
    subtasks 2/3/4 (group positions 1/2/3), kept even off the stride grid so the
    same-observation/different-prompt contrast pairs stay in the training set.
    """

    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if forced_first_n < 0:
        raise ValueError(f"forced_first_n must be non-negative, got {forced_first_n}")
    completion = audit.completion
    is_copy = audit.is_boundary_copy.astype(bool)
    source_frame = audit.source_frame_indices
    positives = np.flatnonzero(completion == 1).astype(np.int64)
    ordinary_mask = (completion == 0) & (~is_copy) & (source_frame % stride == 0)
    ordinary = np.flatnonzero(ordinary_mask).astype(np.int64)
    if audit.group_position in (1, 2, 3):
        forced_mask = (completion == 0) & (~is_copy) & (source_frame < forced_first_n)
        forced = np.flatnonzero(forced_mask).astype(np.int64)
    else:
        forced = np.empty(0, dtype=np.int64)
    return positives, ordinary, forced


def build_boundary_train_sample_set(
    audit: BoundaryEpisodeAudit,
    *,
    stride: int = 15,
    forced_first_n: int = BOUNDARY_COPY_FRAMES,
) -> np.ndarray:
    """Returns the sorted, de-duplicated training sample index set for one episode."""

    positives, ordinary, forced = boundary_train_sample_indices(audit, stride=stride, forced_first_n=forced_first_n)
    return np.unique(np.concatenate([positives, ordinary, forced]))


def _read_window_label_metadata(
    dataset_root: pathlib.Path, *, objective: Literal["binary", "progress"]
) -> tuple[int | None, float | None]:
    """Reads the optional tail-window label contract a label script wrote.

    Returns ``(window_frames, ramp_start)``. Both are ``None`` when the
    dataset's ``meta/info.json`` has no ``window_seconds`` key, meaning it
    uses the legacy last-2-frames / full-episode-linear labels. This keeps
    the window width/shape a property of the dataset itself rather than a
    separate training-config value that could drift out of sync with what
    the dataset was actually generated with.
    """

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"dataset metadata not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    window_seconds = info.get("window_seconds")
    if window_seconds is None:
        return None, None
    fps = float(info["fps"])
    if fps <= 0:
        raise ValueError(f"dataset fps must be positive, got {fps}")
    window_frames = round(float(window_seconds) * fps)
    if window_frames <= 0:
        raise ValueError(f"window_seconds={window_seconds} at fps={fps} rounds to a non-positive window_frames")
    if objective != "progress":
        return window_frames, None
    if "ramp_start" not in info:
        raise ValueError(f"{dataset_root} declares window_seconds but is missing ramp_start for a progress dataset")
    return window_frames, float(info["ramp_start"])


def prepare_completion_data(
    dataset_metadata: Any,
    *,
    repo_id: str,
    dataset_root: str | os.PathLike[str],
    label_key: str,
    manifest_path: str | os.PathLike[str],
    manifest_repo_id: str | None = None,
    objective: Literal["binary", "progress"] = "binary",
    audit_labels: bool = True,
    seed: int = 42,
    episodes_per_group: int = 4,
    val_groups: int = 5,
    test_groups: int = 5,
) -> CompletionDataInfo:
    """Persists/reuses splits and audits labels for the selected head objective."""

    if objective not in ("binary", "progress"):
        raise ValueError(f"unsupported completion objective: {objective!r}")

    if manifest_repo_id is not None and not manifest_repo_id:
        raise ValueError("manifest_repo_id must not be empty when set")
    canonical_manifest_repo_id = manifest_repo_id or repo_id
    episode_ids = tuple(sorted(int(episode_id) for episode_id in dataset_metadata.episodes))
    if not audit_labels:
        manifest = load_or_create_split_manifest(
            manifest_path,
            episode_ids,
            repo_id=canonical_manifest_repo_id,
            seed=seed,
            episodes_per_group=episodes_per_group,
            val_groups=val_groups,
            test_groups=test_groups,
        )
        return CompletionDataInfo(
            manifest=manifest,
            episode_audits={},
            train_positive_count=None,
            train_negative_count=None,
            pos_weight=None,
        )

    if label_key not in dataset_metadata.features:
        raise ValueError(f"{objective} label field {label_key!r} is missing from dataset metadata")
    root = pathlib.Path(dataset_root)
    parquet_paths = {episode_id: root / dataset_metadata.get_data_file_path(episode_id) for episode_id in episode_ids}
    missing_paths = [path for path in parquet_paths.values() if not path.is_file()]
    if missing_paths:
        preview = ", ".join(str(path) for path in missing_paths[:3])
        remaining = len(missing_paths) - 3
        suffix = f", and {remaining} more" if remaining > 0 else ""
        raise FileNotFoundError(
            "completion head training requires a fully mounted local LeRobot dataset; "
            f"missing {len(missing_paths)} episode parquet file(s) under {root}: {preview}{suffix}. "
            "Automatic parquet download is disabled; mount or copy the complete dataset before retrying."
        )
    manifest = load_or_create_split_manifest(
        manifest_path,
        episode_ids,
        repo_id=canonical_manifest_repo_id,
        seed=seed,
        episodes_per_group=episodes_per_group,
        val_groups=val_groups,
        test_groups=test_groups,
    )
    window_frames, ramp_start = _read_window_label_metadata(root, objective=objective)
    label_scheme = _read_label_scheme(root)
    is_boundary_scheme = label_scheme == BOUNDARY_LABEL_SCHEME and objective == "binary"
    boundary_excluded_episode_ids = _read_boundary_excluded_episode_ids(root) if is_boundary_scheme else ()

    if is_boundary_scheme:
        # The boundary scheme needs each episode's 4-episode task group to verify
        # copy provenance and the no-cross-group rule. Canonical contiguous
        # groups are the source of truth (the split manifest only shuffles them).
        canonical_groups = build_task_groups(
            episode_ids,
            episodes_per_group=BOUNDARY_GROUP,
            minimum_groups=1,
        )
        group_of: dict[int, tuple[tuple[int, ...], int]] = {}
        for group in canonical_groups:
            for position, episode_id in enumerate(group.episode_ids):
                group_of[episode_id] = (group.episode_ids, position)

    audits: dict[int, EpisodeAudit] = {}
    boundary_audits: dict[int, BoundaryEpisodeAudit] = {}
    for episode_id in episode_ids:
        episode_metadata = dataset_metadata.episodes[episode_id]
        expected_length = int(episode_metadata["length"])
        if is_boundary_scheme:
            group_episode_ids, group_position = group_of[episode_id]
            boundary_audit = audit_boundary_completion_episode_parquet(
                parquet_paths[episode_id],
                episode_id=episode_id,
                expected_length=expected_length,
                group_episode_ids=group_episode_ids,
                group_position=group_position,
                label_key=label_key,
                excluded_episode_ids=boundary_excluded_episode_ids,
            )
            boundary_audits[episode_id] = boundary_audit
            audits[episode_id] = EpisodeAudit(
                episode_id=episode_id,
                frame_count=boundary_audit.frame_count,
                positive_count=boundary_audit.positive_count,
                negative_count=boundary_audit.negative_count,
                task_index=boundary_audit.task_index,
            )
        elif window_frames is None:
            audit_fn = audit_episode_parquet if objective == "binary" else audit_progress_episode_parquet
            audits[episode_id] = audit_fn(
                parquet_paths[episode_id],
                episode_id=episode_id,
                expected_length=expected_length,
                label_key=label_key,
            )
        elif objective == "binary":
            audits[episode_id] = audit_episode_parquet(
                parquet_paths[episode_id],
                episode_id=episode_id,
                expected_length=expected_length,
                label_key=label_key,
                window_frames=window_frames,
            )
        else:
            assert ramp_start is not None
            audits[episode_id] = audit_window_progress_episode_parquet(
                parquet_paths[episode_id],
                episode_id=episode_id,
                expected_length=expected_length,
                label_key=label_key,
                window_frames=window_frames,
                ramp_start=ramp_start,
            )

    if objective == "progress":
        return CompletionDataInfo(
            manifest=manifest,
            episode_audits=audits,
            train_positive_count=None,
            train_negative_count=None,
            pos_weight=None,
        )

    train_ids = manifest.episode_ids("train")
    positive_count = sum(audits[episode_id].positive_count for episode_id in train_ids)
    negative_count = sum(audits[episode_id].negative_count for episode_id in train_ids)
    if positive_count <= 0:
        raise ValueError("train split has no positive completion labels")
    if is_boundary_scheme:
        # The boundary scheme mandates an unweighted BCE (pos_weight=1.0); the
        # config's ``bce_pos_weight_override`` supplies it, so the data-derived
        # imbalance weight is intentionally not used here.
        return CompletionDataInfo(
            manifest=manifest,
            episode_audits=audits,
            train_positive_count=positive_count,
            train_negative_count=negative_count,
            pos_weight=None,
            boundary_episode_audits=boundary_audits,
        )
    pos_weight = min(negative_count / positive_count, 50.0)
    return CompletionDataInfo(
        manifest=manifest,
        episode_audits=audits,
        train_positive_count=positive_count,
        train_negative_count=negative_count,
        pos_weight=pos_weight,
    )
