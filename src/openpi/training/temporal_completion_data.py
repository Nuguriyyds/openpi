"""Fail-closed indexing for the 2 Hz temporal completion objective.

This module is deliberately independent from the legacy per-frame completion
dataset.  It only describes immutable source references and labels; it never
edits, copies, or opens the source observations themselves.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
import dataclasses
import json
import os
import pathlib
import random
from typing import Any, Literal

FPS = 30
TICK_STRIDE_FRAMES = 15
TASKS_PER_TRAJECTORY = 4
TEMPORAL_HISTORY_STEPS = 3
SPLIT_SEED = 42
MANIFEST_SCHEMA_VERSION = 4
SPLIT_RATIOS: Mapping[str, float] = {"train": 0.72, "val": 0.08, "test": 0.20}

SplitName = Literal["train", "val", "test"]
SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")
SampleKind = Literal["positive", "hard_negative", "ordinary_negative"]
MappingStatus = Literal["matched", "subtask_only", "full_only", "ambiguous"]
TrajectorySource = Literal["subtask_logical", "full_identity"]


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"{context} fields do not match schema; missing={missing}, unexpected={unexpected}")


def ceil_to_tick(frame_position: int, *, stride: int = TICK_STRIDE_FRAMES) -> int:
    """Returns the first global tick at or after an exclusive frame boundary."""

    if frame_position < 0:
        raise ValueError(f"frame_position must be non-negative, got {frame_position}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    return ((int(frame_position) + stride - 1) // stride) * stride


def positive_is_reachable(positive_tick: int, previous_switch_tick: int | None) -> bool:
    """Whether three same-prompt 2 Hz features exist at ``positive_tick``.

    Task 0 starts with a feature at tick zero.  A later task switches after the
    previous positive tick, so its first feature is one tick later.
    """

    if previous_switch_tick is None:
        return positive_tick >= 2 * TICK_STRIDE_FRAMES
    return positive_tick - previous_switch_tick >= 3 * TICK_STRIDE_FRAMES


@dataclasses.dataclass(frozen=True)
class SubtaskEpisodeRecord:
    episode_id: int
    task_index: int
    length: int

    def __post_init__(self) -> None:
        if self.episode_id < 0:
            raise ValueError("subtask episode_id must be non-negative")
        if self.task_index not in range(TASKS_PER_TRAJECTORY):
            raise ValueError(f"subtask {self.episode_id} has invalid task_index {self.task_index}")
        if self.length <= 0:
            raise ValueError(f"subtask {self.episode_id} is empty or has invalid length {self.length}")

    @property
    def end_frame(self) -> int:
        """Inclusive artificial-human completion frame for this subtask.

        The audited LeRobot parquet uses contiguous frame indices ``0..L-1``;
        therefore the last row is the inclusive endpoint ``E=L-1``.  Keeping
        this conversion in the record prevents the temporal sampler from
        accidentally treating the exclusive length as the completion frame.
        """

        return self.length - 1


@dataclasses.dataclass(frozen=True)
class FullEpisodeRecord:
    episode_id: int
    length: int

    def __post_init__(self) -> None:
        if self.episode_id < 0:
            raise ValueError("full episode_id must be non-negative")
        if self.length <= 0:
            raise ValueError(f"full episode {self.episode_id} is empty or has invalid length {self.length}")


@dataclasses.dataclass(frozen=True)
class SubtaskGroupRecord:
    group_id: int
    source_episode_ids: tuple[int, int, int, int]
    task_indices: tuple[int, int, int, int]
    lengths: tuple[int, int, int, int]
    boundaries: tuple[int, int, int, int]
    positive_ticks: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if self.group_id < 0:
            raise ValueError("group_id must be non-negative")
        expected_episode_ids = tuple(self.group_id * TASKS_PER_TRAJECTORY + i for i in range(4))
        if self.source_episode_ids != expected_episode_ids:
            raise ValueError(
                f"group {self.group_id} episode order must be {expected_episode_ids}, got {self.source_episode_ids}"
            )
        if self.task_indices != (0, 1, 2, 3):
            raise ValueError(f"group {self.group_id} task order must be (0, 1, 2, 3), got {self.task_indices}")
        if any(length <= 0 for length in self.lengths):
            raise ValueError(f"group {self.group_id} contains an empty subtask: lengths={self.lengths}")
        expected_boundaries: list[int] = []
        total = 0
        for length in self.lengths:
            total += length
            expected_boundaries.append(total)
        if self.boundaries != tuple(expected_boundaries):
            raise ValueError(
                f"group {self.group_id} has inconsistent exclusive boundaries: "
                f"expected={tuple(expected_boundaries)}, got={self.boundaries}"
            )
        # ``positive_ticks`` is retained as a manifest field for compatibility,
        # but in the subtask protocol it stores each episode's inclusive E,
        # not a cumulative/global tick.
        expected_ticks = tuple(length - 1 for length in self.lengths)
        if self.positive_ticks != expected_ticks:
            raise ValueError(
                f"group {self.group_id} has inconsistent positive ticks: expected={expected_ticks}, "
                f"got={self.positive_ticks}"
            )

    @classmethod
    def from_episodes(cls, group_id: int, episodes: Sequence[SubtaskEpisodeRecord]) -> SubtaskGroupRecord:
        if len(episodes) != TASKS_PER_TRAJECTORY:
            raise ValueError(f"group {group_id} must contain exactly four subtask episodes, got {len(episodes)}")
        source_episode_ids = tuple(episode.episode_id for episode in episodes)
        task_indices = tuple(episode.task_index for episode in episodes)
        lengths = tuple(episode.length for episode in episodes)
        boundaries: list[int] = []
        total = 0
        for length in lengths:
            total += length
            boundaries.append(total)
        return cls(
            group_id=group_id,
            source_episode_ids=source_episode_ids,  # type: ignore[arg-type]
            task_indices=task_indices,  # type: ignore[arg-type]
            lengths=lengths,  # type: ignore[arg-type]
            boundaries=tuple(boundaries),  # type: ignore[arg-type]
            positive_ticks=tuple(episode.end_frame for episode in episodes),  # type: ignore[arg-type]
        )


def build_subtask_groups(episodes: Sequence[SubtaskEpisodeRecord]) -> tuple[SubtaskGroupRecord, ...]:
    """Audits the exact ``4g + task`` layout and returns immutable groups."""

    if not episodes:
        raise ValueError("subtask dataset contains no episodes")
    by_id: dict[int, SubtaskEpisodeRecord] = {}
    for episode in episodes:
        if episode.episode_id in by_id:
            raise ValueError(f"subtask dataset contains duplicate episode_id {episode.episode_id}")
        by_id[episode.episode_id] = episode
    sorted_ids = sorted(by_id)
    expected_ids = list(range(sorted_ids[-1] + 1))
    if sorted_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(sorted_ids))
        raise ValueError(f"subtask episode IDs must be exactly 0..{sorted_ids[-1]}; missing={missing}")
    if len(sorted_ids) % TASKS_PER_TRAJECTORY:
        raise ValueError(f"subtask episode count {len(sorted_ids)} cannot form complete four-stage trajectories")
    return tuple(
        SubtaskGroupRecord.from_episodes(
            group_id,
            [by_id[group_id * 4 + task_index] for task_index in range(4)],
        )
        for group_id in range(len(sorted_ids) // 4)
    )


def reachability_exclusion_reasons(group: SubtaskGroupRecord) -> tuple[str, ...]:
    """Returns task-specific reasons that require quarantining the whole group."""

    reasons: list[str] = []
    previous_tick: int | None = None
    for task_index, positive_tick in enumerate(group.positive_ticks):
        if not positive_is_reachable(positive_tick, previous_tick):
            reasons.append(f"unreachable_positive_after_history_reset:task={task_index}")
        previous_tick = positive_tick
    return tuple(reasons)


def require_reachable(group: SubtaskGroupRecord) -> None:
    reasons = reachability_exclusion_reasons(group)
    if reasons:
        raise ValueError(f"group {group.group_id} is quarantined: {','.join(reasons)}")


def global_feature_ticks(group: SubtaskGroupRecord, task_index: int) -> tuple[int, ...]:
    """Legacy logical-trajectory helper; not used by subtask temporal training."""

    if task_index not in range(TASKS_PER_TRAJECTORY):
        raise ValueError(f"task_index must be in [0, 3], got {task_index}")
    # Preserve the old virtual-concatenation helper for callers that still
    # inspect it, while the new sample builder uses local inclusive endpoints.
    start = 0 if task_index == 0 else ceil_to_tick(group.boundaries[task_index - 1]) + TICK_STRIDE_FRAMES
    stop = ceil_to_tick(group.boundaries[task_index])
    if start > stop:
        return ()
    return tuple(range(start, stop + 1, TICK_STRIDE_FRAMES))


def eligible_decision_ticks(group: SubtaskGroupRecord, task_index: int) -> tuple[int, ...]:
    feature_ticks = global_feature_ticks(group, task_index)
    if len(feature_ticks) < TEMPORAL_HISTORY_STEPS:
        return ()
    return feature_ticks[TEMPORAL_HISTORY_STEPS - 1 :]


@dataclasses.dataclass(frozen=True)
class SourceFrameReference:
    episode_id: int
    frame_index: int
    terminal_hold: bool


def map_logical_frame(group: SubtaskGroupRecord, logical_frame: int) -> SourceFrameReference:
    """Legacy virtual-concatenation helper; not used by subtask temporal training."""

    if logical_frame < 0:
        raise ValueError(f"logical_frame must be non-negative, got {logical_frame}")
    total_length = group.boundaries[-1]
    terminal_limit = ceil_to_tick(group.boundaries[-1])
    if logical_frame >= total_length:
        if logical_frame > terminal_limit:
            raise ValueError(f"logical frame {logical_frame} exceeds task-3 terminal hold limit {terminal_limit}")
        return SourceFrameReference(
            episode_id=group.source_episode_ids[-1],
            frame_index=group.lengths[-1] - 1,
            terminal_hold=True,
        )

    task_index = bisect.bisect_right(group.boundaries, logical_frame)
    task_start = 0 if task_index == 0 else group.boundaries[task_index - 1]
    return SourceFrameReference(
        episode_id=group.source_episode_ids[task_index],
        frame_index=logical_frame - task_start,
        terminal_hold=False,
    )


@dataclasses.dataclass(frozen=True)
class IdentityMatchRecord:
    group_id: int
    full_episode_id: int

    def __post_init__(self) -> None:
        if self.group_id < 0 or self.full_episode_id < 0:
            raise ValueError("identity match IDs must be non-negative")


@dataclasses.dataclass(frozen=True)
class IdentityAmbiguityRecord:
    """One explicitly quarantined many-to-many identity component.

    ``candidate_pairs`` records only relationships supported by the external
    identity audit.  The component itself owns every referenced group and full
    episode, which prevents an uncertain source identity from silently falling
    through to the ordinary unmatched buckets.
    """

    group_ids: tuple[int, ...]
    full_episode_ids: tuple[int, ...]
    candidate_pairs: tuple[tuple[int, int], ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.group_ids or not self.full_episode_ids or not self.candidate_pairs:
            raise ValueError("identity ambiguity requires groups, full episodes, and candidate pairs")
        if any(identifier < 0 for identifier in (*self.group_ids, *self.full_episode_ids)):
            raise ValueError("identity ambiguity IDs must be non-negative")
        if len(set(self.group_ids)) != len(self.group_ids):
            raise ValueError("identity ambiguity repeats a subtask group")
        if len(set(self.full_episode_ids)) != len(self.full_episode_ids):
            raise ValueError("identity ambiguity repeats a full episode")
        if len(set(self.candidate_pairs)) != len(self.candidate_pairs):
            raise ValueError("identity ambiguity repeats a candidate relationship")
        declared_groups = set(self.group_ids)
        declared_full = set(self.full_episode_ids)
        candidate_groups: set[int] = set()
        candidate_full: set[int] = set()
        for group_id, full_episode_id in self.candidate_pairs:
            if group_id not in declared_groups or full_episode_id not in declared_full:
                raise ValueError("identity ambiguity candidate relationship uses an undeclared ID")
            candidate_groups.add(group_id)
            candidate_full.add(full_episode_id)
        if candidate_groups != declared_groups or candidate_full != declared_full:
            raise ValueError("every ambiguous group and full episode must participate in a candidate relationship")
        if not self.reason.strip():
            raise ValueError("identity ambiguity reason must be non-empty")


@dataclasses.dataclass(frozen=True)
class TrajectoryIdentityRecord:
    trajectory_id: str
    mapping_status: MappingStatus
    group: SubtaskGroupRecord | None
    full_episode: FullEpisodeRecord | None
    exclusion_reason: str | None

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory_id must not be empty")
        if self.mapping_status not in ("matched", "subtask_only", "full_only", "ambiguous"):
            raise ValueError(f"unknown mapping_status {self.mapping_status!r}")
        if self.mapping_status == "matched":
            if self.group is None or self.full_episode is None:
                raise ValueError("matched identity requires a subtask group and full episode")
            if self.exclusion_reason is not None:
                raise ValueError("a matched identity cannot have an identity exclusion reason")
        else:
            if self.exclusion_reason is None:
                raise ValueError(f"{self.mapping_status} identity must state an exclusion_reason")
            if self.mapping_status == "subtask_only" and (self.group is None or self.full_episode is not None):
                raise ValueError("subtask_only identity must contain only a subtask group")
            if self.mapping_status == "full_only" and (self.group is not None or self.full_episode is None):
                raise ValueError("full_only identity must contain only a full episode")
            if self.mapping_status == "ambiguous":
                if (self.group is None) == (self.full_episode is None):
                    raise ValueError("ambiguous identity must contain exactly one subtask group or full episode")
                if not self.exclusion_reason.startswith("ambiguous_identity:"):
                    raise ValueError("ambiguous identity requires an explicit ambiguity exclusion reason")


def build_trajectory_identities(
    groups: Sequence[SubtaskGroupRecord],
    full_episodes: Sequence[FullEpisodeRecord],
    matches: Sequence[IdentityMatchRecord],
    ambiguities: Sequence[IdentityAmbiguityRecord] = (),
) -> tuple[TrajectoryIdentityRecord, ...]:
    """Builds a bijective mapping plus explicit quarantine rows."""

    group_by_id = {group.group_id: group for group in groups}
    full_by_id = {episode.episode_id: episode for episode in full_episodes}
    if len(group_by_id) != len(groups):
        raise ValueError("subtask identity input contains duplicate group IDs")
    if len(full_by_id) != len(full_episodes):
        raise ValueError("full identity input contains duplicate episode IDs")

    matched_group_ids: set[int] = set()
    matched_full_ids: set[int] = set()
    records: list[TrajectoryIdentityRecord] = []
    for match in matches:
        if match.group_id not in group_by_id:
            raise ValueError(f"identity mapping refers to unknown subtask group {match.group_id}")
        if match.full_episode_id not in full_by_id:
            raise ValueError(f"identity mapping refers to unknown full episode {match.full_episode_id}")
        if match.group_id in matched_group_ids:
            raise ValueError(f"subtask group {match.group_id} has multiple full-trajectory mappings")
        if match.full_episode_id in matched_full_ids:
            raise ValueError(f"full episode {match.full_episode_id} has multiple subtask-group mappings")
        matched_group_ids.add(match.group_id)
        matched_full_ids.add(match.full_episode_id)
        records.append(
            TrajectoryIdentityRecord(
                trajectory_id=f"full-{match.full_episode_id:06d}",
                mapping_status="matched",
                group=group_by_id[match.group_id],
                full_episode=full_by_id[match.full_episode_id],
                exclusion_reason=None,
            )
        )

    ambiguous_group_ids: set[int] = set()
    ambiguous_full_ids: set[int] = set()
    for ambiguity_index, ambiguity in enumerate(ambiguities, start=1):
        unknown_groups = set(ambiguity.group_ids) - set(group_by_id)
        if unknown_groups:
            raise ValueError(
                f"identity ambiguity {ambiguity_index} refers to unknown subtask groups {sorted(unknown_groups)}"
            )
        unknown_full = set(ambiguity.full_episode_ids) - set(full_by_id)
        if unknown_full:
            raise ValueError(
                f"identity ambiguity {ambiguity_index} refers to unknown full episodes {sorted(unknown_full)}"
            )
        matched_group_overlap = set(ambiguity.group_ids) & matched_group_ids
        if matched_group_overlap:
            raise ValueError(
                f"identity ambiguity {ambiguity_index} overlaps matched subtask groups {sorted(matched_group_overlap)}"
            )
        matched_full_overlap = set(ambiguity.full_episode_ids) & matched_full_ids
        if matched_full_overlap:
            raise ValueError(
                f"identity ambiguity {ambiguity_index} overlaps matched full episodes {sorted(matched_full_overlap)}"
            )
        ambiguity_group_overlap = set(ambiguity.group_ids) & ambiguous_group_ids
        if ambiguity_group_overlap:
            raise ValueError(
                f"identity ambiguity {ambiguity_index} repeats subtask groups from another ambiguity "
                f"{sorted(ambiguity_group_overlap)}"
            )
        ambiguity_full_overlap = set(ambiguity.full_episode_ids) & ambiguous_full_ids
        if ambiguity_full_overlap:
            raise ValueError(
                f"identity ambiguity {ambiguity_index} repeats full episodes from another ambiguity "
                f"{sorted(ambiguity_full_overlap)}"
            )
        ambiguous_group_ids.update(ambiguity.group_ids)
        ambiguous_full_ids.update(ambiguity.full_episode_ids)
        exclusion_reason = f"ambiguous_identity:{ambiguity.reason.strip()}"
        records.extend(
            TrajectoryIdentityRecord(
                trajectory_id=f"ambiguous-subtask-{group_id:06d}",
                mapping_status="ambiguous",
                group=group_by_id[group_id],
                full_episode=None,
                exclusion_reason=exclusion_reason,
            )
            for group_id in ambiguity.group_ids
        )
        records.extend(
            TrajectoryIdentityRecord(
                trajectory_id=f"ambiguous-full-{full_episode_id:06d}",
                mapping_status="ambiguous",
                group=None,
                full_episode=full_by_id[full_episode_id],
                exclusion_reason=exclusion_reason,
            )
            for full_episode_id in ambiguity.full_episode_ids
        )

    records.extend(
        TrajectoryIdentityRecord(
            trajectory_id=f"subtask-{group_id:06d}",
            mapping_status="subtask_only",
            group=group_by_id[group_id],
            full_episode=None,
            exclusion_reason="no_full_trajectory_mapping",
        )
        for group_id in sorted(set(group_by_id) - matched_group_ids - ambiguous_group_ids)
    )
    records.extend(
        TrajectoryIdentityRecord(
            trajectory_id=f"full-{full_episode_id:06d}",
            mapping_status="full_only",
            group=None,
            full_episode=full_by_id[full_episode_id],
            exclusion_reason="no_subtask_group_mapping",
        )
        for full_episode_id in sorted(set(full_by_id) - matched_full_ids - ambiguous_full_ids)
    )
    records.sort(key=lambda record: record.trajectory_id)
    return tuple(records)


@dataclasses.dataclass(frozen=True)
class SplitCounts:
    train: int
    val: int
    test: int

    @property
    def total(self) -> int:
        return self.train + self.val + self.test

    def to_dict(self) -> dict[str, int]:
        return {"train": self.train, "val": self.val, "test": self.test}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SplitCounts:
        _require_exact_keys(value, {"train", "val", "test"}, context="split_counts")
        return cls(train=int(value["train"]), val=int(value["val"]), test=int(value["test"]))


def compute_split_counts(trajectory_count: int) -> SplitCounts:
    """Uses the locked round-half-up 20%, then 10% of the remaining 80%."""

    if trajectory_count <= 0:
        raise ValueError("at least one eligible matched trajectory is required")
    test = (20 * trajectory_count + 50) // 100
    remaining = trajectory_count - test
    val = (10 * remaining + 50) // 100
    train = remaining - val
    if train <= 0:
        raise ValueError(f"{trajectory_count} trajectories leave no training split")
    return SplitCounts(train=train, val=val, test=test)


@dataclasses.dataclass(frozen=True)
class ManifestTrajectoryRecord:
    trajectory_id: str
    group_id: int | None
    source_episode_ids: tuple[int, ...]
    task_indices: tuple[int, ...]
    lengths: tuple[int, ...]
    boundaries: tuple[int, ...]
    positive_ticks: tuple[int, ...]
    split: SplitName | None
    full_episode_id: int | None
    full_length: int | None
    mapping_status: MappingStatus
    exclusion_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "group_id": self.group_id,
            "source_episode_ids": list(self.source_episode_ids),
            "task_indices": list(self.task_indices),
            "lengths": list(self.lengths),
            "boundaries": list(self.boundaries),
            "positive_ticks": list(self.positive_ticks),
            "split": self.split,
            "full_episode_id": self.full_episode_id,
            "full_length": self.full_length,
            "mapping_status": self.mapping_status,
            "exclusion_reason": self.exclusion_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ManifestTrajectoryRecord:
        fields = {
            "trajectory_id",
            "group_id",
            "source_episode_ids",
            "task_indices",
            "lengths",
            "boundaries",
            "positive_ticks",
            "split",
            "full_episode_id",
            "full_length",
            "mapping_status",
            "exclusion_reason",
        }
        _require_exact_keys(value, fields, context="manifest trajectory")
        return cls(
            trajectory_id=str(value["trajectory_id"]),
            group_id=None if value["group_id"] is None else int(value["group_id"]),
            source_episode_ids=tuple(int(item) for item in value["source_episode_ids"]),
            task_indices=tuple(int(item) for item in value["task_indices"]),
            lengths=tuple(int(item) for item in value["lengths"]),
            boundaries=tuple(int(item) for item in value["boundaries"]),
            positive_ticks=tuple(int(item) for item in value["positive_ticks"]),
            split=value["split"],
            full_episode_id=None if value["full_episode_id"] is None else int(value["full_episode_id"]),
            full_length=None if value["full_length"] is None else int(value["full_length"]),
            mapping_status=value["mapping_status"],
            exclusion_reason=value["exclusion_reason"],
        )

    def as_group(self) -> SubtaskGroupRecord:
        if self.group_id is None or len(self.source_episode_ids) != TASKS_PER_TRAJECTORY:
            raise ValueError(f"trajectory {self.trajectory_id} has no complete subtask group")
        return SubtaskGroupRecord(
            group_id=self.group_id,
            source_episode_ids=self.source_episode_ids,  # type: ignore[arg-type]
            task_indices=self.task_indices,  # type: ignore[arg-type]
            lengths=self.lengths,  # type: ignore[arg-type]
            boundaries=self.boundaries,  # type: ignore[arg-type]
            positive_ticks=self.positive_ticks,  # type: ignore[arg-type]
        )


@dataclasses.dataclass(frozen=True)
class TemporalCompletionManifest:
    source_subtask_repo_id: str
    source_subtask_root: str
    source_full_repo_id: str | None
    source_full_root: str | None
    task_prompts: tuple[str, str, str, str]
    split_counts: SplitCounts
    trajectories: tuple[ManifestTrajectoryRecord, ...]
    trajectory_source: TrajectorySource = "full_identity"
    schema_version: int = MANIFEST_SCHEMA_VERSION
    fps: int = FPS
    tick_stride_frames: int = TICK_STRIDE_FRAMES
    split_seed: int = SPLIT_SEED

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_subtask_repo_id": self.source_subtask_repo_id,
            "source_subtask_root": self.source_subtask_root,
            "source_full_repo_id": self.source_full_repo_id,
            "source_full_root": self.source_full_root,
            "trajectory_source": self.trajectory_source,
            "task_prompts": list(self.task_prompts),
            "fps": self.fps,
            "tick_stride_frames": self.tick_stride_frames,
            "split_seed": self.split_seed,
            "split_ratios": dict(SPLIT_RATIOS),
            "split_counts": self.split_counts.to_dict(),
            "trajectories": [trajectory.to_dict() for trajectory in self.trajectories],
        }

    def to_dict(self) -> dict[str, Any]:
        return self._payload()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TemporalCompletionManifest:
        fields = {
            "schema_version",
            "source_subtask_repo_id",
            "source_subtask_root",
            "source_full_repo_id",
            "source_full_root",
            "trajectory_source",
            "task_prompts",
            "fps",
            "tick_stride_frames",
            "split_seed",
            "split_ratios",
            "split_counts",
            "trajectories",
        }
        _require_exact_keys(value, fields, context="temporal completion manifest")
        if value["split_ratios"] != dict(SPLIT_RATIOS):
            raise ValueError(f"manifest split_ratios must be exactly {dict(SPLIT_RATIOS)}")
        manifest = cls(
            schema_version=int(value["schema_version"]),
            source_subtask_repo_id=str(value["source_subtask_repo_id"]),
            source_subtask_root=str(value["source_subtask_root"]),
            source_full_repo_id=None if value["source_full_repo_id"] is None else str(value["source_full_repo_id"]),
            source_full_root=None if value["source_full_root"] is None else str(value["source_full_root"]),
            trajectory_source=value["trajectory_source"],
            task_prompts=tuple(str(item) for item in value["task_prompts"]),  # type: ignore[arg-type]
            fps=int(value["fps"]),
            tick_stride_frames=int(value["tick_stride_frames"]),
            split_seed=int(value["split_seed"]),
            split_counts=SplitCounts.from_dict(value["split_counts"]),
            trajectories=tuple(ManifestTrajectoryRecord.from_dict(item) for item in value["trajectories"]),
        )
        validate_temporal_manifest(manifest)
        return manifest


def _manifest_record(
    identity: TrajectoryIdentityRecord,
    *,
    split: SplitName | None,
    exclusion_reason: str | None,
) -> ManifestTrajectoryRecord:
    group = identity.group
    full_episode = identity.full_episode
    return ManifestTrajectoryRecord(
        trajectory_id=identity.trajectory_id,
        group_id=None if group is None else group.group_id,
        source_episode_ids=() if group is None else group.source_episode_ids,
        task_indices=() if group is None else group.task_indices,
        lengths=() if group is None else group.lengths,
        boundaries=() if group is None else group.boundaries,
        positive_ticks=() if group is None else group.positive_ticks,
        split=split,
        full_episode_id=None if full_episode is None else full_episode.episode_id,
        full_length=None if full_episode is None else full_episode.length,
        mapping_status=identity.mapping_status,
        exclusion_reason=exclusion_reason,
    )


def create_temporal_manifest(
    identities: Sequence[TrajectoryIdentityRecord],
    *,
    source_subtask_repo_id: str,
    source_subtask_root: str | os.PathLike[str],
    source_full_repo_id: str | None,
    source_full_root: str | os.PathLike[str] | None,
    task_prompts: Sequence[str],
    trajectory_source: TrajectorySource = "full_identity",
    split_seed: int = SPLIT_SEED,
) -> TemporalCompletionManifest:
    """Seal the trajectory-level split without imposing temporal reachability.

    A complete breakfast group is only a split unit.  Completion candidates
    are built later per raw subtask episode, so a short episode may contribute
    no rows without quarantining its three siblings or changing the group
    split.
    """

    if split_seed != SPLIT_SEED:
        raise ValueError(f"temporal split_seed is locked to {SPLIT_SEED}, got {split_seed}")
    if not identities:
        raise ValueError("identity audit produced no records")
    task_prompts = tuple(str(prompt) for prompt in task_prompts)
    if len(task_prompts) != TASKS_PER_TRAJECTORY or any(not prompt.strip() for prompt in task_prompts):
        raise ValueError("temporal manifest requires exactly four non-empty ordered task prompts")
    if len(set(task_prompts)) != TASKS_PER_TRAJECTORY:
        raise ValueError("temporal manifest task prompts must be distinct")

    trajectory_ids = [identity.trajectory_id for identity in identities]
    if len(set(trajectory_ids)) != len(trajectory_ids):
        raise ValueError("identity audit contains duplicate trajectory IDs")

    eligible: list[TrajectoryIdentityRecord] = []
    exclusion_by_id: dict[str, str] = {}
    eligible_status = "matched" if trajectory_source == "full_identity" else "subtask_only"
    for identity in identities:
        if identity.mapping_status != eligible_status:
            exclusion_by_id[identity.trajectory_id] = identity.exclusion_reason or "unmatched_identity"
            continue
        assert identity.group is not None
        eligible.append(identity)

    counts = compute_split_counts(len(eligible))
    shuffled_ids = [identity.trajectory_id for identity in sorted(eligible, key=lambda item: item.trajectory_id)]
    random.Random(split_seed).shuffle(shuffled_ids)
    test_ids = set(shuffled_ids[: counts.test])
    val_ids = set(shuffled_ids[counts.test : counts.test + counts.val])
    split_by_id: dict[str, SplitName] = {
        **dict.fromkeys(test_ids, "test"),
        **dict.fromkeys(val_ids, "val"),
        **dict.fromkeys(shuffled_ids[counts.test + counts.val :], "train"),
    }
    records = tuple(
        _manifest_record(
            identity,
            split=split_by_id.get(identity.trajectory_id),
            exclusion_reason=exclusion_by_id.get(identity.trajectory_id),
        )
        for identity in sorted(identities, key=lambda item: item.trajectory_id)
    )
    manifest = TemporalCompletionManifest(
        source_subtask_repo_id=source_subtask_repo_id,
        source_subtask_root=str(pathlib.Path(source_subtask_root).resolve()),
        source_full_repo_id=source_full_repo_id,
        source_full_root=None if source_full_root is None else str(pathlib.Path(source_full_root).resolve()),
        task_prompts=task_prompts,  # type: ignore[arg-type]
        split_counts=counts,
        trajectories=records,
        trajectory_source=trajectory_source,
        split_seed=split_seed,
    )
    validate_temporal_manifest(manifest)
    return manifest


def create_subtask_temporal_manifest(
    groups: Sequence[SubtaskGroupRecord],
    *,
    source_subtask_repo_id: str,
    source_subtask_root: str | os.PathLike[str],
    task_prompts: Sequence[str],
    split_seed: int = SPLIT_SEED,
) -> TemporalCompletionManifest:
    """Seals a leakage-safe split over logical four-subtask trajectories.

    This training-only mode deliberately does not claim correspondence with a
    real full-video episode.  ``group_id`` is used as the stable trajectory
    grouping key when sample rows are materialized.
    """

    identities = tuple(
        TrajectoryIdentityRecord(
            trajectory_id=f"subtask-{group.group_id:06d}",
            mapping_status="subtask_only",
            group=group,
            full_episode=None,
            exclusion_reason="full_identity_not_required_for_logical_training",
        )
        for group in groups
    )
    return create_temporal_manifest(
        identities,
        source_subtask_repo_id=source_subtask_repo_id,
        source_subtask_root=source_subtask_root,
        source_full_repo_id=None,
        source_full_root=None,
        task_prompts=task_prompts,
        trajectory_source="subtask_logical",
        split_seed=split_seed,
    )


def validate_temporal_manifest(manifest: TemporalCompletionManifest) -> None:
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version {manifest.schema_version} is unsupported; expected {MANIFEST_SCHEMA_VERSION}"
        )
    if manifest.fps != FPS or manifest.tick_stride_frames != TICK_STRIDE_FRAMES:
        raise ValueError(f"manifest timing must be {FPS} fps with stride {TICK_STRIDE_FRAMES}")
    if manifest.split_seed != SPLIT_SEED:
        raise ValueError(f"manifest split_seed must be {SPLIT_SEED}")
    if not manifest.source_subtask_repo_id:
        raise ValueError("manifest source_subtask_repo_id must not be empty")
    if manifest.trajectory_source not in ("subtask_logical", "full_identity"):
        raise ValueError(f"manifest has invalid trajectory_source {manifest.trajectory_source!r}")
    if manifest.trajectory_source == "full_identity":
        if not manifest.source_full_repo_id or not manifest.source_full_root:
            raise ValueError("full_identity manifest requires a full dataset source")
    elif manifest.source_full_repo_id is not None or manifest.source_full_root is not None:
        raise ValueError("subtask_logical manifest must not bind a full dataset source")
    if len(manifest.task_prompts) != TASKS_PER_TRAJECTORY or any(
        not prompt.strip() for prompt in manifest.task_prompts
    ):
        raise ValueError("manifest must seal exactly four non-empty ordered task prompts")
    if len(set(manifest.task_prompts)) != TASKS_PER_TRAJECTORY:
        raise ValueError("manifest task prompts must be distinct")
    if not pathlib.Path(manifest.source_subtask_root).is_absolute():
        raise ValueError("manifest source_subtask_root must be absolute")
    if manifest.source_full_root is not None and not pathlib.Path(manifest.source_full_root).is_absolute():
        raise ValueError("manifest source_full_root must be absolute")
    seen_trajectory_ids: set[str] = set()
    seen_group_ids: set[int] = set()
    seen_full_ids: set[int] = set()
    source_episodes_by_split: dict[SplitName, set[int]] = {split: set() for split in SPLIT_NAMES}
    actual_counts = dict.fromkeys(SPLIT_NAMES, 0)
    for record in manifest.trajectories:
        if record.mapping_status not in ("matched", "subtask_only", "full_only", "ambiguous"):
            raise ValueError(f"trajectory {record.trajectory_id} has invalid mapping_status")
        if record.trajectory_id in seen_trajectory_ids:
            raise ValueError(f"manifest contains duplicate trajectory_id {record.trajectory_id}")
        seen_trajectory_ids.add(record.trajectory_id)
        if record.group_id is not None:
            if record.group_id in seen_group_ids:
                raise ValueError(f"manifest contains duplicate subtask group {record.group_id}")
            seen_group_ids.add(record.group_id)
            group = record.as_group()
        else:
            group = None
            if any(
                (
                    record.source_episode_ids,
                    record.task_indices,
                    record.lengths,
                    record.boundaries,
                    record.positive_ticks,
                )
            ):
                raise ValueError(f"trajectory {record.trajectory_id} has group fields without a group_id")
        if record.full_episode_id is not None:
            if record.full_episode_id in seen_full_ids:
                raise ValueError(f"manifest contains duplicate full episode {record.full_episode_id}")
            seen_full_ids.add(record.full_episode_id)
            if record.full_length is None or record.full_length <= 0:
                raise ValueError(f"trajectory {record.trajectory_id} has invalid full_length")
        elif record.full_length is not None:
            raise ValueError(f"trajectory {record.trajectory_id} has full_length without full_episode_id")

        if manifest.trajectory_source == "subtask_logical" and record.mapping_status == "subtask_only":
            if group is None or record.full_episode_id is not None:
                raise ValueError(f"logical trajectory {record.trajectory_id} has invalid identity fields")
            if record.split is None or record.exclusion_reason is not None:
                raise ValueError("subtask logical training groups cannot be quarantined by temporal reachability")
        elif record.mapping_status == "matched":
            if group is None or record.full_episode_id is None:
                raise ValueError(f"matched trajectory {record.trajectory_id} lacks group/full identity")
            if record.split is None or record.exclusion_reason is not None:
                raise ValueError("matched training groups cannot be quarantined by temporal reachability")
        else:
            if record.split is not None:
                raise ValueError(f"unmatched trajectory {record.trajectory_id} cannot enter split {record.split}")
            if record.exclusion_reason is None:
                raise ValueError(f"unmatched trajectory {record.trajectory_id} lacks an exclusion reason")
            if record.mapping_status == "subtask_only" and (group is None or record.full_episode_id is not None):
                raise ValueError(f"subtask_only trajectory {record.trajectory_id} has invalid identity fields")
            if record.mapping_status == "full_only" and (group is not None or record.full_episode_id is None):
                raise ValueError(f"full_only trajectory {record.trajectory_id} has invalid identity fields")
            if record.mapping_status == "ambiguous":
                if (group is None) == (record.full_episode_id is None):
                    raise ValueError(
                        f"ambiguous trajectory {record.trajectory_id} must contain exactly one identity side"
                    )
                if not record.exclusion_reason.startswith("ambiguous_identity:"):
                    raise ValueError(f"ambiguous trajectory {record.trajectory_id} lacks an explicit ambiguity reason")

        if record.split is not None:
            if record.split not in SPLIT_NAMES:
                raise ValueError(f"trajectory {record.trajectory_id} has invalid split {record.split!r}")
            actual_counts[record.split] += 1
            assert group is not None
            overlap = source_episodes_by_split[record.split].intersection(group.source_episode_ids)
            if overlap:
                raise ValueError(f"source episodes are duplicated within split {record.split}: {sorted(overlap)}")
            source_episodes_by_split[record.split].update(group.source_episode_ids)

    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1 :]:
            overlap = source_episodes_by_split[left].intersection(source_episodes_by_split[right])
            if overlap:
                raise ValueError(f"source episodes leak across {left}/{right}: {sorted(overlap)}")
    if actual_counts != manifest.split_counts.to_dict():
        raise ValueError(
            f"manifest split counts do not match records: declared={manifest.split_counts.to_dict()}, "
            f"actual={actual_counts}"
        )
    expected_counts = compute_split_counts(sum(actual_counts.values()))
    if expected_counts != manifest.split_counts:
        raise ValueError(
            f"manifest split counts violate round-half-up rule: expected={expected_counts.to_dict()}, "
            f"got={manifest.split_counts.to_dict()}"
        )
    eligible_ids = sorted(record.trajectory_id for record in manifest.trajectories if record.split is not None)
    random.Random(manifest.split_seed).shuffle(eligible_ids)
    expected_split_by_id: dict[str, SplitName] = {
        **dict.fromkeys(eligible_ids[: expected_counts.test], "test"),
        **dict.fromkeys(eligible_ids[expected_counts.test : expected_counts.test + expected_counts.val], "val"),
        **dict.fromkeys(eligible_ids[expected_counts.test + expected_counts.val :], "train"),
    }
    actual_split_by_id = {
        record.trajectory_id: record.split for record in manifest.trajectories if record.split is not None
    }
    if actual_split_by_id != expected_split_by_id:
        raise ValueError("manifest split assignment does not match the fixed-seed test/val/train shuffle")


def save_temporal_manifest(path: str | os.PathLike[str], manifest: TemporalCompletionManifest) -> None:
    """Writes a manifest atomically and refuses to rewrite sealed contents."""

    validate_temporal_manifest(manifest)
    output_path = pathlib.Path(path)
    if output_path.exists():
        existing = TemporalCompletionManifest.from_dict(json.loads(output_path.read_text(encoding="utf-8")))
        if existing == manifest:
            return
        raise FileExistsError(f"refusing to rewrite sealed temporal manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, output_path)


def load_temporal_manifest(path: str | os.PathLike[str]) -> TemporalCompletionManifest:
    """Loads and validates a temporal completion manifest."""

    return TemporalCompletionManifest.from_dict(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


@dataclasses.dataclass(frozen=True)
class TemporalSampleRow:
    trajectory_id: str
    full_episode_id: int
    task_index: int
    split: SplitName
    logical_tick: int
    label: int
    sample_kind: SampleKind
    boundary_tick: int
    prompt_index: int
    history_logical_ticks: tuple[int, int, int]
    source_episode_ids: tuple[int, int, int]
    source_frame_indices: tuple[int, int, int]
    terminal_hold_flags: tuple[bool, bool, bool]

    @property
    def subtask_episode_id(self) -> int:
        """The single raw subtask episode owning all three source frames."""

        return self.source_episode_ids[0]

    def __post_init__(self) -> None:
        if not self.trajectory_id or self.full_episode_id < 0:
            raise ValueError("sample requires a valid trajectory and full episode identity")
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"sample has invalid split {self.split!r}")
        if self.sample_kind not in ("positive", "hard_negative", "ordinary_negative"):
            raise ValueError(f"sample has invalid sample_kind {self.sample_kind!r}")
        if self.task_index not in range(TASKS_PER_TRAJECTORY) or self.prompt_index != self.task_index:
            raise ValueError("sample prompt_index must equal its logical task_index")
        if any(
            len(values) != TEMPORAL_HISTORY_STEPS
            for values in (
                self.history_logical_ticks,
                self.source_episode_ids,
                self.source_frame_indices,
                self.terminal_hold_flags,
            )
        ):
            raise ValueError("sample history and all source-reference fields must contain exactly three values")
        if any(
            value < 0 for value in (*self.history_logical_ticks, *self.source_episode_ids, *self.source_frame_indices)
        ):
            raise ValueError("sample history/source indices must be non-negative")
        if self.boundary_tick < 0 or self.logical_tick > self.boundary_tick:
            raise ValueError("sample logical_tick must not occur after its non-negative boundary_tick")
        if self.label not in (0, 1):
            raise ValueError("sample label must be binary")
        if (self.sample_kind == "positive") != (self.label == 1):
            raise ValueError("only positive sample_kind may carry label 1")
        if self.history_logical_ticks[2] != self.logical_tick:
            raise ValueError("sample history must end at logical_tick")
        gaps = tuple(
            right - left
            for left, right in zip(self.history_logical_ticks[:-1], self.history_logical_ticks[1:], strict=True)
        )
        if gaps != (TICK_STRIDE_FRAMES, TICK_STRIDE_FRAMES):
            raise ValueError(f"sample history gaps must be (15, 15), got {gaps}")
        if len(set(self.source_episode_ids)) != 1:
            raise ValueError("all three temporal references must belong to one subtask episode")
        if any(frame > self.boundary_tick for frame in self.source_frame_indices):
            raise ValueError("temporal source frame exceeds the subtask completion frame")
        distance = self.boundary_tick - self.logical_tick
        if self.sample_kind == "positive":
            if self.logical_tick != self.boundary_tick or self.label != 1:
                raise ValueError("positive sample must be the inclusive endpoint and have label 1")
        elif self.sample_kind == "hard_negative":
            if distance != TICK_STRIDE_FRAMES or self.label != 0:
                raise ValueError("hard negative must be exactly 15 frames before E and have label 0")
        elif self.label != 0 or distance < 2 * TICK_STRIDE_FRAMES or distance % TICK_STRIDE_FRAMES:
            raise ValueError("ordinary negative must be E-30, E-45, ... and have label 0")
        expected_history = (
            self.logical_tick - 2 * TICK_STRIDE_FRAMES,
            self.logical_tick - TICK_STRIDE_FRAMES,
            self.logical_tick,
        )
        if self.history_logical_ticks != expected_history:
            raise ValueError(f"history must be [t-30, t-15, t], got {self.history_logical_ticks}")
        if self.source_frame_indices != expected_history:
            raise ValueError("source frame indices must equal the local subtask history frames")
        if any(self.terminal_hold_flags):
            raise ValueError("subtask temporal samples do not use terminal holds")


def build_temporal_sample_rows(
    group: SubtaskGroupRecord,
    *,
    trajectory_id: str,
    full_episode_id: int,
    split: SplitName,
) -> tuple[TemporalSampleRow, ...]:
    """Builds reverse-sampled candidates independently for each subtask.

    ``group`` is used only to carry the split/trajectory identity.  Images and
    history are always addressed in one raw episode; no logical concatenation
    or prompt crossing is possible in this indexer.
    """

    if split not in SPLIT_NAMES:
        raise ValueError(f"invalid split {split!r}")
    rows: list[TemporalSampleRow] = []
    for task_index in range(TASKS_PER_TRAJECTORY):
        episode_id = group.source_episode_ids[task_index]
        end_frame = group.lengths[task_index] - 1
        if end_frame < 3 * TICK_STRIDE_FRAMES:
            continue

        def add_row(
            current_frame: int,
            sample_kind: SampleKind,
            label: int,
            *,
            _task_index: int = task_index,
            _end_frame: int = end_frame,
            _episode_id: int = episode_id,
        ) -> None:
            history_frames = (
                current_frame - 2 * TICK_STRIDE_FRAMES,
                current_frame - TICK_STRIDE_FRAMES,
                current_frame,
            )
            rows.append(
                TemporalSampleRow(
                    trajectory_id=trajectory_id,
                    full_episode_id=full_episode_id,
                    task_index=_task_index,
                    split=split,
                    logical_tick=current_frame,
                    label=label,
                    sample_kind=sample_kind,
                    boundary_tick=_end_frame,
                    prompt_index=_task_index,
                    history_logical_ticks=history_frames,
                    source_episode_ids=(_episode_id, _episode_id, _episode_id),
                    source_frame_indices=history_frames,
                    terminal_hold_flags=(False, False, False),
                )
            )

        # Exactly one endpoint positive and one fixed E-15 hard negative.
        add_row(end_frame, "positive", 1)
        add_row(end_frame - TICK_STRIDE_FRAMES, "hard_negative", 0)

        # Ordinary negatives walk backwards from E-30.  Each candidate has its
        # own complete local three-frame history and is sampled afresh by the
        # training sampler.
        current_frame = end_frame - 2 * TICK_STRIDE_FRAMES
        while current_frame >= 2 * TICK_STRIDE_FRAMES:
            add_row(current_frame, "ordinary_negative", 0)
            current_frame -= TICK_STRIDE_FRAMES
    return tuple(rows)


def build_manifest_sample_rows(manifest: TemporalCompletionManifest, split: SplitName) -> tuple[TemporalSampleRow, ...]:
    """Builds natural candidates without resampling or crossing trajectory splits."""

    validate_temporal_manifest(manifest)
    if split not in SPLIT_NAMES:
        raise ValueError(f"invalid split {split!r}")
    rows: list[TemporalSampleRow] = []
    for record in manifest.trajectories:
        if record.split != split:
            continue
        trajectory_numeric_id = record.full_episode_id if record.full_episode_id is not None else record.group_id
        assert trajectory_numeric_id is not None
        rows.extend(
            build_temporal_sample_rows(
                record.as_group(),
                trajectory_id=record.trajectory_id,
                full_episode_id=trajectory_numeric_id,
                split=split,
            )
        )
    return tuple(rows)
