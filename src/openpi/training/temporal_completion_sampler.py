"""Deterministic batch samplers for temporal completion features.

The training sampler consumes only candidate-row metadata.  It deliberately
does not know how observations or cached prefix features are stored, which
keeps the temporal label/index builder and the data loader loosely coupled.
Rows may be dataclass-like objects or mappings, as long as they expose the
fields in :class:`TemporalCompletionSampleMetadata`.
"""

from collections.abc import Hashable, Iterator, Mapping, Sequence
import dataclasses
import math
import operator
from typing import Literal, Protocol, TypeAlias

import numpy as np

TemporalSampleKind: TypeAlias = Literal["positive", "hard_negative", "ordinary_negative"]

TEMPORAL_BATCH_SIZE = 64
POSITIVE_PER_BATCH = 21
HARD_NEGATIVE_PER_BATCH = 21
ORDINARY_NEGATIVE_PER_BATCH = 22
TEMPORAL_TASK_INDICES = (0, 1, 2, 3)
DEFAULT_TICK_STRIDE_FRAMES = 15
HARD_NEGATIVE_OFFSETS = (1, 2, 3, 4)
_STATE_SCHEMA_VERSION = 1


class TemporalCompletionSampleMetadata(Protocol):
    """Minimum candidate-row interface required by the train sampler."""

    trajectory_id: Hashable
    task_index: int
    split: str
    logical_tick: int
    label: int
    sample_kind: TemporalSampleKind
    boundary_tick: int


TemporalSampleLike: TypeAlias = TemporalCompletionSampleMetadata | Mapping[str, object]


@dataclasses.dataclass(frozen=True)
class CompletionEventKey:
    """Stable identity of one unique completion event."""

    trajectory_id: Hashable
    task_index: int
    boundary_tick: int


@dataclasses.dataclass(frozen=True)
class TemporalBatchAudit:
    """Observable composition of one generated batch."""

    positive_count: int
    hard_negative_count: int
    ordinary_negative_count: int
    event_local_hard_count: int
    same_task_fallback_hard_count: int
    positive_task_counts: tuple[int, int, int, int]


@dataclasses.dataclass(frozen=True)
class _SampleView:
    index: int
    trajectory_id: Hashable
    task_index: int
    split: str
    logical_tick: int
    label: int
    sample_kind: TemporalSampleKind
    boundary_tick: int

    @property
    def event_key(self) -> CompletionEventKey:
        return CompletionEventKey(self.trajectory_id, self.task_index, self.boundary_tick)


def _read_field(sample: TemporalSampleLike, name: str, *, index: int) -> object:
    if isinstance(sample, Mapping):
        try:
            return sample[name]
        except KeyError as exc:
            raise ValueError(f"temporal sample {index} is missing required field {name!r}") from exc
    try:
        return getattr(sample, name)
    except AttributeError as exc:
        raise ValueError(f"temporal sample {index} is missing required field {name!r}") from exc


def _normalise_sample_kind(value: object, *, index: int) -> TemporalSampleKind:
    if not isinstance(value, str):
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, str):
            value = enum_value
    if value not in ("positive", "hard_negative", "ordinary_negative"):
        raise ValueError(f"temporal sample {index} has unsupported sample_kind {value!r}")
    return value


def _normalise_sample(sample: TemporalSampleLike, *, index: int) -> _SampleView:
    trajectory_id = _read_field(sample, "trajectory_id", index=index)
    try:
        hash(trajectory_id)
    except TypeError as exc:
        raise ValueError(f"temporal sample {index} trajectory_id must be hashable") from exc

    try:
        task_index = operator.index(_read_field(sample, "task_index", index=index))
        logical_tick = operator.index(_read_field(sample, "logical_tick", index=index))
        boundary_tick = operator.index(_read_field(sample, "boundary_tick", index=index))
        raw_label = float(_read_field(sample, "label", index=index))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"temporal sample {index} has a non-integral index/tick or non-numeric label") from exc

    if raw_label not in (0.0, 1.0):
        raise ValueError(f"temporal sample {index} label must be exactly 0 or 1, got {raw_label!r}")
    sample_kind = _normalise_sample_kind(_read_field(sample, "sample_kind", index=index), index=index)
    split = _read_field(sample, "split", index=index)
    if split != "train":
        raise ValueError(f"temporal train sampler accepts only split='train', got {split!r} at sample {index}")
    label = int(raw_label)
    if (sample_kind == "positive") != (label == 1):
        raise ValueError(f"temporal sample {index} has inconsistent label={label} and sample_kind={sample_kind!r}")

    return _SampleView(
        index=index,
        trajectory_id=trajectory_id,
        task_index=task_index,
        split=split,
        logical_tick=logical_tick,
        label=label,
        sample_kind=sample_kind,
        boundary_tick=boundary_tick,
    )


class TemporalCompletionBatchSampler:
    """Yields strict ``21 positive + 21 paired hard + 22 ordinary`` batches.

    Positive events are sampled without replacement *within* a batch and with
    replacement across batches.  For every selected positive, exactly one hard
    negative is drawn from the same ``(trajectory, task, boundary)`` event when
    one exists.  A reachable event with no earlier eligible decision tick uses
    a real hard negative from the same task in another trajectory.  The task
    receiving the sixth positive rotates over consecutive batches; the other
    tasks each receive five.

    Randomness is derived independently from ``(seed, epoch, batch_index)``.
    Consequently resume can jump directly to a batch without replaying earlier
    RNG draws.  This first implementation is intentionally single-process:
    callers must shard batches explicitly before enabling distributed loading.
    """

    def __init__(
        self,
        samples: Sequence[TemporalSampleLike],
        *,
        seed: int,
        batches_per_epoch: int | None = None,
        batch_size: int = TEMPORAL_BATCH_SIZE,
        tick_stride_frames: int = DEFAULT_TICK_STRIDE_FRAMES,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if batch_size != TEMPORAL_BATCH_SIZE:
            raise ValueError(f"temporal completion batch_size must be exactly {TEMPORAL_BATCH_SIZE}")
        if seed < 0:
            raise ValueError("temporal completion sampler seed must be non-negative")
        if tick_stride_frames <= 0:
            raise ValueError("tick_stride_frames must be positive")
        if num_replicas != 1 or rank != 0:
            raise ValueError(
                "TemporalCompletionBatchSampler is single-process in this version; "
                "explicitly shard batches before using num_replicas != 1"
            )
        if len(samples) == 0:
            raise ValueError("temporal completion sampling requires at least one candidate row")

        views = tuple(_normalise_sample(sample, index=index) for index, sample in enumerate(samples))
        self._validate_candidate_rows(views, tick_stride_frames=tick_stride_frames)

        positive_by_task: dict[int, list[int]] = {task_index: [] for task_index in TEMPORAL_TASK_INDICES}
        positive_event_by_index: dict[int, CompletionEventKey] = {}
        hard_by_event: dict[CompletionEventKey, list[int]] = {}
        hard_by_task: dict[int, list[int]] = {task_index: [] for task_index in TEMPORAL_TASK_INDICES}
        ordinary_by_trajectory: dict[Hashable, list[int]] = {}
        for view in views:
            if view.sample_kind == "positive":
                positive_by_task[view.task_index].append(view.index)
                positive_event_by_index[view.index] = view.event_key
            elif view.sample_kind == "hard_negative":
                hard_by_event.setdefault(view.event_key, []).append(view.index)
                hard_by_task[view.task_index].append(view.index)
            else:
                ordinary_by_trajectory.setdefault(view.trajectory_id, []).append(view.index)

        for task_index, pool in positive_by_task.items():
            required = 6
            if len(pool) < required:
                raise ValueError(
                    f"task {task_index} has only {len(pool)} unique positive events; "
                    f"at least {required} are required for a 5/5/5/6 batch"
                )
        missing_hard = [event for event in positive_event_by_index.values() if event not in hard_by_event]
        positive_events = set(positive_event_by_index.values())
        orphan_hard = [event for event in hard_by_event if event not in positive_events]
        if orphan_hard:
            raise ValueError(f"hard negative(s) have no matching positive event: {orphan_hard[:3]}")
        tasks_without_hard = [task_index for task_index, pool in hard_by_task.items() if not pool]
        if tasks_without_hard:
            raise ValueError(
                "temporal completion task(s) have no hard negatives for event-local pairing or same-task "
                f"fallback: {tasks_without_hard}"
            )
        if not ordinary_by_trajectory:
            raise ValueError("temporal completion sampling found no ordinary negatives")

        positive_count = len(positive_event_by_index)
        if batches_per_epoch is None:
            batches_per_epoch = math.ceil(positive_count / POSITIVE_PER_BATCH)
        if batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive")

        self._views = views
        self._positive_by_task = {
            task_index: np.asarray(pool, dtype=np.int64) for task_index, pool in positive_by_task.items()
        }
        self._positive_event_by_index = positive_event_by_index
        self._hard_by_event = {event: np.asarray(pool, dtype=np.int64) for event, pool in hard_by_event.items()}
        self._hard_by_task = {task_index: np.asarray(pool, dtype=np.int64) for task_index, pool in hard_by_task.items()}
        self._events_without_local_hard = frozenset(missing_hard)
        self._ordinary_trajectory_ids = tuple(ordinary_by_trajectory)
        self._ordinary_by_trajectory = {
            trajectory_id: np.asarray(pool, dtype=np.int64) for trajectory_id, pool in ordinary_by_trajectory.items()
        }
        self._seed = int(seed)
        self._batches_per_epoch = int(batches_per_epoch)
        self._epoch = 0
        self._skip_batches = 0

    @staticmethod
    def _validate_candidate_rows(views: Sequence[_SampleView], *, tick_stride_frames: int) -> None:
        row_keys: set[tuple[Hashable, int, int]] = set()
        positive_events: set[CompletionEventKey] = set()
        positive_trajectory_tasks: set[tuple[Hashable, int]] = set()
        hard_offsets = {offset * tick_stride_frames for offset in HARD_NEGATIVE_OFFSETS}
        for view in views:
            if view.task_index not in TEMPORAL_TASK_INDICES:
                raise ValueError(
                    f"temporal sample {view.index} task_index must be one of {TEMPORAL_TASK_INDICES}, "
                    f"got {view.task_index}"
                )
            row_key = (view.trajectory_id, view.task_index, view.logical_tick)
            if row_key in row_keys:
                raise ValueError(f"duplicate temporal candidate row for {row_key!r}")
            row_keys.add(row_key)

            if view.sample_kind == "positive":
                if view.logical_tick != view.boundary_tick:
                    raise ValueError(
                        f"positive sample {view.index} logical_tick must equal boundary_tick "
                        f"({view.logical_tick} != {view.boundary_tick})"
                    )
                if view.event_key in positive_events:
                    raise ValueError(f"duplicate positive completion event {view.event_key!r}")
                trajectory_task = (view.trajectory_id, view.task_index)
                if trajectory_task in positive_trajectory_tasks:
                    raise ValueError(f"multiple positive events for trajectory/task {trajectory_task!r}")
                positive_events.add(view.event_key)
                positive_trajectory_tasks.add(trajectory_task)
            elif view.sample_kind == "hard_negative":
                frame_offset = view.boundary_tick - view.logical_tick
                if frame_offset not in hard_offsets:
                    raise ValueError(
                        f"hard-negative sample {view.index} must be 1-4 ticks before its boundary; "
                        f"got frame offset {frame_offset}"
                    )
            else:
                frame_offset = view.boundary_tick - view.logical_tick
                if frame_offset == 0 or frame_offset in hard_offsets:
                    raise ValueError(f"ordinary-negative sample {view.index} overlaps the positive/hard-negative pool")

    @property
    def steps_per_epoch(self) -> int:
        return self._batches_per_epoch

    @property
    def batch_composition(self) -> dict[str, int]:
        return {
            "positive": POSITIVE_PER_BATCH,
            "hard_negative": HARD_NEGATIVE_PER_BATCH,
            "ordinary_negative": ORDINARY_NEGATIVE_PER_BATCH,
        }

    @property
    def pool_sizes(self) -> dict[str, int]:
        return {
            "positive": int(sum(pool.size for pool in self._positive_by_task.values())),
            "hard_negative": int(sum(pool.size for pool in self._hard_by_event.values())),
            "ordinary_negative": int(sum(pool.size for pool in self._ordinary_by_trajectory.values())),
        }

    @property
    def events_without_local_hard(self) -> frozenset[CompletionEventKey]:
        """Events that use a same-task hard negative from another trajectory."""

        return self._events_without_local_hard

    def audit_batch(self, batch: Sequence[int]) -> TemporalBatchAudit:
        """Summarizes kind/task balance and event-local versus fallback pairing."""

        try:
            indices = [operator.index(index) for index in batch]
        except TypeError as exc:
            raise ValueError("batch contains an invalid temporal candidate index") from exc
        if any(index < 0 or index >= len(self._views) for index in indices):
            raise ValueError("batch contains an invalid temporal candidate index")
        selected = [self._views[index] for index in indices]
        positives = [view for view in selected if view.sample_kind == "positive"]
        fallback_count = sum(view.event_key in self._events_without_local_hard for view in positives)
        return TemporalBatchAudit(
            positive_count=len(positives),
            hard_negative_count=sum(view.sample_kind == "hard_negative" for view in selected),
            ordinary_negative_count=sum(view.sample_kind == "ordinary_negative" for view in selected),
            event_local_hard_count=len(positives) - fallback_count,
            same_task_fallback_hard_count=fallback_count,
            positive_task_counts=tuple(
                sum(view.task_index == task_index for view in positives) for task_index in TEMPORAL_TASK_INDICES
            ),
        )

    def task_composition(self, batch_index: int, *, epoch: int | None = None) -> dict[int, int]:
        """Returns the positive task counts for one batch."""

        if batch_index < 0 or batch_index >= self._batches_per_epoch:
            raise ValueError(f"batch_index must be in [0, {self._batches_per_epoch})")
        selected_epoch = self._epoch if epoch is None else int(epoch)
        if selected_epoch < 0:
            raise ValueError("epoch must be non-negative")
        composition = dict.fromkeys(TEMPORAL_TASK_INDICES, 5)
        extra_task = (selected_epoch * self._batches_per_epoch + batch_index) % len(TEMPORAL_TASK_INDICES)
        composition[extra_task] += 1
        return composition

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = int(epoch)
        self._skip_batches = 0

    def set_skip_batches(self, skip_batches: int) -> None:
        """Skips already completed batches on the next iteration."""

        if skip_batches < 0 or skip_batches > self._batches_per_epoch:
            raise ValueError(f"skip_batches must be in [0, {self._batches_per_epoch}]")
        self._skip_batches = int(skip_batches)

    def state_dict(self) -> dict[str, int]:
        """Serializes the configuration of the next ``__iter__`` call.

        Trainers checkpointing in the middle of an epoch should first call
        ``set_epoch(current_epoch)`` and ``set_skip_batches(completed_batches)``
        so this cursor describes the desired resume point exactly.
        """

        return {
            "schema_version": _STATE_SCHEMA_VERSION,
            "seed": self._seed,
            "batches_per_epoch": self._batches_per_epoch,
            "epoch": self._epoch,
            "skip_batches": self._skip_batches,
        }

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        if int(state.get("schema_version", -1)) != _STATE_SCHEMA_VERSION:
            raise ValueError("unsupported temporal sampler state schema_version")
        if int(state.get("seed", -1)) != self._seed:
            raise ValueError("temporal sampler state seed does not match this sampler")
        if int(state.get("batches_per_epoch", -1)) != self._batches_per_epoch:
            raise ValueError("temporal sampler state batches_per_epoch does not match this sampler")
        self.set_epoch(int(state["epoch"]))
        self.set_skip_batches(int(state["skip_batches"]))

    def _rng(self, *, epoch: int, batch_index: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self._seed, epoch, batch_index]))

    def _sample_ordinary(self, rng: np.random.Generator) -> list[int]:
        trajectory_count = len(self._ordinary_trajectory_ids)
        selected_trajectory_offsets: list[int] = []
        while len(selected_trajectory_offsets) < ORDINARY_NEGATIVE_PER_BATCH:
            permutation = rng.permutation(trajectory_count).tolist()
            remaining = ORDINARY_NEGATIVE_PER_BATCH - len(selected_trajectory_offsets)
            selected_trajectory_offsets.extend(permutation[:remaining])

        return [
            int(rng.choice(self._ordinary_by_trajectory[self._ordinary_trajectory_ids[offset]]))
            for offset in selected_trajectory_offsets
        ]

    def _make_batch(self, *, epoch: int, batch_index: int) -> list[int]:
        rng = self._rng(epoch=epoch, batch_index=batch_index)
        task_composition = self.task_composition(batch_index, epoch=epoch)
        positive_indices: list[int] = []
        for task_index in TEMPORAL_TASK_INDICES:
            sampled = rng.choice(
                self._positive_by_task[task_index],
                size=task_composition[task_index],
                replace=False,
            )
            positive_indices.extend(int(index) for index in sampled)

        hard_indices: list[int] = []
        for index in positive_indices:
            event = self._positive_event_by_index[index]
            pool = self._hard_by_event.get(event)
            if pool is None:
                # An interval of exactly three 2 Hz ticks has a reachable
                # positive triplet but no earlier eligible decision tick.
                # Keep that positive and draw a real hard negative from the
                # same task in another training trajectory.
                pool = self._hard_by_task[event.task_index]
            hard_indices.append(int(rng.choice(pool)))
        ordinary_indices = self._sample_ordinary(rng)
        batch = np.asarray([*positive_indices, *hard_indices, *ordinary_indices], dtype=np.int64)
        if batch.size != TEMPORAL_BATCH_SIZE:
            raise AssertionError(f"internal temporal batch size error: {batch.size}")
        rng.shuffle(batch)
        return [int(index) for index in batch]

    def __iter__(self) -> Iterator[list[int]]:
        epoch = self._epoch
        skip_batches = self._skip_batches
        self._epoch += 1
        self._skip_batches = 0
        for batch_index in range(skip_batches, self._batches_per_epoch):
            yield self._make_batch(epoch=epoch, batch_index=batch_index)

    def __len__(self) -> int:
        return self._batches_per_epoch - self._skip_batches


class NaturalTemporalEvalBatchSampler:
    """Batches every eval candidate exactly once in its existing natural order."""

    def __init__(self, samples: Sequence[object], *, batch_size: int) -> None:
        if len(samples) == 0:
            raise ValueError("natural temporal evaluation requires at least one candidate row")
        if batch_size <= 0:
            raise ValueError("evaluation batch_size must be positive")
        self._sample_count = len(samples)
        self._batch_size = int(batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        for start in range(0, self._sample_count, self._batch_size):
            yield list(range(start, min(start + self._batch_size, self._sample_count)))

    def __len__(self) -> int:
        return math.ceil(self._sample_count / self._batch_size)
