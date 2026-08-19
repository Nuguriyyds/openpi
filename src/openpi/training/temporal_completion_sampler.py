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
POSITIVE_PER_BATCH = 32
HARD_NEGATIVE_PER_BATCH = 16
ORDINARY_NEGATIVE_PER_BATCH = 16
TEMPORAL_TASK_INDICES = (0, 1, 2, 3)
DEFAULT_TICK_STRIDE_FRAMES = 15
HARD_NEGATIVE_OFFSETS = (1,)
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
    """Sample 32 positives with 16 fixed-hard and 16 ordinary pairs."""

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
        if seed < 0 or tick_stride_frames <= 0:
            raise ValueError("temporal completion sampler seed/stride must be positive")
        if num_replicas != 1 or rank != 0:
            raise ValueError("TemporalCompletionBatchSampler is single-process in this version")
        if not samples:
            raise ValueError("temporal completion sampling requires candidate rows")

        views = tuple(_normalise_sample(sample, index=index) for index, sample in enumerate(samples))
        self._validate_candidate_rows(views, tick_stride_frames=tick_stride_frames)
        positive_by_task: dict[int, list[int]] = {task: [] for task in TEMPORAL_TASK_INDICES}
        positive_by_episode: dict[tuple[Hashable, int], int] = {}
        hard_by_episode: dict[tuple[Hashable, int], int] = {}
        ordinary_by_episode: dict[tuple[Hashable, int], list[int]] = {}
        for view in views:
            key = (view.trajectory_id, view.task_index)
            if view.sample_kind == "positive":
                positive_by_task[view.task_index].append(view.index)
                if key in positive_by_episode:
                    raise ValueError(f"duplicate positive subtask episode {key!r}")
                positive_by_episode[key] = view.index
            elif view.sample_kind == "hard_negative":
                if key in hard_by_episode:
                    raise ValueError(f"duplicate hard negative subtask episode {key!r}")
                hard_by_episode[key] = view.index
            else:
                ordinary_by_episode.setdefault(key, []).append(view.index)

        for task, pool in positive_by_task.items():
            if len(pool) < 8:
                raise ValueError(f"task {task} needs at least 8 positive subtask episodes, got {len(pool)}")
            ordinary_count = sum(key[1] == task for key in ordinary_by_episode)
            if ordinary_count < 4:
                raise ValueError(f"task {task} ordinary pool has only {ordinary_count} episodes; need 4")
            hard_count = sum(key[1] == task for key in hard_by_episode)
            if hard_count < 8:
                raise ValueError(f"task {task} hard pool has only {hard_count} episodes; need 8")
        missing_hard = set(positive_by_episode) - set(hard_by_episode)
        if missing_hard:
            raise ValueError(f"every positive subtask episode needs its fixed hard negative: {sorted(missing_hard)!r}")
        orphan_hard = set(hard_by_episode) - set(positive_by_episode)
        if orphan_hard:
            raise ValueError("every hard negative must have a positive from the same subtask episode")
        orphan_ordinary = set(ordinary_by_episode) - set(positive_by_episode)
        if orphan_ordinary:
            raise ValueError("every ordinary negative must have a positive from the same subtask episode")

        positive_count = len(positive_by_episode)
        batches_per_epoch = math.ceil(positive_count / POSITIVE_PER_BATCH) if batches_per_epoch is None else batches_per_epoch
        if batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive")
        self._views = views
        self._positive_by_task = {task: np.asarray(pool, dtype=np.int64) for task, pool in positive_by_task.items()}
        self._positive_by_episode = positive_by_episode
        self._hard_by_episode = hard_by_episode
        self._ordinary_by_episode = {key: np.asarray(pool, dtype=np.int64) for key, pool in ordinary_by_episode.items()}
        self._episodes_by_task = {
            task: tuple(key for key in positive_by_episode if key[1] == task) for task in TEMPORAL_TASK_INDICES
        }
        self._seed = int(seed)
        self._batches_per_epoch = int(batches_per_epoch)
        self._epoch = 0
        self._skip_batches = 0

    @staticmethod
    def _validate_candidate_rows(views: Sequence[_SampleView], *, tick_stride_frames: int) -> None:
        row_keys: set[tuple[Hashable, int, int]] = set()
        positive_events: set[tuple[Hashable, int]] = set()
        hard_offsets = {offset * tick_stride_frames for offset in HARD_NEGATIVE_OFFSETS}
        for view in views:
            if view.task_index not in TEMPORAL_TASK_INDICES:
                raise ValueError(f"invalid temporal task_index {view.task_index}")
            row_key = (view.trajectory_id, view.task_index, view.logical_tick)
            if row_key in row_keys:
                raise ValueError(f"duplicate temporal candidate row for {row_key!r}")
            row_keys.add(row_key)
            episode_key = (view.trajectory_id, view.task_index)
            if view.sample_kind == "positive":
                if view.logical_tick != view.boundary_tick or episode_key in positive_events:
                    raise ValueError("each subtask episode must have exactly one positive endpoint")
                positive_events.add(episode_key)
            elif view.sample_kind == "hard_negative":
                if view.boundary_tick - view.logical_tick not in hard_offsets:
                    raise ValueError("hard negative must be exactly E-15")
            else:
                distance = view.boundary_tick - view.logical_tick
                if distance < 2 * tick_stride_frames or distance % tick_stride_frames:
                    raise ValueError("ordinary negative must be E-30, E-45, ...")

    @property
    def steps_per_epoch(self) -> int:
        return self._batches_per_epoch

    @property
    def batch_composition(self) -> dict[str, int]:
        return {"positive": 32, "hard_negative": 16, "ordinary_negative": 16}

    @property
    def pool_sizes(self) -> dict[str, int]:
        return {
            "positive": len(self._positive_by_episode),
            "hard_negative": len(self._hard_by_episode),
            "ordinary_negative": sum(len(pool) for pool in self._ordinary_by_episode.values()),
        }

    @property
    def events_without_local_hard(self) -> frozenset[CompletionEventKey]:
        return frozenset()

    def audit_batch(self, batch: Sequence[int]) -> TemporalBatchAudit:
        indices = [operator.index(index) for index in batch]
        if len(indices) != TEMPORAL_BATCH_SIZE or any(index < 0 or index >= len(self._views) for index in indices):
            raise ValueError("batch must contain exactly 64 valid temporal candidate indices")
        selected = [self._views[index] for index in indices]
        positives = [view for view in selected if view.sample_kind == "positive"]
        return TemporalBatchAudit(
            positive_count=len(positives),
            hard_negative_count=sum(view.sample_kind == "hard_negative" for view in selected),
            ordinary_negative_count=sum(view.sample_kind == "ordinary_negative" for view in selected),
            event_local_hard_count=sum(view.sample_kind == "hard_negative" for view in selected),
            same_task_fallback_hard_count=0,
            positive_task_counts=tuple(sum(view.task_index == task for view in positives) for task in TEMPORAL_TASK_INDICES),
        )

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch = int(epoch)
        self._skip_batches = 0

    def set_skip_batches(self, skip_batches: int) -> None:
        if skip_batches < 0 or skip_batches > self._batches_per_epoch:
            raise ValueError(f"skip_batches must be in [0, {self._batches_per_epoch}]")
        self._skip_batches = int(skip_batches)

    def state_dict(self) -> dict[str, int]:
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
        if int(state.get("seed", -1)) != self._seed or int(state.get("batches_per_epoch", -1)) != self._batches_per_epoch:
            raise ValueError("temporal sampler state does not match this sampler")
        self.set_epoch(int(state["epoch"]))
        self.set_skip_batches(int(state["skip_batches"]))

    def _rng(self, *, epoch: int, batch_index: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self._seed, epoch, batch_index]))

    def _make_batch(self, *, epoch: int, batch_index: int) -> list[int]:
        rng = self._rng(epoch=epoch, batch_index=batch_index)
        selected_pairs: list[tuple[int, str]] = []
        for task in TEMPORAL_TASK_INDICES:
            task_keys = list(self._episodes_by_task[task])
            ordinary_keys = [key for key in task_keys if key in self._ordinary_by_episode]
            ordinary_keys = [ordinary_keys[int(index)] for index in rng.choice(len(ordinary_keys), size=4, replace=False)]
            remaining_hard = [key for key in task_keys if key not in ordinary_keys]
            hard_keys = [remaining_hard[int(index)] for index in rng.choice(len(remaining_hard), size=4, replace=False)]
            selected_pairs.extend((self._positive_by_episode[key], "hard") for key in hard_keys)
            selected_pairs.extend((self._positive_by_episode[key], "ordinary") for key in ordinary_keys)

        indices: list[int] = []
        for positive_index, negative_kind in selected_pairs:
            view = self._views[positive_index]
            key = (view.trajectory_id, view.task_index)
            indices.append(positive_index)
            if negative_kind == "hard":
                indices.append(self._hard_by_episode[key])
            else:
                indices.append(int(rng.choice(self._ordinary_by_episode[key])))
        if len(indices) != TEMPORAL_BATCH_SIZE:
            raise AssertionError(f"internal temporal batch size error: {len(indices)}")
        rng.shuffle(indices)
        return [int(index) for index in indices]

    def __iter__(self) -> Iterator[list[int]]:
        epoch, skip = self._epoch, self._skip_batches
        self._epoch += 1
        self._skip_batches = 0
        for batch_index in range(skip, self._batches_per_epoch):
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
