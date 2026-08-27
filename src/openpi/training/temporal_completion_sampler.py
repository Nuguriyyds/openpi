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

TemporalSampleKind: TypeAlias = Literal["positive", "hard_negative", "ordinary_negative", "transition_negative"]

TEMPORAL_BATCH_SIZE = 64
POSITIVE_PER_BATCH = 32
HARD_NEGATIVE_PER_BATCH = 16
ORDINARY_NEGATIVE_PER_BATCH = 16
TEMPORAL_TASK_INDICES = (0, 1, 2, 3)
DEFAULT_TICK_STRIDE_FRAMES = 15
HARD_NEGATIVE_OFFSETS = (1,)
_STATE_SCHEMA_VERSION = 1
START_TERMINAL_WINDOW_VARIANTS = (
    "endpoint_positive",
    "terminal_one_hold_positive",
    "terminal_full_hold_positive",
    "hard_negative",
    "ordinary_negative",
    "start_0_negative",
    "start_15_negative",
)
START_TERMINAL_BATCH_COUNTS = {
    "endpoint_positive": 16,
    "terminal_one_hold_positive": 8,
    "terminal_full_hold_positive": 8,
    "hard_negative": 16,
    "ordinary_negative": 8,
    "start_0_negative": 4,
    "start_15_negative": 4,
}
_START_TERMINAL_STATE_SCHEMA_VERSION = 1


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
    transition_negative_count: int
    event_local_hard_count: int
    same_task_fallback_hard_count: int
    positive_task_counts: tuple[int, int, int, int]
    hard_task_counts: tuple[int, int, int, int]
    ordinary_task_counts: tuple[int, int, int, int]
    transition_task_counts: tuple[int, int, int]
    transition_step_counts: tuple[int, int]


@dataclasses.dataclass(frozen=True)
class StartTerminalBatchAudit:
    """Composition of one fixed start/terminal training batch.

    ``variant_counts`` and ``task_counts`` intentionally expose the seven
    pools independently, so a sampler audit cannot hide a pool behind the
    aggregate positive/negative totals used by the older protocol.
    """

    variant_counts: dict[str, int]
    task_counts: dict[str, tuple[int, int, int, int]]
    transition_negative_count: int = 0

    @property
    def pool_counts(self) -> dict[str, int]:
        return dict(self.variant_counts)

    @property
    def per_task_counts(self) -> dict[str, tuple[int, int, int, int]]:
        return dict(self.task_counts)


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
    window_variant: str = "base"

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


def _read_optional_field(sample: TemporalSampleLike, name: str, *, default: object) -> object:
    if isinstance(sample, Mapping):
        return sample.get(name, default)
    return getattr(sample, name, default)


def _normalise_sample_kind(value: object, *, index: int) -> TemporalSampleKind:
    if not isinstance(value, str):
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, str):
            value = enum_value
    if value not in ("positive", "hard_negative", "ordinary_negative", "transition_negative"):
        raise ValueError(f"temporal sample {index} has unsupported sample_kind {value!r}")
    return value


def _normalise_window_variant(value: object, *, index: int) -> str:
    if not isinstance(value, str):
        enum_value = getattr(value, "value", None)
        if isinstance(enum_value, str):
            value = enum_value
    allowed = {"base", *START_TERMINAL_WINDOW_VARIANTS, "transition_negative"}
    if value not in allowed:
        raise ValueError(f"temporal sample {index} has unsupported window_variant {value!r}")
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
    window_variant = _normalise_window_variant(
        _read_optional_field(sample, "window_variant", default="base"),
        index=index,
    )
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
        window_variant=window_variant,
    )


@dataclasses.dataclass(frozen=True)
class _StartTerminalSampleView:
    view: _SampleView
    source_episode_ids: tuple[int, int, int]
    source_frame_indices: tuple[int, int, int]
    prompt_index: int

    @property
    def event_key(self) -> CompletionEventKey:
        return self.view.event_key


def _normalise_triplet(value: object, *, name: str, index: int, cast: type = int) -> tuple[object, object, object]:
    try:
        values = tuple(cast(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"temporal sample {index} field {name!r} must be a three-value sequence") from exc
    if len(values) != 3:
        raise ValueError(f"temporal sample {index} field {name!r} must contain exactly three values")
    return values  # type: ignore[return-value]


def _normalise_start_terminal_sample(sample: TemporalSampleLike, *, index: int) -> _StartTerminalSampleView:
    view = _normalise_sample(sample, index=index)
    if view.window_variant not in START_TERMINAL_WINDOW_VARIANTS:
        raise ValueError(
            "start_terminal sampler accepts only the seven explicit window variants; "
            f"sample {index} has {view.window_variant!r}"
        )
    expected_kind = {
        "endpoint_positive": "positive",
        "terminal_one_hold_positive": "positive",
        "terminal_full_hold_positive": "positive",
        "hard_negative": "hard_negative",
        "ordinary_negative": "ordinary_negative",
        "start_0_negative": "ordinary_negative",
        "start_15_negative": "ordinary_negative",
    }[view.window_variant]
    if view.sample_kind != expected_kind:
        raise ValueError(
            f"start_terminal sample {index} variant {view.window_variant!r} has sample_kind {view.sample_kind!r}"
        )
    if view.boundary_tick < 0 or (
        view.window_variant not in ("terminal_one_hold_positive", "terminal_full_hold_positive")
        and view.logical_tick > view.boundary_tick
    ):
        raise ValueError(f"start_terminal sample {index} has an invalid logical/boundary tick")

    try:
        prompt_index = operator.index(_read_field(sample, "prompt_index", index=index))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"start_terminal sample {index} has a non-integral prompt_index") from exc
    if prompt_index != view.task_index:
        raise ValueError(f"start_terminal sample {index} prompt_index must equal task_index")
    source_episode_ids = _normalise_triplet(
        _read_field(sample, "source_episode_ids", index=index),
        name="source_episode_ids",
        index=index,
    )
    source_frame_indices = _normalise_triplet(
        _read_field(sample, "source_frame_indices", index=index),
        name="source_frame_indices",
        index=index,
    )
    history = _normalise_triplet(
        _read_field(sample, "history_logical_ticks", index=index),
        name="history_logical_ticks",
        index=index,
    )
    raw_flags = _normalise_triplet(
        _read_field(sample, "terminal_hold_flags", index=index),
        name="terminal_hold_flags",
        index=index,
        cast=bool,
    )
    source_episode_ids = tuple(int(value) for value in source_episode_ids)  # type: ignore[assignment]
    source_frame_indices = tuple(int(value) for value in source_frame_indices)  # type: ignore[assignment]
    history = tuple(int(value) for value in history)  # type: ignore[assignment]
    raw_flags = tuple(bool(value) for value in raw_flags)  # type: ignore[assignment]
    if any(value < 0 for value in (*source_episode_ids, *source_frame_indices, *history)):
        raise ValueError(f"start_terminal sample {index} contains negative source/history coordinates")
    if len(set(source_episode_ids)) != 1:
        raise ValueError(f"start_terminal sample {index} must use one subtask episode")
    if any(frame > view.boundary_tick for frame in source_frame_indices):
        raise ValueError(f"start_terminal sample {index} source frame exceeds its boundary")

    if view.window_variant == "terminal_one_hold_positive":
        expected_history = (
            view.boundary_tick - DEFAULT_TICK_STRIDE_FRAMES,
            view.boundary_tick,
            view.boundary_tick + DEFAULT_TICK_STRIDE_FRAMES,
        )
        expected_frames = (view.boundary_tick - DEFAULT_TICK_STRIDE_FRAMES, view.boundary_tick, view.boundary_tick)
        expected_flags = (False, False, True)
    elif view.window_variant == "terminal_full_hold_positive":
        expected_history = (
            view.boundary_tick,
            view.boundary_tick + DEFAULT_TICK_STRIDE_FRAMES,
            view.boundary_tick + 2 * DEFAULT_TICK_STRIDE_FRAMES,
        )
        expected_frames = (view.boundary_tick,) * 3
        expected_flags = (False, True, True)
    else:
        expected_history = (
            view.logical_tick - 2 * DEFAULT_TICK_STRIDE_FRAMES,
            view.logical_tick - DEFAULT_TICK_STRIDE_FRAMES,
            view.logical_tick,
        )
        expected_frames = expected_history
        expected_flags = (False, False, False)
    if history != expected_history or source_frame_indices != expected_frames or raw_flags != expected_flags:
        raise ValueError(f"start_terminal sample {index} does not match its fixed window variant")
    if history[-1] != view.logical_tick:
        raise ValueError(f"start_terminal sample {index} history must end at logical_tick")
    if view.window_variant == "start_0_negative" and (
        view.boundary_tick < 4 * DEFAULT_TICK_STRIDE_FRAMES
        or history != (0, DEFAULT_TICK_STRIDE_FRAMES, 2 * DEFAULT_TICK_STRIDE_FRAMES)
    ):
        raise ValueError("start_0_negative must be eligible at E>=60 and use [0, 15, 30]")
    if view.window_variant == "start_15_negative" and (
        view.boundary_tick < 5 * DEFAULT_TICK_STRIDE_FRAMES
        or history != (DEFAULT_TICK_STRIDE_FRAMES, 2 * DEFAULT_TICK_STRIDE_FRAMES, 3 * DEFAULT_TICK_STRIDE_FRAMES)
    ):
        raise ValueError("start_15_negative must be eligible at E>=75 and use [15, 30, 45]")
    if view.window_variant == "endpoint_positive" and view.logical_tick != view.boundary_tick:
        raise ValueError("endpoint_positive must end at its completion boundary")
    if view.window_variant == "hard_negative" and view.boundary_tick - view.logical_tick != DEFAULT_TICK_STRIDE_FRAMES:
        raise ValueError("hard_negative must be exactly E-15")
    if view.window_variant == "ordinary_negative":
        distance = view.boundary_tick - view.logical_tick
        if distance < 2 * DEFAULT_TICK_STRIDE_FRAMES or distance % DEFAULT_TICK_STRIDE_FRAMES:
            raise ValueError("ordinary_negative must be E-30, E-45, ...")
    return _StartTerminalSampleView(
        view=view,
        source_episode_ids=source_episode_ids,
        source_frame_indices=source_frame_indices,
        prompt_index=prompt_index,
    )


class TemporalCompletionBatchSampler:
    """Sample either the legacy local composition or history-carry batches."""

    def __init__(
        self,
        samples: Sequence[TemporalSampleLike],
        *,
        seed: int,
        batches_per_epoch: int | None = None,
        batch_size: int = TEMPORAL_BATCH_SIZE,
        tick_stride_frames: int = DEFAULT_TICK_STRIDE_FRAMES,
        positive_per_batch: int = POSITIVE_PER_BATCH,
        hard_negative_per_batch: int = HARD_NEGATIVE_PER_BATCH,
        ordinary_negative_per_batch: int = ORDINARY_NEGATIVE_PER_BATCH,
        transition_negative_per_batch: int = 0,
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
        counts = (
            int(positive_per_batch),
            int(hard_negative_per_batch),
            int(ordinary_negative_per_batch),
            int(transition_negative_per_batch),
        )
        if any(count < 0 for count in counts) or sum(counts) != TEMPORAL_BATCH_SIZE:
            raise ValueError(f"temporal completion batch counts must be non-negative and sum to 64, got {counts}")
        if counts not in ((32, 16, 16, 0), (16, 16, 28, 4), (32, 16, 12, 4)):
            raise ValueError(f"unsupported temporal completion batch composition {counts}")
        self._counts = counts
        self._history_carry = counts[3] > 0
        self._tick_stride_frames = int(tick_stride_frames)

        views = tuple(_normalise_sample(sample, index=index) for index, sample in enumerate(samples))
        self._validate_candidate_rows(views, tick_stride_frames=tick_stride_frames)
        positive_by_task: dict[int, list[int]] = {task: [] for task in TEMPORAL_TASK_INDICES}
        positive_by_episode: dict[tuple[Hashable, int], int] = {}
        hard_by_episode: dict[tuple[Hashable, int], int] = {}
        ordinary_by_episode: dict[tuple[Hashable, int], list[int]] = {}
        transition_by_stratum: dict[tuple[int, int], list[int]] = {}
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
            elif view.sample_kind == "ordinary_negative":
                ordinary_by_episode.setdefault(key, []).append(view.index)
            else:
                if not self._history_carry:
                    raise ValueError("transition_negative rows require the history_carry batch composition")
                transition_by_stratum.setdefault((view.task_index, view.logical_tick), []).append(view.index)

        for task, pool in positive_by_task.items():
            required_positive = positive_per_batch // len(TEMPORAL_TASK_INDICES)
            if len(pool) < required_positive:
                raise ValueError(
                    f"task {task} needs at least {required_positive} positive subtask episodes, got {len(pool)}"
                )
            ordinary_count = sum(key[1] == task for key in ordinary_by_episode)
            required_ordinary = ordinary_negative_per_batch // len(TEMPORAL_TASK_INDICES)
            if ordinary_count < required_ordinary:
                raise ValueError(
                    f"task {task} ordinary pool has only {ordinary_count} episodes; need {required_ordinary}"
                )
            hard_count = sum(key[1] == task for key in hard_by_episode)
            required_hard = hard_negative_per_batch // len(TEMPORAL_TASK_INDICES) if self._history_carry else 8
            if hard_count < required_hard:
                raise ValueError(f"task {task} hard pool has only {hard_count} episodes; need {required_hard}")
        missing_hard = set(positive_by_episode) - set(hard_by_episode)
        if missing_hard:
            raise ValueError(f"every positive subtask episode needs its fixed hard negative: {sorted(missing_hard)!r}")
        orphan_hard = set(hard_by_episode) - set(positive_by_episode)
        if orphan_hard:
            raise ValueError("every hard negative must have a positive from the same subtask episode")
        orphan_ordinary = set(ordinary_by_episode) - set(positive_by_episode)
        if orphan_ordinary:
            raise ValueError("every ordinary negative must have a positive from the same subtask episode")
        for key, positive_index in positive_by_episode.items():
            positive_event = views[positive_index].event_key
            hard_index = hard_by_episode[key]
            if views[hard_index].event_key != positive_event:
                raise ValueError("paired hard negative must use the positive's exact completion event")
            if any(views[index].event_key != positive_event for index in ordinary_by_episode.get(key, ())):
                raise ValueError("ordinary negatives must use the positive's exact completion event")

        if self._history_carry:
            expected_strata = {(task, step) for task in (1, 2, 3) for step in (0, tick_stride_frames)}
            missing_strata = sorted(stratum for stratum in expected_strata if not transition_by_stratum.get(stratum))
            if missing_strata:
                raise ValueError(f"history_carry transition pools are missing strata {missing_strata!r}")
        positive_count = len(positive_by_episode)
        batches_per_epoch = (
            math.ceil(positive_count / positive_per_batch) if batches_per_epoch is None else batches_per_epoch
        )
        if batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive")
        self._views = views
        self._positive_by_task = {task: np.asarray(pool, dtype=np.int64) for task, pool in positive_by_task.items()}
        self._positive_by_episode = positive_by_episode
        self._hard_by_episode = hard_by_episode
        self._ordinary_by_episode = {key: np.asarray(pool, dtype=np.int64) for key, pool in ordinary_by_episode.items()}
        self._transition_by_stratum = {
            key: np.asarray(pool, dtype=np.int64) for key, pool in transition_by_stratum.items()
        }
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
            elif view.sample_kind == "ordinary_negative":
                distance = view.boundary_tick - view.logical_tick
                if distance < 2 * tick_stride_frames or distance % tick_stride_frames:
                    raise ValueError("ordinary negative must be E-30, E-45, ...")
            else:
                if view.task_index not in (1, 2, 3) or view.logical_tick not in (0, tick_stride_frames):
                    raise ValueError("transition negative must be task 1/2/3 at logical tick 0 or 15")
                if view.label != 0:
                    raise ValueError("transition negative must have label 0")

    @property
    def steps_per_epoch(self) -> int:
        return self._batches_per_epoch

    @property
    def batch_composition(self) -> dict[str, int]:
        return {
            "positive": self._counts[0],
            "hard_negative": self._counts[1],
            "ordinary_negative": self._counts[2],
            "transition_negative": self._counts[3],
        }

    @property
    def pool_sizes(self) -> dict[str, int]:
        return {
            "positive": len(self._positive_by_episode),
            "hard_negative": len(self._hard_by_episode),
            "ordinary_negative": sum(len(pool) for pool in self._ordinary_by_episode.values()),
            "transition_negative": sum(len(pool) for pool in self._transition_by_stratum.values()),
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
        hard_negatives = [view for view in selected if view.sample_kind == "hard_negative"]
        ordinary_negatives = [view for view in selected if view.sample_kind == "ordinary_negative"]
        transitions = [view for view in selected if view.sample_kind == "transition_negative"]
        return TemporalBatchAudit(
            positive_count=len(positives),
            hard_negative_count=len(hard_negatives),
            ordinary_negative_count=len(ordinary_negatives),
            transition_negative_count=len(transitions),
            event_local_hard_count=len(hard_negatives),
            same_task_fallback_hard_count=0,
            positive_task_counts=tuple(
                sum(view.task_index == task for view in positives) for task in TEMPORAL_TASK_INDICES
            ),
            hard_task_counts=tuple(
                sum(view.task_index == task for view in hard_negatives) for task in TEMPORAL_TASK_INDICES
            ),
            ordinary_task_counts=tuple(
                sum(view.task_index == task for view in ordinary_negatives) for task in TEMPORAL_TASK_INDICES
            ),
            transition_task_counts=tuple(sum(view.task_index == task for view in transitions) for task in (1, 2, 3)),
            transition_step_counts=tuple(
                sum(view.logical_tick == step for view in transitions) for step in (0, self._tick_stride_frames)
            ),
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
        if (
            int(state.get("seed", -1)) != self._seed
            or int(state.get("batches_per_epoch", -1)) != self._batches_per_epoch
        ):
            raise ValueError("temporal sampler state does not match this sampler")
        self.set_epoch(int(state["epoch"]))
        self.set_skip_batches(int(state["skip_batches"]))

    def _rng(self, *, epoch: int, batch_index: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self._seed, epoch, batch_index]))

    def _make_batch(self, *, epoch: int, batch_index: int) -> list[int]:
        if self._history_carry:
            return self._make_history_carry_batch(epoch=epoch, batch_index=batch_index)
        return self._make_local_batch(epoch=epoch, batch_index=batch_index)

    def _make_local_batch(self, *, epoch: int, batch_index: int) -> list[int]:
        rng = self._rng(epoch=epoch, batch_index=batch_index)
        selected_pairs: list[tuple[int, str]] = []
        for task in TEMPORAL_TASK_INDICES:
            task_keys = list(self._episodes_by_task[task])
            ordinary_keys = [key for key in task_keys if key in self._ordinary_by_episode]
            ordinary_keys = [
                ordinary_keys[int(index)] for index in rng.choice(len(ordinary_keys), size=4, replace=False)
            ]
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

    def _make_history_carry_batch(self, *, epoch: int, batch_index: int) -> list[int]:
        """Build one of the fixed task-balanced history-carry batches."""

        rng = self._rng(epoch=epoch, batch_index=batch_index)
        indices: list[int] = []
        selected_positive_keys: set[tuple[Hashable, int]] = set()

        positive_per_task = self._counts[0] // len(TEMPORAL_TASK_INDICES)
        hard_per_task = self._counts[1] // len(TEMPORAL_TASK_INDICES)
        ordinary_per_task = self._counts[2] // len(TEMPORAL_TASK_INDICES)

        # Select the positive events first.  The hard pool is a deliberately
        # smaller paired subset for the 32/16/12/4 ablation, so not every
        # positive in that composition receives a hard row in this batch.
        selected_positive_by_task: dict[int, list[tuple[Hashable, int]]] = {}
        for task in TEMPORAL_TASK_INDICES:
            pool = self._positive_by_task[task]
            positive_indices = pool[rng.permutation(len(pool))[:positive_per_task]]
            selected_positive_by_task[task] = []
            for raw_positive_index in positive_indices:
                positive_index = int(raw_positive_index)
                positive_view = self._views[positive_index]
                key = (positive_view.trajectory_id, positive_view.task_index)
                selected_positive_keys.add(key)
                selected_positive_by_task[task].append((key, positive_index))
                indices.append(positive_index)

        # Pair a fixed number of hard negatives with positive events from the
        # same batch.  For the original 16/16/28/4 composition this includes
        # every selected positive; for 32/16/12/4 it includes four per task.
        for task in TEMPORAL_TASK_INDICES:
            for key, _ in selected_positive_by_task[task][:hard_per_task]:
                indices.append(int(self._hard_by_episode[key]))

        # Task-balanced ordinary events.  Choose distinct episodes in the
        # batch, then advance the selected episode's row deterministically so
        # repeated batches do not always select the same tick.
        for task in TEMPORAL_TASK_INDICES:
            candidates = [
                key
                for key in self._episodes_by_task[task]
                if key in self._ordinary_by_episode and key not in selected_positive_keys
            ]
            if len(candidates) < self._counts[2] // len(TEMPORAL_TASK_INDICES):
                candidates = [key for key in self._episodes_by_task[task] if key in self._ordinary_by_episode]
            chosen = [candidates[int(index)] for index in rng.permutation(len(candidates))[:ordinary_per_task]]
            for candidate in chosen:
                pool = self._ordinary_by_episode[candidate]
                offset = (epoch + batch_index + self._episodes_by_task[task].index(candidate)) % len(pool)
                indices.append(int(pool[offset]))

        # Three-batch schedule: every task and both transition steps receive
        # the same long-run number of samples while every batch covers tasks
        # 1/2/3.
        transition_schedule = (
            ((1, 0), (1, self._tick_stride_frames), (2, 0), (3, self._tick_stride_frames)),
            ((2, 0), (2, self._tick_stride_frames), (1, self._tick_stride_frames), (3, 0)),
            ((3, 0), (3, self._tick_stride_frames), (1, 0), (2, self._tick_stride_frames)),
        )
        for task, step in transition_schedule[batch_index % len(transition_schedule)]:
            pool = self._transition_by_stratum[(task, step)]
            permutation_rng = np.random.default_rng(np.random.SeedSequence([self._seed, epoch, task, step]))
            permutation = permutation_rng.permutation(len(pool))
            offset = (epoch * self._batches_per_epoch + batch_index) % len(pool)
            indices.append(int(pool[int(permutation[offset])]))

        if len(indices) != TEMPORAL_BATCH_SIZE:
            raise AssertionError(f"internal history-carry batch size error: {len(indices)}")
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


class StartTerminalTemporalCompletionBatchSampler:
    """Task-balanced sampler for the independent start/terminal protocol.

    Endpoint positives are paired only with their same-event hard negatives.
    The two terminal-positive pools and the three negative pools are sampled
    independently by task.  If a pool has fewer events than its per-batch
    quota, deterministic replacement keeps the requested composition intact.
    """

    def __init__(
        self,
        samples: Sequence[TemporalSampleLike],
        *,
        seed: int,
        batches_per_epoch: int | None = None,
        batch_size: int = TEMPORAL_BATCH_SIZE,
        tick_stride_frames: int = DEFAULT_TICK_STRIDE_FRAMES,
        endpoint_positive_per_batch: int = START_TERMINAL_BATCH_COUNTS["endpoint_positive"],
        terminal_one_hold_positive_per_batch: int = START_TERMINAL_BATCH_COUNTS["terminal_one_hold_positive"],
        terminal_full_hold_positive_per_batch: int = START_TERMINAL_BATCH_COUNTS["terminal_full_hold_positive"],
        hard_negative_per_batch: int = START_TERMINAL_BATCH_COUNTS["hard_negative"],
        ordinary_negative_per_batch: int = START_TERMINAL_BATCH_COUNTS["ordinary_negative"],
        start_0_negative_per_batch: int = START_TERMINAL_BATCH_COUNTS["start_0_negative"],
        start_15_negative_per_batch: int = START_TERMINAL_BATCH_COUNTS["start_15_negative"],
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if batch_size != TEMPORAL_BATCH_SIZE:
            raise ValueError(f"start_terminal batch_size must be exactly {TEMPORAL_BATCH_SIZE}")
        if seed < 0 or tick_stride_frames != DEFAULT_TICK_STRIDE_FRAMES:
            raise ValueError("start_terminal sampler requires a non-negative seed and 15-frame stride")
        if num_replicas != 1 or rank != 0:
            raise ValueError("StartTerminalTemporalCompletionBatchSampler is single-process in this version")
        if not samples:
            raise ValueError("start_terminal sampling requires candidate rows")
        counts = {
            "endpoint_positive": int(endpoint_positive_per_batch),
            "terminal_one_hold_positive": int(terminal_one_hold_positive_per_batch),
            "terminal_full_hold_positive": int(terminal_full_hold_positive_per_batch),
            "hard_negative": int(hard_negative_per_batch),
            "ordinary_negative": int(ordinary_negative_per_batch),
            "start_0_negative": int(start_0_negative_per_batch),
            "start_15_negative": int(start_15_negative_per_batch),
        }
        if counts != START_TERMINAL_BATCH_COUNTS:
            raise ValueError(f"start_terminal requires batch composition {START_TERMINAL_BATCH_COUNTS}, got {counts}")

        views = tuple(_normalise_start_terminal_sample(sample, index=index) for index, sample in enumerate(samples))
        self._validate_candidate_rows(views)
        endpoint_by_event: dict[CompletionEventKey, int] = {}
        hard_by_event: dict[CompletionEventKey, int] = {}
        variant_event_rows: dict[str, dict[CompletionEventKey, list[int]]] = {
            variant: {} for variant in START_TERMINAL_WINDOW_VARIANTS
        }
        row_keys: set[tuple[Hashable, int, str, int, int]] = set()
        ordinary_source_keys: set[tuple[Hashable, int, tuple[int, int, int], tuple[int, int, int], int]] = set()
        start_source_keys: set[tuple[Hashable, int, tuple[int, int, int], tuple[int, int, int], int]] = set()
        for item in views:
            view = item.view
            row_key = (view.trajectory_id, view.task_index, view.window_variant, view.logical_tick, view.boundary_tick)
            if row_key in row_keys:
                raise ValueError(f"duplicate start_terminal candidate row for {row_key!r}")
            row_keys.add(row_key)
            event_rows = variant_event_rows[view.window_variant].setdefault(view.event_key, [])
            event_rows.append(view.index)
            if view.window_variant == "endpoint_positive":
                if view.event_key in endpoint_by_event:
                    raise ValueError(f"duplicate endpoint positive event {view.event_key!r}")
                endpoint_by_event[view.event_key] = view.index
            elif view.window_variant == "hard_negative":
                if view.event_key in hard_by_event:
                    raise ValueError(f"duplicate hard negative event {view.event_key!r}")
                hard_by_event[view.event_key] = view.index
            source_key = (
                view.trajectory_id,
                view.task_index,
                item.source_episode_ids,
                item.source_frame_indices,
                item.prompt_index,
            )
            if view.window_variant == "ordinary_negative":
                ordinary_source_keys.add(source_key)
            elif view.window_variant in ("start_0_negative", "start_15_negative"):
                start_source_keys.add(source_key)

        overlap = ordinary_source_keys.intersection(start_source_keys)
        if overlap:
            raise ValueError("ordinary_negative and start negative pools contain an identical source window")
        endpoint_events = set(endpoint_by_event)
        missing_hard = endpoint_events - set(hard_by_event)
        if missing_hard:
            raise ValueError(f"every endpoint positive needs its exact hard negative: {sorted(missing_hard, key=repr)!r}")
        orphan_hard = set(hard_by_event) - endpoint_events
        if orphan_hard:
            raise ValueError("every hard negative must have an endpoint positive from the same completion event")
        for variant, by_event in variant_event_rows.items():
            if variant in ("endpoint_positive", "hard_negative"):
                continue
            orphan = set(by_event) - endpoint_events
            if orphan:
                raise ValueError(f"{variant} contains rows without a matching endpoint completion event")

        variant_by_task_event: dict[str, dict[int, tuple[CompletionEventKey, ...]]] = {}
        for variant, by_event in variant_event_rows.items():
            variant_by_task_event[variant] = {
                task: tuple(event for event in by_event if event.task_index == task)
                for task in TEMPORAL_TASK_INDICES
            }
            for task in TEMPORAL_TASK_INDICES:
                if not variant_by_task_event[variant][task]:
                    raise ValueError(f"start_terminal pool {variant} has no task {task} events")

        if batches_per_epoch is None:
            batches_per_epoch = math.ceil(len(endpoint_events) / START_TERMINAL_BATCH_COUNTS["endpoint_positive"])
        if batches_per_epoch <= 0:
            raise ValueError("batches_per_epoch must be positive")
        self._views = tuple(item.view for item in views)
        self._variant_event_rows = {
            variant: {event: tuple(indices) for event, indices in by_event.items()}
            for variant, by_event in variant_event_rows.items()
        }
        self._variant_by_task_event = variant_by_task_event
        self._endpoint_by_event = endpoint_by_event
        self._hard_by_event = hard_by_event
        self._counts = dict(counts)
        self._seed = int(seed)
        self._batches_per_epoch = int(batches_per_epoch)
        self._epoch = 0
        self._skip_batches = 0

    @staticmethod
    def _validate_candidate_rows(views: Sequence[_StartTerminalSampleView]) -> None:
        for item in views:
            view = item.view
            if view.task_index not in TEMPORAL_TASK_INDICES:
                raise ValueError(f"invalid start_terminal task_index {view.task_index}")
            if view.window_variant in ("endpoint_positive", "terminal_one_hold_positive", "terminal_full_hold_positive"):
                if view.label != 1:
                    raise ValueError(f"{view.window_variant} must have label 1")
            elif view.label != 0:
                raise ValueError(f"{view.window_variant} must have label 0")

    @property
    def steps_per_epoch(self) -> int:
        return self._batches_per_epoch

    @property
    def batch_composition(self) -> dict[str, int]:
        return dict(self._counts)

    @property
    def pool_sizes(self) -> dict[str, int]:
        return {
            variant: sum(len(indices) for indices in by_event.values())
            for variant, by_event in self._variant_event_rows.items()
        }

    @property
    def pool_event_sizes(self) -> dict[str, int]:
        return {variant: len(by_event) for variant, by_event in self._variant_event_rows.items()}

    @property
    def events_without_local_hard(self) -> frozenset[CompletionEventKey]:
        return frozenset()

    def audit_batch(self, batch: Sequence[int]) -> StartTerminalBatchAudit:
        indices = [operator.index(index) for index in batch]
        if len(indices) != TEMPORAL_BATCH_SIZE or any(index < 0 or index >= len(self._views) for index in indices):
            raise ValueError("start_terminal batch must contain exactly 64 valid candidate indices")
        selected = [self._views[index] for index in indices]
        variant_counts = {
            variant: sum(view.window_variant == variant for view in selected)
            for variant in START_TERMINAL_WINDOW_VARIANTS
        }
        if variant_counts != self._counts:
            raise ValueError(f"start_terminal batch has composition {variant_counts}, expected {self._counts}")
        task_counts = {
            variant: tuple(
                sum(view.window_variant == variant and view.task_index == task for view in selected)
                for task in TEMPORAL_TASK_INDICES
            )
            for variant in START_TERMINAL_WINDOW_VARIANTS
        }
        expected_task_counts = {
            variant: (count // len(TEMPORAL_TASK_INDICES),) * len(TEMPORAL_TASK_INDICES)
            for variant, count in self._counts.items()
        }
        if task_counts != expected_task_counts:
            raise ValueError(f"start_terminal batch is not task-balanced: {task_counts}")
        if any(view.sample_kind == "transition_negative" for view in selected):
            raise ValueError("start_terminal batches must not contain transition negatives")
        return StartTerminalBatchAudit(
            variant_counts=variant_counts,
            task_counts=task_counts,
            transition_negative_count=0,
        )

    @staticmethod
    def _choose_positions(rng: np.random.Generator, size: int, count: int) -> np.ndarray:
        if size <= 0:
            raise ValueError("cannot sample from an empty start_terminal pool")
        if size >= count:
            return rng.permutation(size)[:count]
        return rng.integers(0, size, size=count)

    def _rng(self, *, epoch: int, batch_index: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self._seed, epoch, batch_index]))

    def _choose_events(
        self,
        rng: np.random.Generator,
        *,
        variant: str,
        task: int,
        count: int,
    ) -> list[CompletionEventKey]:
        events = self._variant_by_task_event[variant][task]
        positions = self._choose_positions(rng, len(events), count)
        return [events[int(position)] for position in positions]

    def _choose_one_row_for_event(
        self,
        rng: np.random.Generator,
        *,
        variant: str,
        event: CompletionEventKey,
    ) -> int:
        indices = self._variant_event_rows[variant][event]
        return int(indices[int(rng.integers(0, len(indices)))])

    def _make_batch(self, *, epoch: int, batch_index: int) -> list[int]:
        rng = self._rng(epoch=epoch, batch_index=batch_index)
        indices: list[int] = []
        for task in TEMPORAL_TASK_INDICES:
            endpoint_events = self._choose_events(
                rng,
                variant="endpoint_positive",
                task=task,
                count=self._counts["endpoint_positive"] // len(TEMPORAL_TASK_INDICES),
            )
            for event in endpoint_events:
                indices.append(self._endpoint_by_event[event])
                indices.append(self._hard_by_event[event])

            for variant in ("terminal_one_hold_positive", "terminal_full_hold_positive"):
                indices.extend(
                    self._choose_one_row_for_event(rng, variant=variant, event=event)
                    for event in self._choose_events(
                        rng,
                        variant=variant,
                        task=task,
                        count=self._counts[variant] // len(TEMPORAL_TASK_INDICES),
                    )
                )

            for variant in ("ordinary_negative", "start_0_negative", "start_15_negative"):
                indices.extend(
                    self._choose_one_row_for_event(rng, variant=variant, event=event)
                    for event in self._choose_events(
                        rng,
                        variant=variant,
                        task=task,
                        count=self._counts[variant] // len(TEMPORAL_TASK_INDICES),
                    )
                )

        if len(indices) != TEMPORAL_BATCH_SIZE:
            raise AssertionError(f"internal start_terminal batch size error: {len(indices)}")
        rng.shuffle(indices)
        return [int(index) for index in indices]

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
            "schema_version": _START_TERMINAL_STATE_SCHEMA_VERSION,
            "seed": self._seed,
            "batches_per_epoch": self._batches_per_epoch,
            "epoch": self._epoch,
            "skip_batches": self._skip_batches,
        }

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        if int(state.get("schema_version", -1)) != _START_TERMINAL_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported start_terminal sampler state schema_version")
        if (
            int(state.get("seed", -1)) != self._seed
            or int(state.get("batches_per_epoch", -1)) != self._batches_per_epoch
        ):
            raise ValueError("start_terminal sampler state does not match this sampler")
        self.set_epoch(int(state["epoch"]))
        self.set_skip_batches(int(state["skip_batches"]))

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
