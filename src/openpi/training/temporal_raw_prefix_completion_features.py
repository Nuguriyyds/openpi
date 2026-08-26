"""Lightweight three-frame raw-prefix history sidecars.

The current-frame raw-prefix cache remains the source of truth for rows and
all already-extracted features.  This module stores only a per-slot location
map plus float16 extension shards for history keys that are not present in
that base cache.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any

import numpy as np

from openpi.training import raw_prefix_completion_features as _raw_features
from openpi.training import temporal_completion_data as _temporal_data

TEMPORAL_RAW_PREFIX_CACHE_SCHEMA_VERSION = 1
TEMPORAL_RAW_PREFIX_STORAGE_FORMAT = "directory_extension_npy_v1"
TEMPORAL_RAW_PREFIX_STORAGE_DTYPE = np.dtype(np.float16)
DEFAULT_MAX_SHARD_BYTES = 1 << 30
BASE_LOCATION = 0
EXTENSION_LOCATION = 1


@dataclasses.dataclass(frozen=True, order=True)
class TemporalRawPrefixKey:
    """A prompt-conditioned source frame address used by both caches."""

    source_episode_id: int
    source_frame_index: int
    prompt_index: int

    def __post_init__(self) -> None:
        if self.source_episode_id < 0 or self.source_frame_index < 0:
            raise ValueError("temporal raw-prefix source coordinates must be non-negative")
        if self.prompt_index not in range(_temporal_data.TASKS_PER_TRAJECTORY):
            raise ValueError("temporal raw-prefix prompt_index must be in [0, 3]")


@dataclasses.dataclass(frozen=True)
class TemporalRawPrefixHistoryPlan:
    """Canonical rows and their base/extension slot locations."""

    rows: tuple[_temporal_data.TemporalSampleRow, ...]
    history_location_kind: np.ndarray
    history_location_index: np.ndarray
    missing_keys: tuple[TemporalRawPrefixKey, ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def extension_count(self) -> int:
        return len(self.missing_keys)


def _row_history_keys(row: _temporal_data.TemporalSampleRow) -> tuple[TemporalRawPrefixKey, ...]:
    if row.sample_kind == "transition_negative":
        raise ValueError("temporal raw-prefix history does not accept transition negatives")
    prompt_index = int(row.prompt_index)
    return tuple(
        TemporalRawPrefixKey(int(episode_id), int(frame_index), prompt_index)
        for episode_id, frame_index in zip(row.source_episode_ids, row.source_frame_indices, strict=True)
    )


def build_temporal_raw_prefix_history_plan(
    manifest: _temporal_data.TemporalCompletionManifest,
    base_cache: _raw_features.RawPrefixCompletionCache,
) -> TemporalRawPrefixHistoryPlan:
    """Maps every sealed history slot to the base cache or a new extension row."""

    rows = _raw_features.manifest_rows(manifest)
    base_cache.validate(manifest)
    if base_cache.metadata.task_prompts != manifest.task_prompts:
        raise ValueError("base raw-prefix cache prompts differ from the sealed manifest")

    base_by_key: dict[TemporalRawPrefixKey, int] = {}
    for row_index, row in enumerate(base_cache.rows):
        key = TemporalRawPrefixKey(
            int(row.source_episode_ids[-1]),
            int(row.source_frame_indices[-1]),
            int(row.prompt_index),
        )
        feature_index = int(base_cache.row_feature_indices[row_index])
        previous = base_by_key.setdefault(key, feature_index)
        if previous != feature_index:
            raise ValueError("base raw-prefix cache maps one source key to multiple feature rows")

    extension_by_key: dict[TemporalRawPrefixKey, int] = {}
    location_kind = np.empty((len(rows), _temporal_data.TEMPORAL_HISTORY_STEPS), dtype=np.int8)
    location_index = np.empty((len(rows), _temporal_data.TEMPORAL_HISTORY_STEPS), dtype=np.int64)
    for row_index, row in enumerate(rows):
        for slot, key in enumerate(_row_history_keys(row)):
            if key in base_by_key:
                location_kind[row_index, slot] = BASE_LOCATION
                location_index[row_index, slot] = base_by_key[key]
            else:
                extension_index = extension_by_key.setdefault(key, len(extension_by_key))
                location_kind[row_index, slot] = EXTENSION_LOCATION
                location_index[row_index, slot] = extension_index

    return TemporalRawPrefixHistoryPlan(
        rows=rows,
        history_location_kind=location_kind,
        history_location_index=location_index,
        missing_keys=tuple(extension_by_key),
    )


@dataclasses.dataclass(frozen=True)
class TemporalRawPrefixHistoryCacheMetadata:
    model_config_name: str
    checkpoint_path: str
    base_cache_path: str
    task_prompts: tuple[str, str, str, str]
    input_dim: int
    token_count: int
    row_count: int
    extension_count: int
    extension_shard_count: int
    extension_shard_rows: tuple[int, ...]
    base_cache_schema_version: int
    schema_version: int = TEMPORAL_RAW_PREFIX_CACHE_SCHEMA_VERSION
    storage_format: str = TEMPORAL_RAW_PREFIX_STORAGE_FORMAT
    storage_dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.schema_version != TEMPORAL_RAW_PREFIX_CACHE_SCHEMA_VERSION:
            raise ValueError(
                "temporal raw-prefix history schema_version must be "
                f"{TEMPORAL_RAW_PREFIX_CACHE_SCHEMA_VERSION}"
            )
        if self.storage_format != TEMPORAL_RAW_PREFIX_STORAGE_FORMAT:
            raise ValueError(
                "temporal raw-prefix history storage_format must be "
                f"{TEMPORAL_RAW_PREFIX_STORAGE_FORMAT!r}"
            )
        if self.storage_dtype != "float16":
            raise ValueError("temporal raw-prefix history storage_dtype must be 'float16'")
        if not self.model_config_name or not self.checkpoint_path or not self.base_cache_path:
            raise ValueError("temporal raw-prefix history source bindings must be non-empty")
        if len(self.task_prompts) != _temporal_data.TASKS_PER_TRAJECTORY or any(
            not prompt.strip() for prompt in self.task_prompts
        ):
            raise ValueError("temporal raw-prefix history must seal four non-empty task prompts")
        if self.input_dim <= 0 or self.token_count <= 0 or self.row_count <= 0:
            raise ValueError("temporal raw-prefix history dimensions and row_count must be positive")
        if self.extension_count < 0 or self.extension_shard_count < 0:
            raise ValueError("temporal raw-prefix history extension counts must be non-negative")
        if len(self.extension_shard_rows) != self.extension_shard_count:
            raise ValueError("temporal raw-prefix history shard row metadata has the wrong length")
        if any(rows <= 0 for rows in self.extension_shard_rows):
            raise ValueError("temporal raw-prefix history shards must contain positive row counts")
        if sum(self.extension_shard_rows) != self.extension_count:
            raise ValueError("temporal raw-prefix history shard rows do not sum to extension_count")
        if self.base_cache_schema_version <= 0:
            raise ValueError("temporal raw-prefix history base cache schema must be positive")

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> TemporalRawPrefixHistoryCacheMetadata:
        payload = json.loads(value)
        expected = {field.name for field in dataclasses.fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                "temporal raw-prefix history metadata fields do not match schema; "
                f"missing={sorted(expected - set(payload))}, unexpected={sorted(set(payload) - expected)}"
            )
        payload["task_prompts"] = tuple(str(prompt) for prompt in payload["task_prompts"])
        payload["extension_shard_rows"] = tuple(int(rows) for rows in payload["extension_shard_rows"])
        return cls(**payload)


@dataclasses.dataclass(frozen=True)
class TemporalRawPrefixExtension:
    """Mmap-backed extension arrays, empty when all slots reuse the base."""

    prefix_out: _raw_features.ShardedNpyArray | None
    prefix_mask: _raw_features.ShardedNpyArray | None

    @property
    def shape(self) -> tuple[int, ...]:
        if self.prefix_out is None:
            return (0,)
        return self.prefix_out.shape

    def feature(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.prefix_out is None or self.prefix_mask is None:
            raise IndexError("temporal raw-prefix extension has no features")
        return self.prefix_out[index], self.prefix_mask[index]


@dataclasses.dataclass(frozen=True)
class TemporalRawPrefixHistoryCache:
    metadata: TemporalRawPrefixHistoryCacheMetadata
    rows: tuple[_temporal_data.TemporalSampleRow, ...]
    base_cache: _raw_features.RawPrefixCompletionCache
    extension: TemporalRawPrefixExtension
    history_location_kind: np.ndarray
    history_location_index: np.ndarray
    extension_keys: tuple[TemporalRawPrefixKey, ...]

    @property
    def prefix_segment_ids(self) -> np.ndarray:
        return self.base_cache.prefix_segment_ids

    @property
    def prefix_position_ids(self) -> np.ndarray:
        return self.base_cache.prefix_position_ids

    @property
    def extension_count(self) -> int:
        return self.metadata.extension_count

    def validate(self, manifest: _temporal_data.TemporalCompletionManifest) -> None:
        expected_rows = _raw_features.manifest_rows(manifest)
        if self.rows != expected_rows:
            raise ValueError("temporal raw-prefix history rows differ from the sealed manifest")
        if self.metadata.task_prompts != manifest.task_prompts:
            raise ValueError("temporal raw-prefix history prompts differ from the sealed manifest")
        if self.metadata.row_count != len(self.rows):
            raise ValueError("temporal raw-prefix history row_count does not match row metadata")
        if self.metadata.base_cache_schema_version != self.base_cache.metadata.schema_version:
            raise ValueError("temporal raw-prefix history is bound to a different base cache schema")
        if self.metadata.model_config_name != self.base_cache.metadata.model_config_name:
            raise ValueError("temporal raw-prefix history model config differs from the base cache")
        if self.metadata.checkpoint_path != self.base_cache.metadata.checkpoint_path:
            raise ValueError("temporal raw-prefix history checkpoint differs from the base cache")
        if self.metadata.input_dim != self.base_cache.metadata.input_dim:
            raise ValueError("temporal raw-prefix history input_dim differs from the base cache")
        if self.metadata.token_count != self.base_cache.metadata.token_count:
            raise ValueError("temporal raw-prefix history token_count differs from the base cache")

        kinds = np.asarray(self.history_location_kind)
        indices = np.asarray(self.history_location_index)
        expected_locations = (len(self.rows), _temporal_data.TEMPORAL_HISTORY_STEPS)
        if kinds.shape != expected_locations or kinds.dtype.kind not in "iu":
            raise ValueError("history_location_kind must be an integer array with shape [row_count, 3]")
        if indices.shape != expected_locations or indices.dtype.kind not in "iu":
            raise ValueError("history_location_index must be an integer array with shape [row_count, 3]")
        if kinds.size and not np.all(np.logical_or(kinds == BASE_LOCATION, kinds == EXTENSION_LOCATION)):
            raise ValueError("history_location_kind must contain only base (0) or extension (1) locations")
        for kind, limit in ((BASE_LOCATION, self.base_cache.metadata.feature_count), (EXTENSION_LOCATION, self.extension_count)):
            selected = indices[kinds == kind]
            if selected.size and (int(selected.min()) < 0 or int(selected.max()) >= limit):
                raise ValueError("history_location_index contains an out-of-range location")

        if len(self.extension_keys) != self.extension_count:
            raise ValueError("extension key count does not match metadata")
        if len(set(self.extension_keys)) != len(self.extension_keys):
            raise ValueError("extension keys must be unique")
        base_keys = {
            TemporalRawPrefixKey(
                int(row.source_episode_ids[-1]),
                int(row.source_frame_indices[-1]),
                int(row.prompt_index),
            )
            for row in self.base_cache.rows
        }
        if base_keys.intersection(self.extension_keys):
            raise ValueError("extension keys must not duplicate base cache keys")
        if self.extension_count == 0:
            if self.extension.prefix_out is not None or self.extension.prefix_mask is not None:
                raise ValueError("empty temporal raw-prefix history must not expose extension shards")
        else:
            if self.extension.prefix_out is None or self.extension.prefix_mask is None:
                raise ValueError("non-empty temporal raw-prefix history lacks extension shards")
            expected_shape = (
                self.extension_count,
                self.metadata.token_count,
                self.metadata.input_dim,
            )
            if self.extension.prefix_out.shape != expected_shape:
                raise ValueError(f"extension prefix_out shape must be {expected_shape}, got {self.extension.prefix_out.shape}")
            expected_mask_shape = (self.extension_count, self.metadata.token_count)
            if self.extension.prefix_mask.shape != expected_mask_shape:
                raise ValueError(
                    f"extension prefix_mask shape must be {expected_mask_shape}, got {self.extension.prefix_mask.shape}"
                )

    def indices_for_split(self, split: _temporal_data.SplitName) -> np.ndarray:
        if split not in _temporal_data.SPLIT_NAMES:
            raise ValueError(f"invalid temporal split {split!r}")
        return np.asarray([index for index, row in enumerate(self.rows) if row.split == split], dtype=np.int64)

    def features_for_row(self, row_index: int) -> tuple[np.ndarray, np.ndarray]:
        row_index = int(row_index)
        values = np.empty(
            (_temporal_data.TEMPORAL_HISTORY_STEPS, self.metadata.token_count, self.metadata.input_dim),
            dtype=TEMPORAL_RAW_PREFIX_STORAGE_DTYPE,
        )
        masks = np.empty((_temporal_data.TEMPORAL_HISTORY_STEPS, self.metadata.token_count), dtype=np.bool_)
        for slot in range(_temporal_data.TEMPORAL_HISTORY_STEPS):
            kind = int(self.history_location_kind[row_index, slot])
            feature_index = int(self.history_location_index[row_index, slot])
            if kind == BASE_LOCATION:
                values[slot] = self.base_cache.prefix_out[feature_index]
                masks[slot] = self.base_cache.prefix_mask[feature_index]
            else:
                values[slot], masks[slot] = self.extension.feature(feature_index)
        return values, masks

    def features_for_rows(self, row_indices: Sequence[int] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Materializes only the requested history rows for one eval batch."""

        selected = np.asarray(row_indices, dtype=np.int64).reshape(-1)
        values = np.empty(
            (selected.size, _temporal_data.TEMPORAL_HISTORY_STEPS, self.metadata.token_count, self.metadata.input_dim),
            dtype=TEMPORAL_RAW_PREFIX_STORAGE_DTYPE,
        )
        masks = np.empty(
            (selected.size, _temporal_data.TEMPORAL_HISTORY_STEPS, self.metadata.token_count),
            dtype=np.bool_,
        )
        for output_index, row_index in enumerate(selected):
            values[output_index], masks[output_index] = self.features_for_row(int(row_index))
        return values, masks


def _extension_key_arrays(keys: Sequence[TemporalRawPrefixKey]) -> dict[str, np.ndarray]:
    return {
        "source_episode_id": np.asarray([key.source_episode_id for key in keys], dtype=np.int64),
        "source_frame_index": np.asarray([key.source_frame_index for key in keys], dtype=np.int64),
        "prompt_index": np.asarray([key.prompt_index for key in keys], dtype=np.int8),
    }


def _load_extension_keys(root: pathlib.Path, count: int) -> tuple[TemporalRawPrefixKey, ...]:
    names = ("source_episode_id", "source_frame_index", "prompt_index")
    arrays = {
        name: np.load(root / f"extension_{name}.npy", allow_pickle=False, mmap_mode="r") for name in names
    }
    if any(array.shape != (count,) for array in arrays.values()):
        raise ValueError("temporal raw-prefix extension key arrays have inconsistent shapes")
    return tuple(
        TemporalRawPrefixKey(int(arrays["source_episode_id"][index]), int(arrays["source_frame_index"][index]), int(arrays["prompt_index"][index]))
        for index in range(count)
    )


def _load_extension(root: pathlib.Path, metadata: TemporalRawPrefixHistoryCacheMetadata) -> TemporalRawPrefixExtension:
    if metadata.extension_count == 0:
        if any(root.glob("extension_prefix_out_[0-9][0-9][0-9][0-9][0-9].npy")) or any(
            root.glob("extension_prefix_mask_[0-9][0-9][0-9][0-9][0-9].npy")
        ):
            raise ValueError("empty temporal raw-prefix history contains unexpected extension shards")
        return TemporalRawPrefixExtension(prefix_out=None, prefix_mask=None)
    prefix_paths = tuple(sorted(root.glob("extension_prefix_out_[0-9][0-9][0-9][0-9][0-9].npy")))
    mask_paths = tuple(sorted(root.glob("extension_prefix_mask_[0-9][0-9][0-9][0-9][0-9].npy")))
    if len(prefix_paths) != metadata.extension_shard_count or len(mask_paths) != metadata.extension_shard_count:
        raise ValueError("temporal raw-prefix history shard count does not match metadata")
    prefix_shards = tuple(np.load(path, allow_pickle=False, mmap_mode="r") for path in prefix_paths)
    mask_shards = tuple(np.load(path, allow_pickle=False, mmap_mode="r") for path in mask_paths)
    if tuple(int(shard.shape[0]) for shard in prefix_shards) != metadata.extension_shard_rows:
        raise ValueError("temporal raw-prefix history prefix shard rows do not match metadata")
    if tuple(int(shard.shape[0]) for shard in mask_shards) != metadata.extension_shard_rows:
        raise ValueError("temporal raw-prefix history mask shard rows do not match metadata")
    if any(shard.dtype != TEMPORAL_RAW_PREFIX_STORAGE_DTYPE for shard in prefix_shards):
        raise ValueError("temporal raw-prefix history extension prefix shards must be float16")
    if any(shard.dtype != np.bool_ for shard in mask_shards):
        raise ValueError("temporal raw-prefix history extension mask shards must be bool")
    return TemporalRawPrefixExtension(
        prefix_out=_raw_features.ShardedNpyArray(prefix_shards),
        prefix_mask=_raw_features.ShardedNpyArray(mask_shards),
    )


class TemporalRawPrefixHistoryWriter:
    """Streams missing history features into bounded sidecar shards.

    The base current-frame cache is never copied.  Only keys absent from that
    cache are appended here, using the same bounded-shard writer pattern as
    the current-frame raw-prefix extractor.  The writer deliberately emits a
    shard from each incoming model batch instead of allocating a full shard
    buffer, because some mounted filesystems reject very large temporary
    mappings and the raw token payload is already large.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        manifest: _temporal_data.TemporalCompletionManifest,
        base_cache: _raw_features.RawPrefixCompletionCache,
        plan: TemporalRawPrefixHistoryPlan,
        base_cache_path: str | os.PathLike[str],
        max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    ) -> None:
        expected_rows = _raw_features.manifest_rows(manifest)
        if plan.rows != expected_rows:
            raise ValueError("temporal raw-prefix history plan rows differ from the manifest")
        base_cache.validate(manifest)
        if plan.history_location_kind.shape != (
            len(expected_rows),
            _temporal_data.TEMPORAL_HISTORY_STEPS,
        ):
            raise ValueError("history_location_kind has the wrong shape")
        if plan.history_location_index.shape != plan.history_location_kind.shape:
            raise ValueError("history location arrays must have the same shape")
        if max_shard_bytes <= 0:
            raise ValueError("max_shard_bytes must be positive")

        self.manifest = manifest
        self.base_cache = base_cache
        self.plan = plan
        self.output_path = pathlib.Path(path)
        if self.output_path.exists():
            raise FileExistsError(f"refusing to overwrite sealed temporal raw-prefix history: {self.output_path}")
        self.base_cache_path = str(pathlib.Path(base_cache_path).resolve())
        self.max_shard_bytes = int(max_shard_bytes)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = pathlib.Path(
            tempfile.mkdtemp(prefix=f".{self.output_path.name}.{os.getpid()}.", dir=self.output_path.parent)
        )

        self._token_count = int(base_cache.metadata.token_count)
        self._input_dim = int(base_cache.metadata.input_dim)
        self._segment_ids = np.asarray(base_cache.prefix_segment_ids, dtype=np.int32).copy()
        self._position_ids = np.asarray(base_cache.prefix_position_ids, dtype=np.int32).copy()
        self._written_count = 0
        self._shard_count = 0
        self._shard_rows: list[int] = []
        self._finalized = False

        bytes_per_feature = self._token_count * (
            self._input_dim * TEMPORAL_RAW_PREFIX_STORAGE_DTYPE.itemsize + np.dtype(np.bool_).itemsize
        )
        if plan.extension_count and max_shard_bytes < bytes_per_feature:
            raise ValueError("max_shard_bytes is smaller than one temporal raw-prefix extension feature")
        self._features_per_shard = (
            max(1, min(plan.extension_count, max_shard_bytes // bytes_per_feature))
            if plan.extension_count
            else 0
        )
        # Keep only the per-model-batch NumPy arrays supplied to append; the
        # shard-size calculation above splits oversized incoming batches
        # without retaining all missing features.

    def append(
        self,
        prefix_out: np.ndarray,
        prefix_mask: np.ndarray,
        prefix_segment_ids: np.ndarray,
        prefix_position_ids: np.ndarray,
    ) -> None:
        """Appends one model batch without retaining previous batches in RAM."""

        if self._finalized:
            raise RuntimeError("cannot append to a finalized temporal raw-prefix history")
        values = np.asarray(prefix_out)
        masks = np.asarray(prefix_mask, dtype=np.bool_)
        segment_ids = np.asarray(prefix_segment_ids, dtype=np.int32)
        position_ids = np.asarray(prefix_position_ids, dtype=np.int32)
        expected_values = (self._token_count, self._input_dim)
        if values.ndim != 3 or values.shape[0] <= 0 or values.shape[1:] != expected_values:
            raise ValueError(
                "temporal raw-prefix extension prefix_out must have shape "
                f"[batch, {self._token_count}, {self._input_dim}], got {values.shape}"
            )
        if masks.shape != values.shape[:2]:
            raise ValueError(f"temporal raw-prefix extension prefix_mask must have shape {values.shape[:2]}")
        if segment_ids.shape != (self._token_count,) or position_ids.shape != segment_ids.shape:
            raise ValueError("temporal raw-prefix extension layout ids must each have shape [S]")
        if not np.array_equal(segment_ids, self._segment_ids) or not np.array_equal(position_ids, self._position_ids):
            raise ValueError("temporal raw-prefix extension layout differs from the base cache")
        if self._written_count + values.shape[0] > self.plan.extension_count:
            raise ValueError("received more temporal raw-prefix extension features than declared")

        cursor = 0
        while cursor < values.shape[0]:
            take = min(self._features_per_shard, values.shape[0] - cursor)
            piece = values[cursor : cursor + take]
            if not np.isfinite(piece).all():
                raise ValueError("temporal raw-prefix extension contains non-finite values")
            self._write_shard(
                piece,
                masks[cursor : cursor + take],
            )
            cursor += take

    def _write_shard(self, values: np.ndarray, masks: np.ndarray) -> None:
        if values.shape[0] <= 0 or values.shape[0] > self._features_per_shard:
            raise ValueError("temporal raw-prefix extension shard has an invalid row count")
        suffix = f"{self._shard_count:05d}.npy"
        try:
            np.save(
                self.temporary_path / f"extension_prefix_out_{suffix}",
                values.astype(TEMPORAL_RAW_PREFIX_STORAGE_DTYPE, copy=False),
                allow_pickle=False,
            )
            np.save(
                self.temporary_path / f"extension_prefix_mask_{suffix}",
                masks,
                allow_pickle=False,
            )
        except Exception:
            self.abort()
            raise
        self._written_count += values.shape[0]
        self._shard_rows.append(int(values.shape[0]))
        self._shard_count += 1

    def finalize(self) -> TemporalRawPrefixHistoryCacheMetadata:
        if self._finalized:
            raise RuntimeError("temporal raw-prefix history writer is already finalized")
        if self._written_count != self.plan.extension_count:
            raise ValueError(
                f"expected {self.plan.extension_count} extension features, received {self._written_count}"
            )
        shard_rows = tuple(self._shard_rows)
        metadata = TemporalRawPrefixHistoryCacheMetadata(
            model_config_name=self.base_cache.metadata.model_config_name,
            checkpoint_path=self.base_cache.metadata.checkpoint_path,
            base_cache_path=self.base_cache_path,
            task_prompts=self.manifest.task_prompts,
            input_dim=self._input_dim,
            token_count=self._token_count,
            row_count=len(self.plan.rows),
            extension_count=self.plan.extension_count,
            extension_shard_count=self._shard_count,
            extension_shard_rows=shard_rows,
            base_cache_schema_version=self.base_cache.metadata.schema_version,
        )
        try:
            np.save(
                self.temporary_path / "history_location_kind.npy",
                np.asarray(self.plan.history_location_kind, dtype=np.int8),
                allow_pickle=False,
            )
            np.save(
                self.temporary_path / "history_location_index.npy",
                np.asarray(self.plan.history_location_index, dtype=np.int64),
                allow_pickle=False,
            )
            for name, array in _extension_key_arrays(self.plan.missing_keys).items():
                np.save(self.temporary_path / f"extension_{name}.npy", array, allow_pickle=False)
            (self.temporary_path / "metadata.json").write_text(metadata.to_json() + "\n", encoding="utf-8")
            os.replace(self.temporary_path, self.output_path)
        except Exception:
            self.abort()
            raise
        self._finalized = True
        return metadata

    def abort(self) -> None:
        if not self._finalized and self.temporary_path.exists():
            shutil.rmtree(self.temporary_path, ignore_errors=True)


def save_temporal_raw_prefix_history(
    path: str | os.PathLike[str],
    *,
    manifest: _temporal_data.TemporalCompletionManifest,
    base_cache: _raw_features.RawPrefixCompletionCache,
    plan: TemporalRawPrefixHistoryPlan,
    extension_prefix_out: np.ndarray,
    extension_prefix_mask: np.ndarray,
    base_cache_path: str | os.PathLike[str],
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> TemporalRawPrefixHistoryCacheMetadata:
    """Writes an immutable sidecar from an already materialized test/helper array.

    Production extraction should use :class:`TemporalRawPrefixHistoryWriter`
    directly so missing batches are written incrementally and never collected
    into one full extension array.
    """

    values = np.asarray(extension_prefix_out)
    masks = np.asarray(extension_prefix_mask, dtype=np.bool_)
    expected_value_shape = (
        plan.extension_count,
        base_cache.metadata.token_count,
        base_cache.metadata.input_dim,
    )
    if values.shape != expected_value_shape:
        raise ValueError(f"extension_prefix_out must have shape {expected_value_shape}, got {values.shape}")
    if masks.shape != expected_value_shape[:2]:
        raise ValueError(f"extension_prefix_mask must have shape {expected_value_shape[:2]}, got {masks.shape}")
    if values.size and not np.isfinite(values).all():
        raise ValueError("extension_prefix_out contains non-finite values")

    writer = TemporalRawPrefixHistoryWriter(
        path,
        manifest=manifest,
        base_cache=base_cache,
        plan=plan,
        base_cache_path=base_cache_path,
        max_shard_bytes=max_shard_bytes,
    )
    try:
        if plan.extension_count:
            writer.append(values, masks, base_cache.prefix_segment_ids, base_cache.prefix_position_ids)
        return writer.finalize()
    except Exception:
        writer.abort()
        raise


def load_temporal_raw_prefix_history(
    path: str | os.PathLike[str],
    *,
    manifest: _temporal_data.TemporalCompletionManifest,
    base_cache: _raw_features.RawPrefixCompletionCache,
    expected_base_cache_path: str | os.PathLike[str] | None = None,
    expected_checkpoint_path: str | None = None,
    expected_model_config_name: str | None = None,
) -> TemporalRawPrefixHistoryCache:
    """Loads and validates an immutable history sidecar plus its base cache."""

    cache_path = pathlib.Path(path)
    if not cache_path.is_dir():
        raise ValueError("temporal raw-prefix history must be a directory sidecar")
    metadata_path = cache_path / "metadata.json"
    required = ("history_location_kind", "history_location_index")
    if not metadata_path.is_file() or any(not (cache_path / f"{name}.npy").is_file() for name in required):
        raise ValueError("temporal raw-prefix history lacks metadata.json or required location arrays")
    metadata = TemporalRawPrefixHistoryCacheMetadata.from_json(metadata_path.read_text(encoding="utf-8"))
    if expected_base_cache_path is not None and metadata.base_cache_path != str(pathlib.Path(expected_base_cache_path).resolve()):
        raise ValueError("temporal raw-prefix history base cache path does not match the requested base cache")
    if expected_checkpoint_path is not None and metadata.checkpoint_path != expected_checkpoint_path:
        raise ValueError("temporal raw-prefix history checkpoint does not match the requested source checkpoint")
    if expected_model_config_name is not None and metadata.model_config_name != expected_model_config_name:
        raise ValueError("temporal raw-prefix history model config does not match")

    base_cache.validate(manifest)
    location_kind = np.load(cache_path / "history_location_kind.npy", allow_pickle=False, mmap_mode="r")
    location_index = np.load(cache_path / "history_location_index.npy", allow_pickle=False, mmap_mode="r")
    extension_keys = _load_extension_keys(cache_path, metadata.extension_count)
    extension = _load_extension(cache_path, metadata)
    cache = TemporalRawPrefixHistoryCache(
        metadata=metadata,
        rows=_raw_features.manifest_rows(manifest),
        base_cache=base_cache,
        extension=extension,
        history_location_kind=location_kind,
        history_location_index=location_index,
        extension_keys=extension_keys,
    )
    cache.validate(manifest)
    return cache


class TemporalRawPrefixCompletionDataset:
    """Random-access samples returning ``[3, S, D]`` float16 histories."""

    def __init__(self, cache: TemporalRawPrefixHistoryCache, split: _temporal_data.SplitName):
        self._cache = cache
        self._indices = cache.indices_for_split(split)
        if self._indices.size == 0:
            raise ValueError(f"temporal raw-prefix history has no rows for split {split!r}")
        self.samples = tuple(cache.rows[int(index)] for index in self._indices)

    def __len__(self) -> int:
        return int(self._indices.size)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.float32]:
        cache_index = int(self._indices[int(index)])
        prefix_history, prefix_mask_history = self._cache.features_for_row(cache_index)
        return (
            prefix_history,
            prefix_mask_history,
            self._cache.prefix_segment_ids,
            self._cache.prefix_position_ids,
            np.float32(self._cache.rows[cache_index].label),
        )
