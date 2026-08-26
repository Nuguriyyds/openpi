"""Immutable current-frame raw-prefix caches for completion-head training."""

from __future__ import annotations

import bisect
from collections.abc import Sequence
import dataclasses
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any

import numpy as np

from openpi.models.completion import RAW_PREFIX_MAX_LANGUAGE_POSITION
from openpi.models.completion import RAW_PREFIX_MAX_WITHIN_SEGMENT_POSITION
from openpi.models.completion import RAW_PREFIX_SEGMENT_COUNT
from openpi.training import temporal_completion_data as _temporal_data

RAW_PREFIX_CACHE_SCHEMA_VERSION = 2
RAW_PREFIX_STORAGE_FORMAT = "directory_sharded_npy_v2"
RAW_PREFIX_PREPROCESS_PROTOCOL_VERSION = 1
RAW_PREFIX_STORAGE_DTYPE = np.dtype(np.float16)
DEFAULT_MAX_SHARD_BYTES = 1 << 30


def manifest_rows(
    manifest: _temporal_data.TemporalCompletionManifest,
) -> tuple[_temporal_data.TemporalSampleRow, ...]:
    """Returns canonical subtask-local current-frame rows in sealed order."""

    _temporal_data.validate_temporal_manifest(manifest)
    rows = tuple(
        row
        for split in _temporal_data.SPLIT_NAMES
        for row in _temporal_data.build_manifest_sample_rows(
            manifest,
            split,
            sampling_protocol="subtask_local",
        )
    )
    if any(row.sample_kind == "transition_negative" for row in rows):
        raise ValueError("raw-prefix cache rows must not contain transition negatives")
    return rows


@dataclasses.dataclass(frozen=True)
class RawPrefixCacheMetadata:
    model_config_name: str
    checkpoint_path: str
    task_prompts: tuple[str, str, str, str]
    input_dim: int
    token_count: int
    row_count: int
    feature_count: int
    shard_count: int
    schema_version: int = RAW_PREFIX_CACHE_SCHEMA_VERSION
    storage_format: str = RAW_PREFIX_STORAGE_FORMAT
    preprocess_protocol_version: int = RAW_PREFIX_PREPROCESS_PROTOCOL_VERSION
    storage_dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.schema_version != RAW_PREFIX_CACHE_SCHEMA_VERSION:
            raise ValueError(f"raw-prefix cache schema_version must be {RAW_PREFIX_CACHE_SCHEMA_VERSION}")
        if self.storage_format != RAW_PREFIX_STORAGE_FORMAT:
            raise ValueError(f"raw-prefix cache storage_format must be {RAW_PREFIX_STORAGE_FORMAT!r}")
        if self.preprocess_protocol_version != RAW_PREFIX_PREPROCESS_PROTOCOL_VERSION:
            raise ValueError("raw-prefix cache preprocessing protocol mismatch")
        if self.storage_dtype != "float16":
            raise ValueError("raw-prefix cache storage_dtype must be 'float16'")
        if not self.model_config_name or not self.checkpoint_path:
            raise ValueError("raw-prefix cache model and checkpoint bindings must be non-empty")
        if len(self.task_prompts) != _temporal_data.TASKS_PER_TRAJECTORY or any(
            not prompt.strip() for prompt in self.task_prompts
        ):
            raise ValueError("raw-prefix cache must seal four non-empty task prompts")
        if self.input_dim <= 0 or self.token_count <= 0 or self.row_count <= 0 or self.feature_count <= 0:
            raise ValueError("raw-prefix cache dimensions, row_count, and feature_count must be positive")
        if self.feature_count > self.row_count:
            raise ValueError("raw-prefix cache feature_count cannot exceed row_count")
        if self.shard_count <= 0:
            raise ValueError("raw-prefix cache shard_count must be positive")

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> RawPrefixCacheMetadata:
        payload = json.loads(value)
        expected = {field.name for field in dataclasses.fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                "raw-prefix cache metadata fields do not match schema; "
                f"missing={sorted(expected - set(payload))}, unexpected={sorted(set(payload) - expected)}"
            )
        payload["task_prompts"] = tuple(str(prompt) for prompt in payload["task_prompts"])
        return cls(**payload)


class ShardedNpyArray:
    """A small axis-0 facade over mmap-backed NPY shards."""

    def __init__(self, shards: Sequence[np.ndarray]):
        if not shards:
            raise ValueError("a sharded array requires at least one shard")
        first = shards[0]
        tail_shape = first.shape[1:]
        dtype = first.dtype
        if any(shard.ndim != first.ndim or shard.shape[1:] != tail_shape for shard in shards):
            raise ValueError("sharded arrays must agree on rank and trailing shape")
        if any(shard.dtype != dtype or shard.shape[0] <= 0 for shard in shards):
            raise ValueError("sharded arrays must agree on dtype and contain non-empty shards")
        self._shards = tuple(shards)
        self._ends = np.cumsum([int(shard.shape[0]) for shard in shards], dtype=np.int64)
        self.shape = (int(self._ends[-1]), *tail_shape)
        self.dtype = dtype
        self.ndim = first.ndim

    def _one(self, index: int) -> np.ndarray:
        if index < 0:
            index += self.shape[0]
        if index < 0 or index >= self.shape[0]:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._ends, index)
        shard_start = 0 if shard_index == 0 else int(self._ends[shard_index - 1])
        return self._shards[shard_index][index - shard_start]

    def __getitem__(self, index: Any) -> np.ndarray:
        if isinstance(index, (int, np.integer)):
            return self._one(int(index))
        if isinstance(index, slice):
            selected = np.arange(self.shape[0], dtype=np.int64)[index]
        else:
            selected = np.asarray(index)
            if selected.dtype == np.bool_:
                if selected.shape != (self.shape[0],):
                    raise IndexError("boolean shard index must have shape [feature_count]")
                selected = np.flatnonzero(selected)
            selected = selected.astype(np.int64, copy=False)
        flat = selected.reshape(-1)
        normalized = np.where(flat < 0, flat + self.shape[0], flat)
        if normalized.size and (int(normalized.min()) < 0 or int(normalized.max()) >= self.shape[0]):
            raise IndexError("sharded array index out of range")
        result = np.empty((normalized.size, *self.shape[1:]), dtype=self.dtype)
        starts = np.concatenate((np.asarray([0], dtype=np.int64), self._ends[:-1]))
        for shard_index, (start, end) in enumerate(zip(starts, self._ends, strict=True)):
            positions = np.flatnonzero((normalized >= start) & (normalized < end))
            if positions.size:
                result[positions] = self._shards[shard_index][normalized[positions] - start]
        return result.reshape((*selected.shape, *self.shape[1:]))


@dataclasses.dataclass(frozen=True)
class RawPrefixCompletionCache:
    metadata: RawPrefixCacheMetadata
    rows: tuple[_temporal_data.TemporalSampleRow, ...]
    prefix_out: ShardedNpyArray
    prefix_mask: ShardedNpyArray
    prefix_segment_ids: np.ndarray
    prefix_position_ids: np.ndarray
    row_feature_indices: np.ndarray

    def validate(self, manifest: _temporal_data.TemporalCompletionManifest) -> None:
        expected_rows = manifest_rows(manifest)
        if self.rows != expected_rows:
            raise ValueError("raw-prefix cache rows differ from the manifest's canonical subtask-local rows")
        if self.metadata.task_prompts != manifest.task_prompts:
            raise ValueError("raw-prefix cache task prompts differ from the sealed manifest")
        if self.metadata.row_count != len(self.rows):
            raise ValueError("raw-prefix cache row_count does not match row metadata")

        segment_ids = np.asarray(self.prefix_segment_ids)
        position_ids = np.asarray(self.prefix_position_ids)
        row_feature_indices = np.asarray(self.row_feature_indices)
        expected = (self.metadata.feature_count, self.metadata.token_count)
        if self.prefix_out.shape != (*expected, self.metadata.input_dim):
            raise ValueError(
                "prefix_out must have shape "
                f"{(self.metadata.feature_count, self.metadata.token_count, self.metadata.input_dim)}, "
                f"got {self.prefix_out.shape}"
            )
        if self.prefix_out.dtype != RAW_PREFIX_STORAGE_DTYPE:
            raise ValueError(f"prefix_out must be stored as float16, got {self.prefix_out.dtype}")
        if self.prefix_mask.shape != expected or self.prefix_mask.dtype != np.bool_:
            raise ValueError(
                f"prefix_mask must be bool with shape {expected}, got {self.prefix_mask.shape}/{self.prefix_mask.dtype}"
            )
        if row_feature_indices.shape != (len(self.rows),) or row_feature_indices.dtype.kind not in "iu":
            raise ValueError("row_feature_indices must be an integer array with shape [row_count]")
        if row_feature_indices.size and (
            int(row_feature_indices.min()) < 0
            or int(row_feature_indices.max()) >= self.metadata.feature_count
        ):
            raise ValueError("row_feature_indices contains an out-of-range feature index")
        if segment_ids.shape != (self.metadata.token_count,) or position_ids.shape != (self.metadata.token_count,):
            raise ValueError("raw-prefix layout ids must each have shape [S]")
        if segment_ids.dtype.kind not in "iu" or position_ids.dtype.kind not in "iu":
            raise ValueError("raw-prefix layout ids must be integer arrays")
        if segment_ids.size and (
            int(segment_ids.min()) < 0 or int(segment_ids.max()) >= RAW_PREFIX_SEGMENT_COUNT
        ):
            raise ValueError("raw-prefix segment ids must be in [0, 3]")
        if position_ids.size and (
            int(position_ids.min()) < 0 or int(position_ids.max()) >= RAW_PREFIX_MAX_WITHIN_SEGMENT_POSITION
        ):
            raise ValueError("raw-prefix position ids must be in [0, 255]")
        prompt_positions = position_ids[segment_ids == RAW_PREFIX_SEGMENT_COUNT - 1]
        if prompt_positions.size and int(prompt_positions.max()) >= RAW_PREFIX_MAX_LANGUAGE_POSITION:
            raise ValueError("raw-prefix prompt/state positions must be in [0, 199]")
    def indices_for_split(self, split: _temporal_data.SplitName) -> np.ndarray:
        if split not in _temporal_data.SPLIT_NAMES:
            raise ValueError(f"invalid temporal split {split!r}")
        return np.asarray([index for index, row in enumerate(self.rows) if row.split == split], dtype=np.int64)

    def features_for_rows(self, row_indices: np.ndarray | Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Loads only the unique feature rows needed by a training/eval batch."""

        selected = np.asarray(row_indices, dtype=np.int64)
        feature_indices = self.row_feature_indices[selected]
        return self.prefix_out[feature_indices], self.prefix_mask[feature_indices]


def _row_arrays(rows: Sequence[_temporal_data.TemporalSampleRow]) -> dict[str, np.ndarray]:
    """Persists the shared temporal row fields plus explicit current source coordinates."""

    return {
        "trajectory_id": np.asarray([row.trajectory_id for row in rows]),
        "full_episode_id": np.asarray([row.full_episode_id for row in rows], dtype=np.int64),
        "task_index": np.asarray([row.task_index for row in rows], dtype=np.int8),
        "split": np.asarray([row.split for row in rows]),
        "logical_tick": np.asarray([row.logical_tick for row in rows], dtype=np.int64),
        "label": np.asarray([row.label for row in rows], dtype=np.int8),
        "sample_kind": np.asarray([row.sample_kind for row in rows]),
        "boundary_tick": np.asarray([row.boundary_tick for row in rows], dtype=np.int64),
        "prompt_index": np.asarray([row.prompt_index for row in rows], dtype=np.int8),
        "source_episode_id": np.asarray([row.source_episode_ids[-1] for row in rows], dtype=np.int64),
        "source_frame_index": np.asarray([row.source_frame_indices[-1] for row in rows], dtype=np.int64),
        "history_logical_ticks": np.asarray([row.history_logical_ticks for row in rows], dtype=np.int64),
        "source_episode_ids": np.asarray([row.source_episode_ids for row in rows], dtype=np.int64),
        "source_frame_indices": np.asarray([row.source_frame_indices for row in rows], dtype=np.int64),
        "terminal_hold_flags": np.asarray([row.terminal_hold_flags for row in rows], dtype=np.bool_),
    }


def _rows_from_arrays(arrays: Any) -> tuple[_temporal_data.TemporalSampleRow, ...]:
    required = {
        "trajectory_id",
        "full_episode_id",
        "task_index",
        "split",
        "logical_tick",
        "label",
        "sample_kind",
        "boundary_tick",
        "prompt_index",
        "source_episode_id",
        "source_frame_index",
        "history_logical_ticks",
        "source_episode_ids",
        "source_frame_indices",
        "terminal_hold_flags",
    }
    missing = sorted(required - set(arrays.files))
    if missing:
        raise ValueError(f"raw-prefix cache is missing row arrays: {missing}")
    loaded = {name: arrays[name] for name in required}
    count = len(loaded["label"])
    if any(len(values) != count for values in loaded.values()):
        raise ValueError("raw-prefix row arrays have inconsistent row counts")
    if (
        loaded["source_episode_ids"].ndim != 2
        or loaded["source_frame_indices"].ndim != 2
        or loaded["source_episode_ids"].shape[1] != 3
        or loaded["source_frame_indices"].shape[1] != 3
    ):
        raise ValueError("raw-prefix source history arrays must have shape [N, 3]")
    if not np.array_equal(loaded["source_episode_id"], loaded["source_episode_ids"][:, -1]):
        raise ValueError("raw-prefix current source episode ids disagree with source history")
    if not np.array_equal(loaded["source_frame_index"], loaded["source_frame_indices"][:, -1]):
        raise ValueError("raw-prefix current source frame indices disagree with source history")
    return tuple(
        _temporal_data.TemporalSampleRow(
            trajectory_id=str(loaded["trajectory_id"][index]),
            full_episode_id=int(loaded["full_episode_id"][index]),
            task_index=int(loaded["task_index"][index]),
            split=str(loaded["split"][index]),
            logical_tick=int(loaded["logical_tick"][index]),
            label=int(loaded["label"][index]),
            sample_kind=str(loaded["sample_kind"][index]),
            boundary_tick=int(loaded["boundary_tick"][index]),
            prompt_index=int(loaded["prompt_index"][index]),
            history_logical_ticks=tuple(int(value) for value in loaded["history_logical_ticks"][index]),
            source_episode_ids=tuple(int(value) for value in loaded["source_episode_ids"][index]),
            source_frame_indices=tuple(int(value) for value in loaded["source_frame_indices"][index]),
            terminal_hold_flags=tuple(bool(value) for value in loaded["terminal_hold_flags"][index]),
        )
        for index in range(count)
    )


class RawPrefixCacheWriter:
    """Streams unique raw-prefix features into bounded NPY shards."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        manifest: _temporal_data.TemporalCompletionManifest,
        row_feature_indices: np.ndarray,
        feature_count: int,
        model_config_name: str,
        checkpoint_path: str,
        max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    ) -> None:
        self.rows = manifest_rows(manifest)
        self.manifest = manifest
        self.output_path = pathlib.Path(path)
        if self.output_path.exists():
            raise FileExistsError(f"refusing to overwrite sealed raw-prefix cache: {self.output_path}")
        if feature_count <= 0 or feature_count > len(self.rows):
            raise ValueError("feature_count must be in [1, row_count]")
        mapping = np.asarray(row_feature_indices, dtype=np.int64)
        if mapping.shape != (len(self.rows),):
            raise ValueError(f"row_feature_indices must have shape {(len(self.rows),)}, got {mapping.shape}")
        if mapping.size and (int(mapping.min()) < 0 or int(mapping.max()) >= feature_count):
            raise ValueError("row_feature_indices contains an out-of-range feature index")
        if max_shard_bytes <= 0:
            raise ValueError("max_shard_bytes must be positive")
        self.row_feature_indices = mapping
        self.feature_count = int(feature_count)
        self.model_config_name = model_config_name
        self.checkpoint_path = checkpoint_path
        self.max_shard_bytes = int(max_shard_bytes)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary_path = pathlib.Path(
            tempfile.mkdtemp(prefix=f".{self.output_path.name}.{os.getpid()}.", dir=self.output_path.parent)
        )
        self._token_count: int | None = None
        self._input_dim: int | None = None
        self._segment_ids: np.ndarray | None = None
        self._position_ids: np.ndarray | None = None
        self._values_buffer: np.ndarray | None = None
        self._mask_buffer: np.ndarray | None = None
        self._buffer_count = 0
        self._written_count = 0
        self._shard_count = 0
        self._finalized = False

    def _initialize_layout(
        self,
        values: np.ndarray,
        segment_ids: np.ndarray,
        position_ids: np.ndarray,
    ) -> None:
        token_count = int(values.shape[1])
        input_dim = int(values.shape[2])
        if token_count <= 0 or input_dim <= 0:
            raise ValueError("raw-prefix token count and input dim must be positive")
        if segment_ids.shape != (token_count,) or position_ids.shape != (token_count,):
            raise ValueError("raw-prefix layout ids must each have shape [S]")
        bytes_per_feature = token_count * (input_dim * RAW_PREFIX_STORAGE_DTYPE.itemsize + np.dtype(np.bool_).itemsize)
        features_per_shard = max(1, min(self.feature_count, self.max_shard_bytes // bytes_per_feature))
        self._token_count = token_count
        self._input_dim = input_dim
        self._segment_ids = segment_ids.copy()
        self._position_ids = position_ids.copy()
        self._values_buffer = np.empty(
            (features_per_shard, token_count, input_dim),
            dtype=RAW_PREFIX_STORAGE_DTYPE,
        )
        self._mask_buffer = np.empty((features_per_shard, token_count), dtype=np.bool_)

    def append(
        self,
        prefix_out: np.ndarray,
        prefix_mask: np.ndarray,
        prefix_segment_ids: np.ndarray,
        prefix_position_ids: np.ndarray,
    ) -> None:
        if self._finalized:
            raise RuntimeError("cannot append to a finalized raw-prefix cache")
        values = np.asarray(prefix_out)
        mask = np.asarray(prefix_mask, dtype=np.bool_)
        segment_ids = np.asarray(prefix_segment_ids, dtype=np.int32)
        position_ids = np.asarray(prefix_position_ids, dtype=np.int32)
        if values.ndim != 3 or values.shape[0] <= 0 or values.shape[-1] <= 0:
            raise ValueError(f"prefix_out must have shape [batch, S, D], got {values.shape}")
        if mask.shape != values.shape[:2]:
            raise ValueError(f"prefix_mask must have shape {values.shape[:2]}, got {mask.shape}")
        if self._values_buffer is None:
            self._initialize_layout(values, segment_ids, position_ids)
        assert self._values_buffer is not None
        assert self._mask_buffer is not None
        assert self._segment_ids is not None
        assert self._position_ids is not None
        if values.shape[1:] != self._values_buffer.shape[1:]:
            raise ValueError("raw-prefix shape changed between extraction batches")
        if not np.array_equal(segment_ids, self._segment_ids) or not np.array_equal(position_ids, self._position_ids):
            raise ValueError("raw-prefix layout changed between extraction batches")
        if self._written_count + self._buffer_count + values.shape[0] > self.feature_count:
            raise ValueError("received more raw-prefix features than declared")

        cursor = 0
        while cursor < values.shape[0]:
            available = self._values_buffer.shape[0] - self._buffer_count
            take = min(available, values.shape[0] - cursor)
            piece = values[cursor : cursor + take]
            if not np.isfinite(piece).all():
                raise ValueError("prefix_out contains non-finite values")
            stop = self._buffer_count + take
            self._values_buffer[self._buffer_count : stop] = piece.astype(RAW_PREFIX_STORAGE_DTYPE, copy=False)
            self._mask_buffer[self._buffer_count : stop] = mask[cursor : cursor + take]
            self._buffer_count = stop
            cursor += take
            if self._buffer_count == self._values_buffer.shape[0]:
                self._flush()

    def _flush(self) -> None:
        if self._buffer_count == 0:
            return
        assert self._values_buffer is not None
        assert self._mask_buffer is not None
        suffix = f"{self._shard_count:05d}.npy"
        try:
            np.save(
                self.temporary_path / f"prefix_out_{suffix}",
                self._values_buffer[: self._buffer_count],
                allow_pickle=False,
            )
            np.save(
                self.temporary_path / f"prefix_mask_{suffix}",
                self._mask_buffer[: self._buffer_count],
                allow_pickle=False,
            )
        except Exception:
            self.abort()
            raise
        self._written_count += self._buffer_count
        self._buffer_count = 0
        self._shard_count += 1

    def finalize(self) -> RawPrefixCacheMetadata:
        if self._finalized:
            raise RuntimeError("raw-prefix cache writer is already finalized")
        self._flush()
        if self._written_count != self.feature_count:
            raise ValueError(f"expected {self.feature_count} features, received {self._written_count}")
        assert self._token_count is not None
        assert self._input_dim is not None
        assert self._segment_ids is not None
        assert self._position_ids is not None
        metadata = RawPrefixCacheMetadata(
            model_config_name=self.model_config_name,
            checkpoint_path=self.checkpoint_path,
            task_prompts=self.manifest.task_prompts,
            input_dim=self._input_dim,
            token_count=self._token_count,
            row_count=len(self.rows),
            feature_count=self.feature_count,
            shard_count=self._shard_count,
        )
        try:
            np.save(self.temporary_path / "prefix_segment_ids.npy", self._segment_ids, allow_pickle=False)
            np.save(self.temporary_path / "prefix_position_ids.npy", self._position_ids, allow_pickle=False)
            np.save(self.temporary_path / "row_feature_indices.npy", self.row_feature_indices, allow_pickle=False)
            for name, array in _row_arrays(self.rows).items():
                np.save(self.temporary_path / f"{name}.npy", array, allow_pickle=False)
            (self.temporary_path / "metadata.json").write_text(metadata.to_json() + "\n", encoding="utf-8")
            os.replace(self.temporary_path, self.output_path)
        except Exception:
            self.abort()
            raise
        self._finalized = True
        return metadata

    def abort(self) -> None:
        if not self._finalized and self.temporary_path.exists():
            # Cleanup must not hide the extraction/write exception that caused the abort.
            shutil.rmtree(self.temporary_path, ignore_errors=True)


def save_raw_prefix_cache(
    path: str | os.PathLike[str],
    *,
    manifest: _temporal_data.TemporalCompletionManifest,
    prefix_out: np.ndarray,
    prefix_mask: np.ndarray,
    prefix_segment_ids: np.ndarray,
    prefix_position_ids: np.ndarray,
    model_config_name: str,
    checkpoint_path: str,
    row_feature_indices: np.ndarray | None = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> RawPrefixCacheMetadata:
    """Writes an immutable cache while bounding every feature shard."""

    rows = manifest_rows(manifest)
    values = np.asarray(prefix_out)
    if values.ndim != 3:
        raise ValueError(f"prefix_out must have shape [feature_count, S, D], got {values.shape}")
    mapping = (
        np.arange(len(rows), dtype=np.int64)
        if row_feature_indices is None
        else np.asarray(row_feature_indices, dtype=np.int64)
    )
    if row_feature_indices is None and values.shape[0] != len(rows):
        raise ValueError(f"prefix_out must have {len(rows)} row-aligned features, got {values.shape[0]}")
    writer = RawPrefixCacheWriter(
        path,
        manifest=manifest,
        row_feature_indices=mapping,
        feature_count=int(values.shape[0]),
        model_config_name=model_config_name,
        checkpoint_path=checkpoint_path,
        max_shard_bytes=max_shard_bytes,
    )
    try:
        writer.append(prefix_out, prefix_mask, prefix_segment_ids, prefix_position_ids)
        return writer.finalize()
    except Exception:
        writer.abort()
        raise


class _NpyArrayDirectory:
    def __init__(self, root: pathlib.Path):
        self._root = root
        self.files = tuple(path.stem for path in root.glob("*.npy"))

    def __getitem__(self, name: str) -> np.ndarray:
        return np.load(self._root / f"{name}.npy", allow_pickle=False, mmap_mode="r")


def load_raw_prefix_cache(
    path: str | os.PathLike[str],
    *,
    manifest: _temporal_data.TemporalCompletionManifest,
    expected_checkpoint_path: str | None = None,
    expected_model_config_name: str | None = None,
) -> RawPrefixCompletionCache:
    """Loads and validates an immutable raw-prefix cache."""

    cache_path = pathlib.Path(path)
    if not cache_path.is_dir():
        raise ValueError("raw-prefix cache must be a directory cache")
    metadata_path = cache_path / "metadata.json"
    required_arrays = ("prefix_segment_ids", "prefix_position_ids", "row_feature_indices")
    if not metadata_path.is_file() or any(not (cache_path / f"{name}.npy").is_file() for name in required_arrays):
        raise ValueError("raw-prefix cache lacks metadata.json or required arrays")
    metadata = RawPrefixCacheMetadata.from_json(metadata_path.read_text(encoding="utf-8"))
    prefix_paths = tuple(sorted(cache_path.glob("prefix_out_[0-9][0-9][0-9][0-9][0-9].npy")))
    mask_paths = tuple(sorted(cache_path.glob("prefix_mask_[0-9][0-9][0-9][0-9][0-9].npy")))
    if len(prefix_paths) != metadata.shard_count or len(mask_paths) != metadata.shard_count:
        raise ValueError("raw-prefix cache shard count does not match metadata")
    arrays = _NpyArrayDirectory(cache_path)
    rows = _rows_from_arrays(arrays)
    cache = RawPrefixCompletionCache(
        metadata=metadata,
        rows=rows,
        prefix_out=ShardedNpyArray(
            tuple(np.load(shard, allow_pickle=False, mmap_mode="r") for shard in prefix_paths)
        ),
        prefix_mask=ShardedNpyArray(
            tuple(np.load(shard, allow_pickle=False, mmap_mode="r") for shard in mask_paths)
        ),
        prefix_segment_ids=np.load(cache_path / "prefix_segment_ids.npy", allow_pickle=False, mmap_mode="r"),
        prefix_position_ids=np.load(cache_path / "prefix_position_ids.npy", allow_pickle=False, mmap_mode="r"),
        row_feature_indices=np.load(cache_path / "row_feature_indices.npy", allow_pickle=False, mmap_mode="r"),
    )
    cache.validate(manifest)
    if expected_checkpoint_path is not None and metadata.checkpoint_path != expected_checkpoint_path:
        raise ValueError("raw-prefix cache checkpoint path does not match the requested source checkpoint")
    if expected_model_config_name is not None and metadata.model_config_name != expected_model_config_name:
        raise ValueError("raw-prefix cache model config does not match")
    return cache


class RawPrefixCompletionDataset:
    """Random-access current-frame samples that preserve on-disk float16."""

    def __init__(self, cache: RawPrefixCompletionCache, split: _temporal_data.SplitName):
        self._cache = cache
        self._indices = cache.indices_for_split(split)
        if self._indices.size == 0:
            raise ValueError(f"raw-prefix cache has no rows for split {split!r}")
        self.samples = tuple(cache.rows[int(index)] for index in self._indices)

    def __len__(self) -> int:
        return int(self._indices.size)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.float32]:
        cache_index = int(self._indices[int(index)])
        feature_index = int(self._cache.row_feature_indices[cache_index])
        return (
            self._cache.prefix_out[feature_index],
            self._cache.prefix_mask[feature_index],
            self._cache.prefix_segment_ids,
            self._cache.prefix_position_ids,
            np.float32(self._cache.rows[cache_index].label),
        )
