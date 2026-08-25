"""Immutable current-frame raw-prefix caches for completion-head training."""

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

from openpi.models.completion import RAW_PREFIX_MAX_LANGUAGE_POSITION
from openpi.models.completion import RAW_PREFIX_MAX_WITHIN_SEGMENT_POSITION
from openpi.models.completion import RAW_PREFIX_SEGMENT_COUNT
from openpi.training import temporal_completion_data as _temporal_data

RAW_PREFIX_CACHE_SCHEMA_VERSION = 1
RAW_PREFIX_STORAGE_FORMAT = "directory_npy_v1"
RAW_PREFIX_PREPROCESS_PROTOCOL_VERSION = 1
RAW_PREFIX_STORAGE_DTYPE = np.dtype(np.float16)


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
        if self.input_dim <= 0 or self.token_count <= 0 or self.row_count <= 0:
            raise ValueError("raw-prefix cache dimensions and row_count must be positive")

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> "RawPrefixCacheMetadata":
        payload = json.loads(value)
        expected = {field.name for field in dataclasses.fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                "raw-prefix cache metadata fields do not match schema; "
                f"missing={sorted(expected - set(payload))}, unexpected={sorted(set(payload) - expected)}"
            )
        payload["task_prompts"] = tuple(str(prompt) for prompt in payload["task_prompts"])
        return cls(**payload)


@dataclasses.dataclass(frozen=True)
class RawPrefixCompletionCache:
    metadata: RawPrefixCacheMetadata
    rows: tuple[_temporal_data.TemporalSampleRow, ...]
    prefix_out: np.ndarray
    prefix_mask: np.ndarray
    prefix_segment_ids: np.ndarray
    prefix_position_ids: np.ndarray

    def validate(self, manifest: _temporal_data.TemporalCompletionManifest) -> None:
        expected_rows = manifest_rows(manifest)
        if self.rows != expected_rows:
            raise ValueError("raw-prefix cache rows differ from the manifest's canonical subtask-local rows")
        if self.metadata.task_prompts != manifest.task_prompts:
            raise ValueError("raw-prefix cache task prompts differ from the sealed manifest")
        if self.metadata.row_count != len(self.rows):
            raise ValueError("raw-prefix cache row_count does not match row metadata")

        prefix_out = np.asarray(self.prefix_out)
        prefix_mask = np.asarray(self.prefix_mask)
        segment_ids = np.asarray(self.prefix_segment_ids)
        position_ids = np.asarray(self.prefix_position_ids)
        expected = (len(self.rows), self.metadata.token_count)
        if prefix_out.shape != (*expected, self.metadata.input_dim):
            raise ValueError(
                f"prefix_out must have shape {(len(self.rows), self.metadata.token_count, self.metadata.input_dim)}, "
                f"got {prefix_out.shape}"
            )
        if prefix_out.dtype != RAW_PREFIX_STORAGE_DTYPE:
            raise ValueError(f"prefix_out must be stored as float16, got {prefix_out.dtype}")
        if prefix_mask.shape != expected or prefix_mask.dtype != np.bool_:
            raise ValueError(f"prefix_mask must be bool with shape {expected}, got {prefix_mask.shape}/{prefix_mask.dtype}")
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
        if not np.isfinite(prefix_out).all():
            raise ValueError("raw-prefix cache contains non-finite prefix values")

    def indices_for_split(self, split: _temporal_data.SplitName) -> np.ndarray:
        if split not in _temporal_data.SPLIT_NAMES:
            raise ValueError(f"invalid temporal split {split!r}")
        return np.asarray([index for index, row in enumerate(self.rows) if row.split == split], dtype=np.int64)


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
) -> RawPrefixCacheMetadata:
    """Writes one immutable current-frame raw-prefix cache."""

    rows = manifest_rows(manifest)
    values = np.asarray(prefix_out)
    if values.ndim != 3 or values.shape[0] != len(rows) or values.shape[-1] <= 0:
        raise ValueError(f"prefix_out must have shape [{len(rows)}, S, D], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("prefix_out contains non-finite values")
    stored_values = values.astype(np.float16, copy=False)
    mask = np.asarray(prefix_mask, dtype=np.bool_)
    if mask.shape != values.shape[:2]:
        raise ValueError(f"prefix_mask must have shape {values.shape[:2]}, got {mask.shape}")
    segment_ids = np.asarray(prefix_segment_ids, dtype=np.int32)
    position_ids = np.asarray(prefix_position_ids, dtype=np.int32)
    metadata = RawPrefixCacheMetadata(
        model_config_name=model_config_name,
        checkpoint_path=checkpoint_path,
        task_prompts=manifest.task_prompts,
        input_dim=int(values.shape[-1]),
        token_count=int(values.shape[1]),
        row_count=len(rows),
    )
    cache = RawPrefixCompletionCache(
        metadata=metadata,
        rows=rows,
        prefix_out=stored_values,
        prefix_mask=mask,
        prefix_segment_ids=segment_ids,
        prefix_position_ids=position_ids,
    )
    cache.validate(manifest)

    output_path = pathlib.Path(path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite sealed raw-prefix cache: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = pathlib.Path(tempfile.mkdtemp(prefix=f".{output_path.name}.{os.getpid()}.", dir=output_path.parent))
    try:
        (temporary_path / "metadata.json").write_text(metadata.to_json() + "\n", encoding="utf-8")
        np.save(temporary_path / "prefix_out.npy", stored_values, allow_pickle=False)
        np.save(temporary_path / "prefix_mask.npy", mask, allow_pickle=False)
        np.save(temporary_path / "prefix_segment_ids.npy", segment_ids, allow_pickle=False)
        np.save(temporary_path / "prefix_position_ids.npy", position_ids, allow_pickle=False)
        for name, array in _row_arrays(rows).items():
            np.save(temporary_path / f"{name}.npy", array, allow_pickle=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
    return metadata


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
    required_arrays = ("prefix_out", "prefix_mask", "prefix_segment_ids", "prefix_position_ids")
    if not metadata_path.is_file() or any(not (cache_path / f"{name}.npy").is_file() for name in required_arrays):
        raise ValueError("raw-prefix cache lacks metadata.json or required arrays")
    metadata = RawPrefixCacheMetadata.from_json(metadata_path.read_text(encoding="utf-8"))
    arrays = _NpyArrayDirectory(cache_path)
    rows = _rows_from_arrays(arrays)
    cache = RawPrefixCompletionCache(
        metadata=metadata,
        rows=rows,
        prefix_out=np.load(cache_path / "prefix_out.npy", allow_pickle=False, mmap_mode="r"),
        prefix_mask=np.load(cache_path / "prefix_mask.npy", allow_pickle=False, mmap_mode="r"),
        prefix_segment_ids=np.load(cache_path / "prefix_segment_ids.npy", allow_pickle=False, mmap_mode="r"),
        prefix_position_ids=np.load(cache_path / "prefix_position_ids.npy", allow_pickle=False, mmap_mode="r"),
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
        return (
            self._cache.prefix_out[cache_index],
            self._cache.prefix_mask[cache_index],
            self._cache.prefix_segment_ids,
            self._cache.prefix_position_ids,
            np.float32(self._cache.rows[cache_index].label),
        )
