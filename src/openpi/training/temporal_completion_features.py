"""Versioned frozen-prefix feature caches for temporal completion training."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
import os
import pathlib
from typing import Any

import numpy as np

import openpi.training.temporal_completion_data as _temporal_data

FEATURE_CACHE_SCHEMA_VERSION = 3
PREPROCESS_PROTOCOL_VERSION = 1
POOLING_METHOD = "masked_mean_fp32"
SUPPORTED_FEATURE_DTYPES = (np.dtype(np.float16), np.dtype(np.float32))


def manifest_rows(
    manifest: _temporal_data.TemporalCompletionManifest,
) -> tuple[_temporal_data.TemporalSampleRow, ...]:
    """Returns the canonical natural row order sealed into every cache."""

    return tuple(
        row
        for split in _temporal_data.SPLIT_NAMES
        for row in _temporal_data.build_manifest_sample_rows(manifest, split)
    )


@dataclasses.dataclass(frozen=True)
class TemporalFeatureCacheMetadata:
    model_config_name: str
    checkpoint_path: str
    task_prompts: tuple[str, str, str, str]
    feature_dim: int
    row_count: int
    schema_version: int = FEATURE_CACHE_SCHEMA_VERSION
    preprocess_protocol_version: int = PREPROCESS_PROTOCOL_VERSION
    pooling_method: str = POOLING_METHOD
    history_steps: int = _temporal_data.TEMPORAL_HISTORY_STEPS
    fps: int = _temporal_data.FPS
    tick_stride_frames: int = _temporal_data.TICK_STRIDE_FRAMES

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"feature cache schema_version must be {FEATURE_CACHE_SCHEMA_VERSION}, got {self.schema_version}"
            )
        if self.preprocess_protocol_version != PREPROCESS_PROTOCOL_VERSION:
            raise ValueError(
                "feature cache preprocessing protocol mismatch: "
                f"expected {PREPROCESS_PROTOCOL_VERSION}, got {self.preprocess_protocol_version}"
            )
        if self.pooling_method != POOLING_METHOD:
            raise ValueError(f"feature cache pooling_method must be {POOLING_METHOD!r}")
        if not self.model_config_name:
            raise ValueError("feature cache model_config_name must not be empty")
        if not self.checkpoint_path:
            raise ValueError("feature cache checkpoint_path must not be empty")
        if len(self.task_prompts) != 4 or any(not prompt.strip() for prompt in self.task_prompts):
            raise ValueError("feature cache must seal exactly four non-empty task prompts")
        if len(set(self.task_prompts)) != 4:
            raise ValueError("feature cache task prompts must be distinct")
        if self.feature_dim <= 0 or self.row_count <= 0:
            raise ValueError("feature cache feature_dim and row_count must be positive")
        if self.history_steps != 3:
            raise ValueError("feature cache must contain exactly three temporal features")
        if self.fps != 30 or self.tick_stride_frames != 15:
            raise ValueError("feature cache timing must be 30 fps with a 15-frame stride")

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> TemporalFeatureCacheMetadata:
        payload = json.loads(value)
        expected = {field.name for field in dataclasses.fields(cls)}
        if set(payload) != expected:
            raise ValueError(
                "feature cache metadata fields do not match schema; "
                f"missing={sorted(expected - set(payload))}, unexpected={sorted(set(payload) - expected)}"
            )
        payload["task_prompts"] = tuple(str(prompt) for prompt in payload["task_prompts"])
        return cls(**payload)


@dataclasses.dataclass(frozen=True)
class TemporalFeatureCache:
    metadata: TemporalFeatureCacheMetadata
    rows: tuple[_temporal_data.TemporalSampleRow, ...]
    prefix_history: np.ndarray

    def validate(self, manifest: _temporal_data.TemporalCompletionManifest) -> None:
        _temporal_data.validate_temporal_manifest(manifest)
        expected_rows = manifest_rows(manifest)
        if self.metadata.task_prompts != manifest.task_prompts:
            raise ValueError("feature cache task prompts differ from the sealed temporal manifest")
        if self.rows != expected_rows:
            raise ValueError("feature cache rows differ from the manifest's canonical natural candidate rows")
        if self.metadata.row_count != len(self.rows):
            raise ValueError("feature cache row_count does not match row metadata")
        history = np.asarray(self.prefix_history)
        expected_shape = (len(self.rows), 3, self.metadata.feature_dim)
        if history.shape != expected_shape:
            raise ValueError(f"prefix_history must have shape {expected_shape}, got {history.shape}")
        if history.dtype not in SUPPORTED_FEATURE_DTYPES:
            raise ValueError(f"prefix_history must be float16 or float32, got {history.dtype}")
        if not np.isfinite(history).all():
            raise ValueError("prefix_history contains non-finite values")

    def indices_for_split(self, split: _temporal_data.SplitName) -> np.ndarray:
        if split not in _temporal_data.SPLIT_NAMES:
            raise ValueError(f"invalid temporal split {split!r}")
        return np.asarray([index for index, row in enumerate(self.rows) if row.split == split], dtype=np.int64)


def _row_arrays(rows: Sequence[_temporal_data.TemporalSampleRow]) -> dict[str, np.ndarray]:
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
        "history_logical_ticks",
        "source_episode_ids",
        "source_frame_indices",
        "terminal_hold_flags",
    }
    missing = sorted(required - set(arrays.files))
    if missing:
        raise ValueError(f"temporal feature cache is missing row arrays: {missing}")
    # NpzFile does not memoize member access.  Materialise every column once;
    # indexing ``arrays[name]`` inside the row loop would reread an entire
    # member O(N) times and make production cache loading effectively O(N^2).
    loaded = {name: arrays[name] for name in required}
    count = len(loaded["label"])
    for name in required:
        if len(loaded[name]) != count:
            raise ValueError(f"temporal feature cache array {name!r} has inconsistent row count")
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


def save_temporal_feature_cache(
    path: str | os.PathLike[str],
    *,
    manifest: _temporal_data.TemporalCompletionManifest,
    prefix_history: np.ndarray,
    model_config_name: str,
    checkpoint_path: str,
) -> TemporalFeatureCacheMetadata:
    """Atomically writes one immutable cache in canonical manifest row order."""

    rows = manifest_rows(manifest)
    history = np.asarray(prefix_history)
    if history.ndim != 3 or history.shape[:2] != (len(rows), 3):
        raise ValueError(f"prefix_history must have shape [{len(rows)}, 3, D], got {history.shape}")
    metadata = TemporalFeatureCacheMetadata(
        model_config_name=model_config_name,
        checkpoint_path=checkpoint_path,
        task_prompts=manifest.task_prompts,
        feature_dim=int(history.shape[-1]),
        row_count=len(rows),
    )
    cache = TemporalFeatureCache(metadata=metadata, rows=rows, prefix_history=history)
    cache.validate(manifest)

    output_path = pathlib.Path(path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite sealed temporal feature cache: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    with temporary_path.open("wb") as file:
        np.savez(file, metadata_json=np.asarray(metadata.to_json()), prefix_history=history, **_row_arrays(rows))
    os.replace(temporary_path, output_path)
    return metadata


def load_temporal_feature_cache(
    path: str | os.PathLike[str],
    *,
    manifest: _temporal_data.TemporalCompletionManifest,
    expected_checkpoint_path: str | None = None,
    expected_model_config_name: str | None = None,
) -> TemporalFeatureCache:
    """Loads and fully audits a cache before exposing any training samples."""

    with np.load(path, allow_pickle=False) as arrays:
        if "metadata_json" not in arrays.files or "prefix_history" not in arrays.files:
            raise ValueError("temporal feature cache lacks metadata_json or prefix_history")
        metadata_value = arrays["metadata_json"]
        if metadata_value.ndim != 0:
            raise ValueError("feature cache metadata_json must be a scalar string")
        metadata = TemporalFeatureCacheMetadata.from_json(str(metadata_value.item()))
        rows = _rows_from_arrays(arrays)
        # NpzFile materialises an owning ndarray for each member.  Retain that
        # array directly; an additional copy briefly doubles a GB-scale cache.
        prefix_history = arrays["prefix_history"]
    cache = TemporalFeatureCache(metadata=metadata, rows=rows, prefix_history=prefix_history)
    cache.validate(manifest)
    if expected_checkpoint_path is not None and metadata.checkpoint_path != expected_checkpoint_path:
        raise ValueError("temporal feature cache checkpoint path does not match the requested clean checkpoint")
    if expected_model_config_name is not None and metadata.model_config_name != expected_model_config_name:
        raise ValueError("temporal feature cache model config does not match")
    return cache


class TemporalFeatureDataset:
    """Random-access FP32 histories for one natural trajectory split."""

    def __init__(self, cache: TemporalFeatureCache, split: _temporal_data.SplitName) -> None:
        self._cache = cache
        self._indices = cache.indices_for_split(split)
        if self._indices.size == 0:
            raise ValueError(f"temporal feature cache has no rows for split {split!r}")
        self.samples = tuple(cache.rows[int(index)] for index in self._indices)

    def __len__(self) -> int:
        return int(self._indices.size)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.float32]:
        cache_index = int(self._indices[int(index)])
        history = np.asarray(self._cache.prefix_history[cache_index], dtype=np.float32)
        target = np.float32(self._cache.rows[cache_index].label)
        return history, target
