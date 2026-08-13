"""Generate a boundary-labeled copy of a LeRobot dataset with strict 0/1 labels.

For each 4-episode task group [e0, e1, e2, e3] (subtasks 1/2/3/4):

* e0/e1/e2 (subtasks 1/2/3): the last 5 original frames are labeled
  ``completion=1``; the next subtask's first 5 frames are appended as
  boundary copies — relabeled with the *current* subtask's ``task_index``,
  ``completion=1`` — giving exactly 10 positives.  The next subtask's
  original first-5 frames keep their own ``task_index`` and ``completion=0``
  (same observation, different prompt, opposite label).
* e3 (subtask 4): the last 10 frames are labeled ``completion=1``; no copies.

The original dataset is never modified — all output is written to
``--output-root``.  Videos for subtasks 1/2/3 are extended via ffmpeg
(current video + next video's first 5 frames).  Parquet rows, video frames,
``frame_index``, ``timestamp``, ``index``, ``episodes.jsonl``, and
``info.json`` are all rebuilt consistently and audited.

Produces ``label_audit.json`` with train/test pos/neg counts, ratios,
per-subtask counts, boundary copy counts, and consistency checks.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from openpi.training.completion_data import BOUNDARY_COPY_FRAMES
from openpi.training.completion_data import BOUNDARY_GROUP
from openpi.training.completion_data import BOUNDARY_LABEL_SCHEME
from openpi.training.completion_data import BOUNDARY_POSITIVE_TAIL
from openpi.training.completion_data import BOUNDARY_SUBTASK4_TAIL
from openpi.training.completion_data import BOUNDARY_TOTAL_POSITIVES
from openpi.training.completion_data import audit_boundary_completion_episode_parquet
from openpi.training.completion_data import boundary_train_sample_indices
from openpi.training.completion_data import build_task_groups
from openpi.training.completion_data import create_split_manifest

DEFAULT_SRC_ROOT = "/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730"

COPY_FRAMES = BOUNDARY_COPY_FRAMES  # 5
POSITIVE_TAIL = BOUNDARY_POSITIVE_TAIL  # 5
SUBTASK4_TAIL = BOUNDARY_SUBTASK4_TAIL  # 10
TOTAL_POSITIVES = BOUNDARY_TOTAL_POSITIVES  # 10
GROUP_SIZE = BOUNDARY_GROUP  # 4

BOUNDARY_FEATURES = {
    "completion": {"dtype": "float32", "shape": [1], "names": None},
    "source_episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "source_frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "is_boundary_copy": {"dtype": "int8", "shape": [1], "names": None},
}

BOUNDARY_HF_FEATURES = {
    "completion": {"dtype": "float32", "_type": "Value"},
    "source_episode_index": {"dtype": "int64", "_type": "Value"},
    "source_frame_index": {"dtype": "int64", "_type": "Value"},
    "is_boundary_copy": {"dtype": "int8", "_type": "Value"},
}

INDEX_COLUMNS = ["frame_index", "timestamp", "episode_index", "index", "task_index"]
NEW_COLUMNS = ["completion", "source_episode_index", "source_frame_index", "is_boundary_copy"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_episode_lengths(meta_dir: pathlib.Path) -> dict[int, int]:
    """Reads ``episodes.jsonl`` and returns ``{episode_index: length}``."""

    lengths: dict[int, int] = {}
    for line in (meta_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        lengths[int(record["episode_index"])] = int(record["length"])
    return lengths


def read_info(meta_dir: pathlib.Path) -> dict:
    return json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))


def get_video_keys(info: dict) -> list[str]:
    return sorted(k for k, v in info.get("features", {}).items() if v.get("dtype") == "video")


def get_episode_data_path(root: pathlib.Path, episode_id: int, chunks_size: int) -> pathlib.Path:
    chunk = episode_id // chunks_size
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_id:06d}.parquet"


def get_episode_video_path(root: pathlib.Path, episode_id: int, key: str, chunks_size: int) -> pathlib.Path:
    chunk = episode_id // chunks_size
    return root / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{episode_id:06d}.mp4"


def count_video_frames(video_path: pathlib.Path) -> int:
    """Counts frames in a video file using ffprobe."""

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    val = result.stdout.strip()
    if val and val != "N/A":
        return int(val)
    # Fallback: count packets.
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def format_ratio(positive: int, negative: int) -> dict:
    total = positive + negative
    ratio_str = f"1:{negative / positive:.2f}" if positive > 0 else "N/A"
    positive_pct = round(positive / total * 100, 4) if total > 0 else 0.0
    return {
        "positive": positive,
        "negative": negative,
        "total": total,
        "ratio": ratio_str,
        "positive_pct": positive_pct,
    }


def update_boundary_hf_schema_metadata(table: pa.Table) -> pa.Table:
    """Injects boundary fields into the parquet ``huggingface`` schema metadata."""

    schema_metadata = table.schema.metadata
    if schema_metadata is None:
        hf_meta: dict = {"info": {"features": {k: dict(v) for k, v in BOUNDARY_HF_FEATURES.items()}}}
    else:
        existing = {k.decode() if isinstance(k, bytes) else k: v for k, v in schema_metadata.items()}
        raw = existing.get("huggingface", b"{}")
        if isinstance(raw, bytes):
            raw = raw.decode()
        hf_meta = json.loads(raw)
        features = hf_meta.setdefault("info", {}).setdefault("features", {})
        for name, feat in BOUNDARY_HF_FEATURES.items():
            features[name] = dict(feat)
    new_metadata = {b"huggingface": json.dumps(hf_meta).encode()}
    if schema_metadata:
        for key, value in schema_metadata.items():
            key_str = key.decode() if isinstance(key, bytes) else key
            if key_str != "huggingface":
                new_metadata[key if isinstance(key, bytes) else key.encode()] = value
    return table.replace_schema_metadata(new_metadata)


# ---------------------------------------------------------------------------
# Parquet processing
# ---------------------------------------------------------------------------


def process_boundary_parquet(
    src_path: pathlib.Path,
    dst_path: pathlib.Path,
    *,
    episode_id: int,
    group_position: int,
    task_index: int,
    next_src_path: pathlib.Path | None,
    global_offset: int,
    old_length: int,
    fps: float,
    force: bool,
) -> dict[str, int]:
    """Reads a source parquet, appends boundary copy rows + new columns, writes ``dst_path``."""

    if dst_path.exists() and not force:
        existing = pq.ParquetFile(dst_path)
        if all(col in existing.schema_arrow.names for col in NEW_COLUMNS):
            return {"skipped": 1, "written": 0}

    table = pq.read_table(src_path)
    num_rows = table.num_rows
    if num_rows != old_length:
        raise ValueError(f"{src_path.name}: parquet has {num_rows} rows but episodes.jsonl length is {old_length}")

    # Drop any pre-existing boundary columns so the copy is clean.
    for col in NEW_COLUMNS:
        if col in table.column_names:
            table = table.drop([col])

    is_subtask4 = group_position == GROUP_SIZE - 1
    copy_n = 0 if is_subtask4 else COPY_FRAMES
    new_length = num_rows + copy_n

    # Append copy rows (next episode's first copy_n rows) for subtasks 1/2/3.
    if copy_n > 0:
        assert next_src_path is not None
        next_table = pq.read_table(next_src_path)
        if next_table.num_rows < copy_n:
            raise ValueError(
                f"episode {episode_id}: next episode {next_src_path.name} has only "
                f"{next_table.num_rows} rows, need {copy_n} copy frames"
            )
        next_slice = next_table.slice(0, copy_n)
        # Drop boundary columns from next_slice if present (shouldn't be, but be safe).
        for col in NEW_COLUMNS:
            if col in next_slice.column_names:
                next_slice = next_slice.drop([col])
        table = pa.concat_tables([table, next_slice], promote_options="default")

    # --- Rebuild index columns for all rows ---
    replacements: dict[str, np.ndarray] = {
        "frame_index": np.arange(new_length, dtype=np.int64),
        "timestamp": (np.arange(new_length, dtype=np.float32) / fps).astype(np.float32),
        "episode_index": np.full(new_length, episode_id, dtype=np.int64),
        "index": np.arange(global_offset, global_offset + new_length, dtype=np.int64),
        "task_index": np.full(new_length, task_index, dtype=np.int64),
    }
    for col_name, data in replacements.items():
        if col_name in table.column_names:
            field_idx = table.column_names.index(col_name)
            col_type = table.schema.field(col_name).type
            table = table.set_column(field_idx, col_name, pa.array(data, type=col_type))
        else:
            table = table.append_column(col_name, pa.array(data))

    # --- Build boundary label columns ---
    completion = np.zeros(new_length, dtype=np.float32)
    completion[-TOTAL_POSITIVES:] = 1.0  # last 10 = 1 for every episode

    source_episode = np.full(new_length, episode_id, dtype=np.int64)
    source_frame = np.arange(new_length, dtype=np.int64)
    is_copy = np.zeros(new_length, dtype=np.int8)

    if copy_n > 0:
        next_episode_id = episode_id + 1
        source_episode[num_rows:] = next_episode_id
        source_frame[num_rows:] = np.arange(copy_n, dtype=np.int64)
        is_copy[num_rows:] = 1

    new_cols = {
        "completion": pa.array(completion, type=pa.float32()),
        "source_episode_index": pa.array(source_episode, type=pa.int64()),
        "source_frame_index": pa.array(source_frame, type=pa.int64()),
        "is_boundary_copy": pa.array(is_copy, type=pa.int8()),
    }
    for col_name, col_data in new_cols.items():
        if col_name in table.column_names:
            field_idx = table.column_names.index(col_name)
            table = table.set_column(field_idx, col_name, col_data)
        else:
            table = table.append_column(col_name, col_data)

    table = update_boundary_hf_schema_metadata(table)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, dst_path)
    return {"skipped": 0, "written": 1}


def _process_parquet_worker(args: tuple) -> dict:
    (
        src_path,
        dst_path,
        episode_id,
        group_position,
        task_index,
        next_src_path,
        global_offset,
        old_length,
        fps,
        force,
    ) = args
    try:
        return process_boundary_parquet(
            pathlib.Path(src_path),
            pathlib.Path(dst_path),
            episode_id=episode_id,
            group_position=group_position,
            task_index=task_index,
            next_src_path=pathlib.Path(next_src_path) if next_src_path else None,
            global_offset=global_offset,
            old_length=old_length,
            fps=fps,
            force=force,
        )
    except Exception as exc:
        return {"error": str(exc), "src": str(src_path), "episode": episode_id}


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------


def extend_episode_video(
    current_video: pathlib.Path,
    next_video: pathlib.Path,
    dst_video: pathlib.Path,
    copy_frames: int,
) -> None:
    """Extends ``current_video`` with the first ``copy_frames`` frames of ``next_video``.

    Writes to a ``.tmp`` path first and verifies the frame count before
    atomically renaming, so a failed ffmpeg never leaves a truncated file at
    the final path (P1-4).

    Uses lossless encoding (CRF 0) with all I-frames (``-g 1``) so that the
    same source frame produces byte-identical output regardless of its
    position in the concatenated stream.  This guarantees pixel-level
    consistency between the positive (copy frame) and negative (original
    first-5) contrast pair — both are decoded from the same source video and
    re-encoded with the same lossless settings (P1-D).
    """

    filter_complex = f"[1:v]trim=end_frame={copy_frames},setpts=PTS-STARTPTS[n5];[0:v][n5]concat=n=2:v=1:a=0[outv]"
    tmp_video = dst_video.with_suffix(".mp4.tmp")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(current_video),
        "-i",
        str(next_video),
        "-filter_complex",
        filter_complex,
        "-map",
        "[outv]",
        "-f",
        "mp4",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "0",
        "-g",
        "1",
        "-an",
        "-pix_fmt",
        "yuv420p",
        str(tmp_video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        # Clean up partial output.
        if tmp_video.exists():
            tmp_video.unlink()
        raise RuntimeError(f"ffmpeg failed for {current_video.name} + {next_video.name}: {result.stderr[-500:]}")
    os.replace(tmp_video, dst_video)


def reencode_episode_video(
    src_video: pathlib.Path,
    dst_video: pathlib.Path,
) -> None:
    """Re-encodes a video with the same lossless settings used by
    :func:`extend_episode_video`.

    This is used for subtask-4 episodes (no copy frames) so that all videos
    in the boundary dataset go through the same encoding pipeline, ensuring
    pixel-level consistency between copy frames and their original
    counterparts (P1-D).
    """

    tmp_video = dst_video.with_suffix(".mp4.tmp")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src_video),
        "-f",
        "mp4",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "0",
        "-g",
        "1",
        "-an",
        "-pix_fmt",
        "yuv420p",
        str(tmp_video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if tmp_video.exists():
            tmp_video.unlink()
        raise RuntimeError(f"ffmpeg failed for {src_video.name}: {result.stderr[-500:]}")
    os.replace(tmp_video, dst_video)


def process_episode_videos(
    src_root: pathlib.Path,
    dst_root: pathlib.Path,
    episode_id: int,
    next_episode_id: int | None,
    video_keys: list[str],
    chunks_size: int,
    copy_n: int,
    *,
    force: bool,
    expected_length: int,
) -> dict:
    """Copies or extends all camera videos for one episode.

    All videos are re-encoded with the same lossless settings (CRF 0, all
    I-frames) so that copy frames and their original counterparts are
    pixel-identical (P1-D).

    After writing each video, verifies the frame count matches
    ``expected_length`` using ffprobe (P1-4).
    """

    for key in video_keys:
        src_video = get_episode_video_path(src_root, episode_id, key, chunks_size)
        dst_video = get_episode_video_path(dst_root, episode_id, key, chunks_size)
        if not src_video.exists():
            raise FileNotFoundError(f"source video not found: {src_video}")

        if dst_video.exists() and not force:
            continue

        dst_video.parent.mkdir(parents=True, exist_ok=True)
        if copy_n > 0 and next_episode_id is not None:
            next_video = get_episode_video_path(src_root, next_episode_id, key, chunks_size)
            if not next_video.exists():
                raise FileNotFoundError(f"next video not found: {next_video}")
            extend_episode_video(src_video, next_video, dst_video, copy_n)
        else:
            reencode_episode_video(src_video, dst_video)

        # Verify frame count (P1-4).
        nb = count_video_frames(dst_video)
        if nb != expected_length:
            dst_video.unlink(missing_ok=True)
            raise RuntimeError(
                f"video frame count mismatch for episode {episode_id} key {key}: "
                f"video {nb} frames != expected {expected_length}"
            )
    return {"ok": 1}


def _process_video_worker(args: tuple) -> dict:
    (
        src_root,
        dst_root,
        episode_id,
        next_episode_id,
        video_keys,
        chunks_size,
        copy_n,
        force,
        expected_length,
    ) = args
    try:
        return process_episode_videos(
            pathlib.Path(src_root),
            pathlib.Path(dst_root),
            episode_id=episode_id,
            next_episode_id=next_episode_id,
            video_keys=video_keys,
            chunks_size=chunks_size,
            copy_n=copy_n,
            force=force,
            expected_length=expected_length,
        )
    except Exception as exc:
        return {"error": str(exc), "episode": episode_id}


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


# P1-B: LeRobot stats files must NOT be copied as-is — they describe the
# original frame distribution and become stale after boundary copies.
# Instead, compute_boundary_stats() recalculates them from the final parquet
# data after generation.
_STATS_FILES_TO_SKIP = {"stats.json", "episodes_stats.jsonl", "stats"}


def write_boundary_meta(
    src_meta: pathlib.Path,
    dst_meta: pathlib.Path,
    *,
    new_lengths: dict[int, int],
    total_frames: int,
    fps: float,
) -> None:
    """Copies ``meta/`` and patches ``info.json`` + ``episodes.jsonl``.

    Stats files (``stats.json``, ``episodes_stats.jsonl``, ``stats/``) are
    deliberately skipped here — they are recomputed from the final parquet
    data by :func:`compute_boundary_stats` after all parquet files are written.
    """

    dst_meta.mkdir(parents=True, exist_ok=True)
    for item in src_meta.iterdir():
        if item.name == "info.json":
            continue
        # P1-3: Skip stale stats files.
        if item.name in _STATS_FILES_TO_SKIP:
            continue
        dst_item = dst_meta / item.name
        if item.is_file():
            shutil.copy2(item, dst_item)
        else:
            shutil.copytree(item, dst_item, dirs_exist_ok=True)

    # --- Patch info.json ---
    info = json.loads((src_meta / "info.json").read_text(encoding="utf-8"))
    features = info.setdefault("features", {})
    for name, feat in BOUNDARY_FEATURES.items():
        features[name] = dict(feat)
    info["total_frames"] = total_frames
    info["label_scheme"] = BOUNDARY_LABEL_SCHEME
    info["boundary_copy_frames"] = COPY_FRAMES
    info["positive_tail"] = POSITIVE_TAIL
    info["subtask4_tail"] = SUBTASK4_TAIL
    info.pop("window_seconds", None)

    info_path = dst_meta / "info.json"
    tmp = info_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, info_path)

    # --- Patch episodes.jsonl with new lengths ---
    episodes_path = dst_meta / "episodes.jsonl"
    tmp = episodes_path.with_suffix(".jsonl.tmp")
    with open(src_meta / "episodes.jsonl", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            eid = int(record["episode_index"])
            record["length"] = new_lengths[eid]
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, episodes_path)


# ---------------------------------------------------------------------------
# Stats recomputation (P1-B)
# ---------------------------------------------------------------------------


def _get_feature_stats(array: np.ndarray, *, axis: int = 0, keepdims: bool = True) -> dict[str, np.ndarray]:
    """Computes min/max/mean/std/count for a single feature array."""
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([len(array)]),
    }


def _aggregate_feature_stats(stats_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Aggregates per-episode stats for a single feature into global stats."""
    means = np.stack([s["mean"] for s in stats_list])
    variances = np.stack([s["std"] ** 2 for s in stats_list])
    counts = np.stack([s["count"] for s in stats_list])
    total_count = counts.sum(axis=0)

    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    return {
        "min": np.min(np.stack([s["min"] for s in stats_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }


def _serialize_stats(stats: dict[str, dict[str, np.ndarray]]) -> dict:
    """Converts numpy arrays in stats to lists for JSON serialization."""
    result = {}
    for feat_name, feat_stats in stats.items():
        result[feat_name] = {k: v.tolist() for k, v in feat_stats.items()}
    return result


def compute_boundary_stats(
    dst_root: pathlib.Path,
    src_meta: pathlib.Path,
    new_lengths: dict[int, int],
    chunks_size: int,
    video_keys: list[str],
) -> None:
    """P1-B: Computes and writes LeRobot stats from the boundary parquet files.

    For non-video features, stats are computed from the parquet data.  For
    video features (pixel data not in parquet), the source dataset's per-episode
    stats are reused with the count updated to the new episode length.

    Writes ``meta/episodes_stats.jsonl`` (v2.1) and ``meta/stats.json`` (global
    aggregate).  Deletes any residual ``stats/`` directory.
    """

    dst_meta = dst_root / "meta"
    info = json.loads((dst_meta / "info.json").read_text(encoding="utf-8"))
    features = info.get("features", {})

    # Read source per-episode stats for video features.
    src_ep_stats: dict[int, dict] = {}
    src_global_stats: dict = {}
    src_ep_stats_path = src_meta / "episodes_stats.jsonl"
    src_stats_path = src_meta / "stats.json"
    if src_ep_stats_path.exists():
        for line in src_ep_stats_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            src_ep_stats[int(record["episode_index"])] = record["stats"]
    elif src_stats_path.exists():
        src_global_stats = json.loads(src_stats_path.read_text(encoding="utf-8"))

    video_feature_names = set()
    for key in video_keys:
        if key in features:
            video_feature_names.add(key)

    # Compute per-episode stats.
    all_episode_stats: list[dict[str, dict[str, np.ndarray]]] = []
    episode_ids_sorted = sorted(new_lengths.keys())

    for eid in episode_ids_sorted:
        parquet_path = get_episode_data_path(dst_root, eid, chunks_size)
        table = pq.read_table(parquet_path)
        new_len = new_lengths[eid]

        ep_stats: dict[str, dict[str, np.ndarray]] = {}
        for col_name in table.column_names:
            feat = features.get(col_name, {})
            dtype = feat.get("dtype", "")

            if dtype in ("image", "video") or col_name in video_feature_names:
                # Video/image: reuse source stats with updated count.
                src_stats_ep = src_ep_stats.get(eid, {})
                if col_name in src_stats_ep:
                    ft_stats = src_stats_ep[col_name]
                elif col_name in src_global_stats:
                    ft_stats = src_global_stats[col_name]
                else:
                    continue  # No source stats for this video feature — skip.
                ep_stats[col_name] = {
                    k: np.array(v) if k != "count" else np.array([new_len])
                    for k, v in ft_stats.items()
                }
            elif dtype == "string":
                continue
            else:
                # Non-video: compute from parquet.
                arr = table[col_name].to_numpy()
                if arr.dtype == object:
                    # List-type column (e.g. vector features): stack into 2D.
                    try:
                        arr = np.stack([np.asarray(row, dtype=np.float64) for row in arr])
                    except (ValueError, TypeError):
                        continue
                keepdims = arr.ndim == 1
                ep_stats[col_name] = _get_feature_stats(arr, axis=0, keepdims=keepdims)

        all_episode_stats.append(ep_stats)

    # Aggregate global stats.
    data_keys = {key for stats in all_episode_stats for key in stats}
    global_stats: dict[str, dict[str, np.ndarray]] = {}
    for key in data_keys:
        stats_with_key = [s[key] for s in all_episode_stats if key in s]
        global_stats[key] = _aggregate_feature_stats(stats_with_key)

    # Write episodes_stats.jsonl.
    episodes_stats_path = dst_meta / "episodes_stats.jsonl"
    if episodes_stats_path.exists():
        episodes_stats_path.unlink()
    with open(episodes_stats_path, "w", encoding="utf-8") as f:
        for i, eid in enumerate(episode_ids_sorted):
            record = {
                "episode_index": eid,
                "stats": _serialize_stats(all_episode_stats[i]),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write stats.json (global aggregate).
    stats_path = dst_meta / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(_serialize_stats(global_stats), f, indent=4, ensure_ascii=False)

    # Delete residual stats/ directory.
    stats_dir = dst_meta / "stats"
    if stats_dir.exists():
        shutil.rmtree(stats_dir)

    # Verify coverage: total count in global stats should match total frames.
    total_frames = sum(new_lengths.values())
    for feat_name, feat_stats in global_stats.items():
        count = int(feat_stats["count"].flat[0])
        if count != total_frames:
            raise RuntimeError(
                f"Stats coverage mismatch for feature '{feat_name}': "
                f"count={count} != total_frames={total_frames}"
            )


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_boundary_dataset(
    dst_root: pathlib.Path,
    *,
    new_lengths: dict[int, int],
    fps: float,
    chunks_size: int,
    video_keys: list[str],
    seed: int,
    episodes_per_group: int,
    val_groups: int,
    test_groups: int,
    repo_id: str,
    verify_frames: bool,
    stride: int = 15,
    forced_first_n: int = COPY_FRAMES,
) -> dict:
    """Audits the generated dataset and returns the ``label_audit`` dict.

    Raises ``SystemExit`` on any audit failure (non-0/1 labels, subtask positives
    != 10, count inconsistency, or cross-group copy), reporting the failing
    episode id and length.
    """

    episode_ids = sorted(new_lengths.keys())
    groups = build_task_groups(episode_ids, episodes_per_group=episodes_per_group, minimum_groups=1)
    group_of: dict[int, tuple[tuple[int, ...], int]] = {}
    for group in groups:
        for pos, eid in enumerate(group.episode_ids):
            group_of[eid] = (group.episode_ids, pos)

    # --- Per-episode audit ---
    audits: dict[int, object] = {}
    failures: list[dict] = []
    for eid in episode_ids:
        group_ep_ids, pos = group_of[eid]
        parquet_path = get_episode_data_path(dst_root, eid, chunks_size)
        try:
            audits[eid] = audit_boundary_completion_episode_parquet(
                parquet_path,
                episode_id=eid,
                expected_length=new_lengths[eid],
                group_episode_ids=group_ep_ids,
                group_position=pos,
            )
        except Exception as exc:
            failures.append({"episode_id": eid, "length": new_lengths[eid], "error": str(exc)})

    if failures:
        print("AUDIT FAILURES:")
        for f in failures[:20]:
            print(f"  episode {f['episode_id']} (length {f['length']}): {f['error']}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        raise SystemExit(1)

    # --- Split manifest ---
    manifest = create_split_manifest(
        episode_ids,
        repo_id=repo_id,
        seed=seed,
        episodes_per_group=episodes_per_group,
        val_groups=val_groups,
        test_groups=test_groups,
    )
    train_ids = manifest.episode_ids("train")
    test_ids = manifest.episode_ids("test")
    val_ids = manifest.episode_ids("val")

    # --- Full-frame counts ---
    def full_counts(ids):
        pos = sum(audits[e].positive_count for e in ids)
        neg = sum(audits[e].negative_count for e in ids)
        return pos, neg

    train_pos, train_neg = full_counts(train_ids)
    test_pos, test_neg = full_counts(test_ids)
    val_pos, val_neg = full_counts(val_ids)

    # --- Sampled counts (same rule as training) ---
    def sampled_counts(ids):
        pos = 0
        neg = 0
        for e in ids:
            audit = audits[e]
            positives, ordinary, forced = boundary_train_sample_indices(
                audit, stride=stride, forced_first_n=forced_first_n
            )
            sample_set = np.unique(np.concatenate([positives, ordinary, forced]))
            ep_pos = len(positives)
            ep_neg = len(sample_set) - ep_pos
            pos += ep_pos
            neg += ep_neg
        return pos, neg

    train_samp_pos, train_samp_neg = sampled_counts(train_ids)
    test_samp_pos, test_samp_neg = sampled_counts(test_ids)

    # --- Per-subtask positive counts ---
    per_subtask_pos: dict[str, int] = {}
    for eid in episode_ids:
        pos_label = str(audits[eid].group_position)
        per_subtask_pos.setdefault(pos_label, 0)
        per_subtask_pos[pos_label] += audits[eid].positive_count

    # --- Boundary copy + first-5 negatives ---
    boundary_copy_pos = sum(audits[e].boundary_copy_count for e in episode_ids)
    first5_neg = 0
    for eid in episode_ids:
        _, _, forced = boundary_train_sample_indices(audits[eid], stride=stride, forced_first_n=forced_first_n)
        first5_neg += len(forced)

    # --- Per-episode detail ---
    per_episode = [
        {
            "episode_id": eid,
            "group_position": audits[eid].group_position,
            "length": new_lengths[eid],
            "positive": audits[eid].positive_count,
            "negative": audits[eid].negative_count,
            "boundary_copy_count": audits[eid].boundary_copy_count,
        }
        for eid in episode_ids
    ]

    # --- Consistency checks ---
    total_frames = sum(new_lengths.values())
    consistency = {
        "total_frames_equals_pos_plus_neg": (train_pos + train_neg + test_pos + test_neg + val_pos + val_neg)
        == total_frames,
        "per_episode_positive_all_10": all(audits[eid].positive_count == TOTAL_POSITIVES for eid in episode_ids),
        "no_val_episodes": len(val_ids) == 0,
        "parquet_video_match": True,
        "index_contiguous": True,
        "timestamp_correct": True,
        "metadata_consistent": True,
    }

    # Verify info.json total_frames.
    info = read_info(dst_root / "meta")
    if info.get("total_frames") != total_frames:
        consistency["metadata_consistent"] = False

    # Verify global index contiguity + timestamp.
    expected_index = 0
    for eid in episode_ids:
        table = pq.read_table(
            get_episode_data_path(dst_root, eid, chunks_size),
            columns=["index", "timestamp", "frame_index"],
        )
        idx = table["index"].to_numpy()
        ts = table["timestamp"].to_numpy()
        fi = table["frame_index"].to_numpy()
        if not np.array_equal(idx, np.arange(expected_index, expected_index + len(idx))):
            consistency["index_contiguous"] = False
        if not np.allclose(ts, np.arange(len(ts)) / fps, atol=1e-5):
            consistency["timestamp_correct"] = False
        if not np.array_equal(fi, np.arange(len(fi))):
            consistency["timestamp_correct"] = False
        expected_index += len(idx)

    # Verify parquet row count == video frame count.
    if verify_frames:
        for eid in tqdm.tqdm(episode_ids, desc="Verifying video frames", file=sys.stderr):
            for key in video_keys:
                vpath = get_episode_video_path(dst_root, eid, key, chunks_size)
                if not vpath.exists():
                    consistency["parquet_video_match"] = False
                    break
                nb = count_video_frames(vpath)
                if nb != new_lengths[eid]:
                    consistency["parquet_video_match"] = False
                    print(f"  MISMATCH episode {eid} key {key}: video {nb} frames != parquet {new_lengths[eid]} rows")
            if not consistency["parquet_video_match"]:
                break

    all_ok = all(consistency.values())
    consistency["all_checks_passed"] = all_ok

    audit_dict = {
        "label_scheme": BOUNDARY_LABEL_SCHEME,
        "total_episodes": len(episode_ids),
        "total_frames": total_frames,
        "train_episode_count": len(train_ids),
        "test_episode_count": len(test_ids),
        "val_episode_count": len(val_ids),
        "train_full": format_ratio(train_pos, train_neg),
        "train_sampled": format_ratio(train_samp_pos, train_samp_neg),
        "test_full": format_ratio(test_pos, test_neg),
        "test_sparse": format_ratio(test_samp_pos, test_samp_neg),
        "val_full": format_ratio(val_pos, val_neg),
        "per_subtask_positive_counts": per_subtask_pos,
        "boundary_copy_positive_count": boundary_copy_pos,
        "original_first5_negative_count": first5_neg,
        "per_episode": per_episode,
        "consistency": consistency,
    }

    if not all_ok:
        print("CONSISTENCY CHECK FAILURES:")
        for k, v in consistency.items():
            if k != "all_checks_passed" and not v:
                print(f"  {k}: FAIL")
        raise SystemExit(1)

    return audit_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a boundary-labeled dataset copy with strict 0/1 completion labels."
    )
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--src-root", type=pathlib.Path, default=pathlib.Path(DEFAULT_SRC_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument(
        "--skip-frame-verification",
        action="store_true",
        help="Skip ffprobe frame-count verification (faster, less thorough).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes-per-group", type=int, default=GROUP_SIZE)
    parser.add_argument("--val-groups", type=int, default=0)
    parser.add_argument("--test-groups", type=int, default=10)
    parser.add_argument("--stride", type=int, default=15)
    args = parser.parse_args()

    src_root: pathlib.Path = args.src_root.resolve()
    dst_root: pathlib.Path = args.output_root.resolve()

    if src_root == dst_root:
        parser.error("--output-root must not be the same as --src-root (original dataset must not be modified).")
    if not (src_root / "meta" / "info.json").is_file():
        parser.error(f"source dataset not found: {src_root / 'meta' / 'info.json'}")

    # P1-C: Generate in a staging directory, then atomically publish.
    staging_root = dst_root.parent / (dst_root.name + ".staging")
    if staging_root.exists():
        print(f"Cleaning up leftover staging directory: {staging_root}")
        shutil.rmtree(staging_root)
    if dst_root.exists() and not args.force:
        parser.error(f"Output directory already exists: {dst_root}. Use --force to overwrite.")

    print(f"Source dataset : {src_root}")
    print(f"Output dataset : {dst_root}")
    print(f"Staging        : {staging_root}")

    # --- Load source metadata ---
    info = read_info(src_root / "meta")
    fps = float(info["fps"])
    chunks_size = int(info.get("chunks_size", 1000))
    video_keys = get_video_keys(info)
    episode_lengths = load_episode_lengths(src_root / "meta")
    episode_ids = sorted(episode_lengths.keys())
    print(f"Episodes       : {len(episode_ids)}")
    print(f"FPS            : {fps}")
    print(f"Video keys     : {video_keys}")

    if len(episode_ids) % GROUP_SIZE != 0:
        parser.error(f"episode count {len(episode_ids)} is not divisible by group size {GROUP_SIZE}")

    # --- Compute new lengths + global offsets ---
    new_lengths: dict[int, int] = {}
    for eid in episode_ids:
        pos = eid % GROUP_SIZE
        old = episode_lengths[eid]
        if pos < GROUP_SIZE - 1:
            if old < POSITIVE_TAIL:
                parser.error(
                    f"episode {eid} (subtask {pos + 1}) has only {old} frames; "
                    f"need >= {POSITIVE_TAIL} for the boundary tail"
                )
            new_lengths[eid] = old + COPY_FRAMES
        else:
            if old < SUBTASK4_TAIL:
                parser.error(
                    f"episode {eid} (subtask 4) has only {old} frames; need >= {SUBTASK4_TAIL} for the boundary tail"
                )
            new_lengths[eid] = old
    total_frames = sum(new_lengths.values())

    global_offsets: dict[int, int] = {}
    offset = 0
    for eid in episode_ids:
        global_offsets[eid] = offset
        offset += new_lengths[eid]

    print(f"New total frames: {total_frames} (was {info.get('total_frames')})")

    # --- Build parquet jobs ---
    parquet_jobs: list[tuple] = []
    for eid in episode_ids:
        pos = eid % GROUP_SIZE
        task_idx = pos
        src_path = get_episode_data_path(src_root, eid, chunks_size)
        dst_path = get_episode_data_path(staging_root, eid, chunks_size)
        next_src_path: pathlib.Path | None = None
        if pos < GROUP_SIZE - 1:
            next_src_path = get_episode_data_path(src_root, eid + 1, chunks_size)
        parquet_jobs.append(
            (
                str(src_path),
                str(dst_path),
                eid,
                pos,
                task_idx,
                str(next_src_path) if next_src_path else None,
                global_offsets[eid],
                episode_lengths[eid],
                fps,
                args.force,
            )
        )

    # --- Build video jobs ---
    video_jobs: list[tuple] = []
    if not args.skip_videos:
        for eid in episode_ids:
            pos = eid % GROUP_SIZE
            copy_n = 0 if pos == GROUP_SIZE - 1 else COPY_FRAMES
            next_eid = eid + 1 if pos < GROUP_SIZE - 1 else None
            video_jobs.append(
                (
                    str(src_root),
                    str(staging_root),
                    eid,
                    next_eid,
                    video_keys,
                    chunks_size,
                    copy_n,
                    args.force,
                    new_lengths[eid],
                )
            )

    if args.dry_run:
        print("\n[DRY RUN] No files will be written.")
        print(f"  Would process {len(parquet_jobs)} parquet files (add boundary columns + copy frames)")
        print(
            f"  Would process {len(video_jobs)} video episodes ({sum(1 for j in video_jobs if j[6] > 0)} extended, {sum(1 for j in video_jobs if j[6] == 0)} re-encoded)"
        )
        print("  Would write meta/ (patch info.json + episodes.jsonl)")
        print("  Would compute stats from parquet (episodes_stats.jsonl + stats.json)")
        print("  Would audit + write label_audit.json")
        print(f"  Would atomically publish: {staging_root} -> {dst_root}")
        # Quick validation on a few files.
        for src_path, _, eid, *_ in parquet_jobs[:3]:
            table = pq.read_table(src_path, columns=["frame_index"])
            print(f"    episode_{eid:06d}: {table.num_rows} rows")
        return

    try:
        _generate_in_staging(
            staging_root=staging_root,
            src_root=src_root,
            dst_root=dst_root,
            parquet_jobs=parquet_jobs,
            video_jobs=video_jobs,
            new_lengths=new_lengths,
            total_frames=total_frames,
            fps=fps,
            chunks_size=chunks_size,
            video_keys=video_keys,
            args=args,
        )
    except Exception:
        # P1-C: Clean up staging directory on any failure.
        print(f"\nGeneration failed — cleaning up staging directory: {staging_root}")
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise


def _generate_in_staging(
    *,
    staging_root: pathlib.Path,
    src_root: pathlib.Path,
    dst_root: pathlib.Path,
    parquet_jobs: list[tuple],
    video_jobs: list[tuple],
    new_lengths: dict[int, int],
    total_frames: int,
    fps: float,
    chunks_size: int,
    video_keys: list[str],
    args: argparse.Namespace,
) -> None:
    """Generates the entire dataset in ``staging_root``, audits, then publishes."""

    # --- 1. Process parquet files ---
    print(f"\n[1/5] Processing {len(parquet_jobs)} parquet files with {args.workers} worker(s) ...")
    total_written = 0
    total_skipped = 0
    errors: list[dict] = []
    if args.workers <= 1:
        for item in tqdm.tqdm(parquet_jobs, desc="Labeling parquet"):
            result = _process_parquet_worker(item)
            total_written += result.get("written", 0)
            total_skipped += result.get("skipped", 0)
            if "error" in result:
                errors.append(result)
    else:
        with mp.Pool(args.workers) as pool:
            for result in tqdm.tqdm(
                pool.imap_unordered(_process_parquet_worker, parquet_jobs),
                total=len(parquet_jobs),
                desc="Labeling parquet",
            ):
                total_written += result.get("written", 0)
                total_skipped += result.get("skipped", 0)
                if "error" in result:
                    errors.append(result)
    print(f"  Written: {total_written}  Skipped: {total_skipped}")
    if errors:
        print(f"  Errors: {len(errors)}")
        for err in errors[:10]:
            print(f"    episode {err.get('episode', '?')}: {err.get('error', '?')}")
        raise RuntimeError(f"{len(errors)} parquet processing errors")

    # --- 2. Process videos ---
    if args.skip_videos:
        print("\n[2/5] Skipping videos (--skip-videos)")
    else:
        print(f"\n[2/5] Processing {len(video_jobs)} video episodes with {args.workers} worker(s) ...")
        vid_errors: list[dict] = []
        if args.workers <= 1:
            for item in tqdm.tqdm(video_jobs, desc="Processing videos"):
                result = _process_video_worker(item)
                if "error" in result:
                    vid_errors.append(result)
        else:
            with mp.Pool(args.workers) as pool:
                vid_errors.extend(
                    result
                    for result in tqdm.tqdm(
                        pool.imap_unordered(_process_video_worker, video_jobs),
                        total=len(video_jobs),
                        desc="Processing videos",
                    )
                    if "error" in result
                )
        if vid_errors:
            print(f"  Video errors: {len(vid_errors)}")
            for err in vid_errors[:10]:
                print(f"    episode {err.get('episode', '?')}: {err.get('error', '?')}")
            raise RuntimeError(f"{len(vid_errors)} video processing errors")
        print("  Done.")

    # --- 3. Write meta/ ---
    print("\n[3/5] Writing meta/ ...")
    write_boundary_meta(
        src_root / "meta",
        staging_root / "meta",
        new_lengths=new_lengths,
        total_frames=total_frames,
        fps=fps,
    )
    print(f"  Patched info.json (label_scheme=boundary, total_frames={total_frames})")

    # --- 4. Compute stats (P1-B) ---
    print("\n[4/5] Computing stats from parquet data ...")
    compute_boundary_stats(
        staging_root,
        src_root / "meta",
        new_lengths=new_lengths,
        chunks_size=chunks_size,
        video_keys=video_keys,
    )
    print("  Written episodes_stats.jsonl + stats.json")

    # --- 5. Audit + label_audit.json ---
    print("\n[5/5] Auditing dataset ...")
    audit = audit_boundary_dataset(
        staging_root,
        new_lengths=new_lengths,
        fps=fps,
        chunks_size=chunks_size,
        video_keys=video_keys,
        seed=args.seed,
        episodes_per_group=args.episodes_per_group,
        val_groups=args.val_groups,
        test_groups=args.test_groups,
        repo_id=dst_root.name,
        verify_frames=not args.skip_frame_verification,
        stride=args.stride,
    )

    audit_path = staging_root / "label_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n  label_audit.json written to: {audit_path}")
    print(
        f"  Train: {audit['train_episode_count']} episodes, full {audit['train_full']['ratio']} "
        f"({audit['train_full']['positive_pct']}%), sampled {audit['train_sampled']['ratio']}"
    )
    print(
        f"  Test : {audit['test_episode_count']} episodes, full {audit['test_full']['ratio']} "
        f"({audit['test_full']['positive_pct']}%), sparse {audit['test_sparse']['ratio']}"
    )
    print(f"  Per-subtask positives: {audit['per_subtask_positive_counts']}")
    print(f"  Boundary copy positives: {audit['boundary_copy_positive_count']}")
    print(f"  Consistency: {'PASS' if audit['consistency']['all_checks_passed'] else 'FAIL'}")

    # --- Publish: atomically rename staging → final (P1-C) ---
    if dst_root.exists():
        print(f"\nRemoving old dataset: {dst_root}")
        shutil.rmtree(dst_root)
    os.rename(staging_root, dst_root)
    print(f"\nDone. Boundary dataset at: {dst_root}")
    print(f"  To use it for S2 training, set lerobot_home to: {dst_root.parent}")
    print(f"  and repo_id to: {dst_root.name}")


if __name__ == "__main__":
    main()
