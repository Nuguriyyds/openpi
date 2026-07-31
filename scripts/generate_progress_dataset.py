"""Build an idle-trimmed LeRobot dataset with FurnitureVLA progress labels.

The annotation files use source episode indices in their filenames. This script
only touches the annotated source episodes; it does not scan or rewrite the full
20k-episode source dataset.

By default the script is read-only. Pass ``--write`` to build a new dataset.
The source dataset is never modified.

Examples:

    # Fast annotation-only validation (does not need the source dataset).
    uv run scripts/generate_progress_dataset.py \
        --annotation-dir /path/to/split \
        --annotations-only

    # Validate annotations against source Parquet files and video existence.
    uv run scripts/generate_progress_dataset.py \
        --annotation-dir /path/to/split

    # Build an independent new dataset. Parquet rows and videos are cropped to the four stages.
    uv run scripts/generate_progress_dataset.py \
        --annotation-dir /path/to/split \
        --write
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import copy
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from fractions import Fraction
from itertools import pairwise
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np

DEFAULT_SOURCE_ROOT = Path(
    "/mnt/data/dataset/ei/huggingface/modanqing/agilex_pick_and_place_one_object_d435_d405_20000"
)
DEFAULT_ANNOTATION_DIR = Path("/mnt/data/dataset/ei/huggingface/modanqing/split/split")
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/dataset/ei/huggingface/modanqing/split/progress/full_trajectories_progress")
DEFAULT_LOG_ROOT = Path("/mnt/data/dataset/ei/huggingface/modanqing/split/progress/logs")

LOGGER = logging.getLogger("progress_dataset")

EXPECTED_ANNOTATION_COUNT = 3399
EXPECTED_VALID_EPISODE_COUNT = 3337
EXPECTED_FPS = 30
EXPECTED_ACTION_DIM = 14

STAGES = ("extend", "pick", "place", "return")
STAGE_PROGRESS = {
    "extend": (0.0, 0.25),
    "pick": (0.25, 0.5),
    "place": (0.5, 0.75),
    "return": (0.75, 1.0),
}
TASK = "Pick up one object from the near box, place it into the far box, and return to the starting position."
EPISODE_FILE_RE = re.compile(r"episode_(\d{6})\.json$")


@dataclass(frozen=True)
class FrameInterval:
    name: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class Annotation:
    source_episode_index: int
    annotation_file: str
    stages: tuple[FrameInterval, ...]
    idle: tuple[FrameInterval, ...]
    annotation_end_frame: int


@dataclass(frozen=True)
class ExcludedAnnotation:
    annotation_file: str
    source_episode_index: int | None
    reason: str
    detail: str


@dataclass(frozen=True)
class WarningRecord:
    annotation_file: str
    source_episode_index: int | None
    code: str
    detail: str


@dataclass(frozen=True)
class EpisodePlan:
    output_episode_index: int
    source_episode_index: int
    annotation_file: str
    length: int
    source_length: int
    source_start_frame: int
    source_end_frame: int
    final_endpoint_clamped: bool
    global_index_start: int
    source_parquet: Path
    source_videos: tuple[tuple[str, Path], ...]
    progress: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value must be an object")
    return value


def _read_jsonl_subset(path: Path, wanted_indices: set[int]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            episode_index = item.get("episode_index")
            if episode_index in wanted_indices:
                if episode_index in result:
                    raise ValueError(f"Duplicate episode_index {episode_index} in {path}")
                result[episode_index] = item
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def _setup_run_logging(log_root: Path) -> Path:
    run_name = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = log_root.resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)
    return run_dir


def _persist_summary(summary: dict[str, Any], run_dir: Path, report_path: Path | None) -> None:
    _write_json(run_dir / "summary.json", summary)
    if report_path is not None:
        _write_json(report_path, summary)


def _parse_int_list(value: Any, expected_length: int, field: str) -> list[int]:
    if not isinstance(value, list) or len(value) != expected_length:
        actual = len(value) if isinstance(value, list) else type(value).__name__
        raise ValueError(f"{field} must have length {expected_length}, got {actual}")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{field} must contain only integers")
    return value


def _parse_float_list(value: Any, expected_length: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != expected_length:
        actual = len(value) if isinstance(value, list) else type(value).__name__
        raise ValueError(f"{field} must have length {expected_length}, got {actual}")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item):
            raise ValueError(f"{field} must contain only finite numbers")
        result.append(float(item))
    return result


def _parse_annotation(path: Path) -> tuple[Annotation | None, ExcludedAnnotation | None, list[WarningRecord]]:
    warnings: list[WarningRecord] = []
    match = EPISODE_FILE_RE.fullmatch(path.name)
    if match is None:
        return None, ExcludedAnnotation(path.name, None, "filename", "Expected episode_XXXXXX.json"), warnings
    filename_index = int(match.group(1))

    try:
        data = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return None, ExcludedAnnotation(path.name, filename_index, "json_parse", str(error)), warnings

    json_index = data.get("index")
    if json_index != filename_index:
        return (
            None,
            ExcludedAnnotation(
                path.name,
                filename_index,
                "identity_mismatch",
                f"filename index={filename_index}, JSON index={json_index}",
            ),
            warnings,
        )
    expected_video_name = f"episode_{filename_index:06d}.mp4"
    if data.get("name") != expected_video_name:
        return (
            None,
            ExcludedAnnotation(
                path.name,
                filename_index,
                "identity_mismatch",
                f"Expected name={expected_video_name}, got {data.get('name')!r}",
            ),
            warnings,
        )

    sub_task = data.get("sub_task")
    if not isinstance(sub_task, dict):
        return None, ExcludedAnnotation(path.name, filename_index, "missing_sub_task", "sub_task is not an object"), warnings
    stage_names = sub_task.get("subtask")
    if stage_names != list(STAGES):
        return (
            None,
            ExcludedAnnotation(
                path.name,
                filename_index,
                "stage_sequence",
                f"Expected {list(STAGES)}, got {stage_names!r}",
            ),
            warnings,
        )

    try:
        stage_starts = _parse_int_list(sub_task.get("start_frame"), len(STAGES), "sub_task.start_frame")
        stage_ends = _parse_int_list(sub_task.get("end_frame"), len(STAGES), "sub_task.end_frame")
        stage_start_times = _parse_float_list(sub_task.get("start_time"), len(STAGES), "sub_task.start_time")
        stage_end_times = _parse_float_list(sub_task.get("end_time"), len(STAGES), "sub_task.end_time")
    except ValueError as error:
        return None, ExcludedAnnotation(path.name, filename_index, "invalid_stage_fields", str(error)), warnings

    stages = tuple(
        FrameInterval(name=name, start=start, end=end)
        for name, start, end in zip(STAGES, stage_starts, stage_ends, strict=True)
    )
    for index, interval in enumerate(stages):
        if interval.start < 0 or interval.end < interval.start:
            return (
                None,
                ExcludedAnnotation(
                    path.name,
                    filename_index,
                    "invalid_stage_interval",
                    f"{interval.name}=[{interval.start},{interval.end}]",
                ),
                warnings,
            )
        if stage_end_times[index] <= stage_start_times[index]:
            return (
                None,
                ExcludedAnnotation(
                    path.name,
                    filename_index,
                    "invalid_stage_time",
                    f"{interval.name}=[{stage_start_times[index]},{stage_end_times[index]})",
                ),
                warnings,
            )
        for timestamp, frame in (
            (stage_start_times[index], interval.start),
            (stage_end_times[index], interval.end),
        ):
            if abs(timestamp * EXPECTED_FPS - frame) > 1.01:
                return (
                    None,
                    ExcludedAnnotation(
                        path.name,
                        filename_index,
                        "frame_time_mismatch",
                        f"{interval.name}: time={timestamp}, frame={frame}",
                    ),
                    warnings,
                )
        if index < len(stages) - 1:
            if interval.end != stages[index + 1].start:
                return (
                    None,
                    ExcludedAnnotation(
                        path.name,
                        filename_index,
                        "stage_frame_gap_or_overlap",
                        f"{interval.name} ends at {interval.end}, next starts at {stages[index + 1].start}",
                    ),
                    warnings,
                )
            if not math.isclose(stage_end_times[index], stage_start_times[index + 1], abs_tol=1e-9):
                return (
                    None,
                    ExcludedAnnotation(
                        path.name,
                        filename_index,
                        "stage_time_gap_or_overlap",
                        (
                            f"{interval.name} ends at {stage_end_times[index]}, "
                            f"next starts at {stage_start_times[index + 1]}"
                        ),
                    ),
                    warnings,
                )

    idle_data = data.get("idle_frames")
    if not isinstance(idle_data, dict):
        return None, ExcludedAnnotation(path.name, filename_index, "missing_idle", "idle_frames is not an object"), warnings
    idle_lengths = [
        len(idle_data.get(key)) if isinstance(idle_data.get(key), list) else -1
        for key in ("start_time", "end_time", "start_frame", "end_frame")
    ]
    if len(set(idle_lengths)) != 1 or idle_lengths[0] < 0:
        return (
            None,
            ExcludedAnnotation(path.name, filename_index, "idle_length_mismatch", f"lengths={idle_lengths}"),
            warnings,
        )
    idle_count = idle_lengths[0]
    try:
        idle_starts = _parse_int_list(idle_data.get("start_frame"), idle_count, "idle_frames.start_frame")
        idle_ends = _parse_int_list(idle_data.get("end_frame"), idle_count, "idle_frames.end_frame")
        idle_start_times = _parse_float_list(idle_data.get("start_time"), idle_count, "idle_frames.start_time")
        idle_end_times = _parse_float_list(idle_data.get("end_time"), idle_count, "idle_frames.end_time")
    except ValueError as error:
        return None, ExcludedAnnotation(path.name, filename_index, "invalid_idle_fields", str(error)), warnings

    idle: list[FrameInterval] = []
    initial_count = 0
    trailing_count = 0
    for index, (start, end) in enumerate(zip(idle_starts, idle_ends, strict=True)):
        if start < 0 or end < start:
            return (
                None,
                ExcludedAnnotation(
                    path.name,
                    filename_index,
                    "invalid_idle_interval",
                    f"idle[{index}]=[{start},{end}]",
                ),
                warnings,
            )

        if idle_end_times[index] <= idle_start_times[index]:
            return (
                None,
                ExcludedAnnotation(
                    path.name,
                    filename_index,
                    "invalid_idle_time",
                    f"idle[{index}]=[{idle_start_times[index]},{idle_end_times[index]})",
                ),
                warnings,
            )
        for timestamp, frame in ((idle_start_times[index], start), (idle_end_times[index], end)):
            if abs(timestamp * EXPECTED_FPS - frame) > 1.01:
                return (
                    None,
                    ExcludedAnnotation(
                        path.name,
                        filename_index,
                        "frame_time_mismatch",
                        f"idle[{index}]: time={timestamp}, frame={frame}",
                    ),
                    warnings,
                )
        if start == 0 and end == stages[0].start:
            name = "idle_start"
            initial_count += 1
        elif start == stages[-1].end:
            name = "idle_end"
            trailing_count += 1
        else:
            return (
                None,
                ExcludedAnnotation(
                    path.name,
                    filename_index,
                    "unexpected_idle_position",
                    f"idle[{index}]=[{start},{end}]",
                ),
                warnings,
            )
        idle.append(FrameInterval(name=name, start=start, end=end))
    if initial_count > 1 or trailing_count > 1:
        return (
            None,
            ExcludedAnnotation(
                path.name,
                filename_index,
                "duplicate_idle_region",
                f"initial={initial_count}, trailing={trailing_count}",
            ),
            warnings,
        )

    combined = sorted((*stages, *idle), key=lambda interval: (interval.start, interval.end))
    if combined[0].start != 0:
        return (
            None,
            ExcludedAnnotation(
                path.name,
                filename_index,
                "annotation_not_start_at_zero",
                f"first interval starts at {combined[0].start}",
            ),
            warnings,
        )
    for previous, current in pairwise(combined):
        if previous.end != current.start:
            relation = "gap" if previous.end < current.start else "overlap"
            return (
                None,
                ExcludedAnnotation(
                    path.name,
                    filename_index,
                    f"annotation_{relation}",
                    f"{previous.name} ends at {previous.end}, {current.name} starts at {current.start}",
                ),
                warnings,
            )

    annotation = Annotation(
        source_episode_index=filename_index,
        annotation_file=path.name,
        stages=stages,
        idle=tuple(idle),
        annotation_end_frame=combined[-1].end,
    )
    return annotation, None, warnings


def load_annotations(annotation_dir: Path) -> tuple[list[Annotation], list[ExcludedAnnotation], list[WarningRecord]]:
    if not annotation_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory does not exist: {annotation_dir}")
    paths = sorted(annotation_dir.glob("episode_*.json"))
    annotations: list[Annotation] = []
    excluded: list[ExcludedAnnotation] = []
    warnings: list[WarningRecord] = []
    for path in paths:
        annotation, exclusion, annotation_warnings = _parse_annotation(path)
        warnings.extend(annotation_warnings)
        if exclusion is not None:
            excluded.append(exclusion)
        elif annotation is not None:
            annotations.append(annotation)
    annotations.sort(key=lambda item: item.source_episode_index)
    return annotations, excluded, warnings


def _annotation_audit_rows(
    annotations: list[Annotation],
    excluded: list[ExcludedAnnotation],
    warnings: list[WarningRecord],
) -> list[dict[str, Any]]:
    warnings_by_file: dict[str, list[dict[str, Any]]] = {}
    for warning in warnings:
        warnings_by_file.setdefault(warning.annotation_file, []).append(asdict(warning))
    rows = [
        {
            "annotation_file": annotation.annotation_file,
            "source_episode_index": annotation.source_episode_index,
            "status": "valid",
            "annotation_end_frame": annotation.annotation_end_frame,
            "expected_closed_source_rows": annotation.annotation_end_frame + 1,
            "annotated_training_start_frame": annotation.stages[0].start,
            "annotated_training_end_frame": annotation.stages[-1].end,
            "training_interval": "closed [extend.start_frame, return.end_frame]",
            "stages": [asdict(interval) for interval in annotation.stages],
            "idle": [asdict(interval) for interval in annotation.idle],
            "warnings": warnings_by_file.get(annotation.annotation_file, []),
        }
        for annotation in annotations
    ]
    rows.extend(
        {
            "annotation_file": item.annotation_file,
            "source_episode_index": item.source_episode_index,
            "status": "excluded",
            "reason": item.reason,
            "detail": item.detail,
            "warnings": warnings_by_file.get(item.annotation_file, []),
        }
        for item in excluded
    )
    return sorted(rows, key=lambda row: (row["source_episode_index"] is None, row["source_episode_index"] or -1))


def make_progress(
    annotation: Annotation,
    source_length: int,
) -> tuple[np.ndarray, int, int, bool]:
    expected_source_length = annotation.annotation_end_frame + 1
    source_row_delta = source_length - expected_source_length
    if source_row_delta not in (0, -1):
        raise ValueError(
            f"closed annotation ends at frame index {annotation.annotation_end_frame}, so expected "
            f"{expected_source_length} source rows; only an exact match or one missing final row is supported, "
            f"got {source_length}"
        )

    crop_start = annotation.stages[0].start
    annotated_crop_end = annotation.stages[-1].end
    final_endpoint_clamped = annotated_crop_end == source_length
    crop_end = min(annotated_crop_end, source_length - 1)
    if crop_start < 0 or crop_start >= source_length:
        raise ValueError(f"extend starts outside source rows: start={crop_start}, source_rows={source_length}")
    if annotated_crop_end > source_length:
        raise ValueError(
            f"return ends beyond the supported one-row clamp: end={annotated_crop_end}, source_rows={source_length}"
        )
    if crop_end < crop_start:
        raise ValueError(f"empty training crop: [{crop_start}, {crop_end}]")

    progress = np.full((crop_end - crop_start + 1,), np.nan, dtype=np.float32)
    for interval in annotation.stages:
        effective_end = min(interval.end, crop_end)
        local_start = interval.start - crop_start
        local_end = effective_end - crop_start
        if local_end < local_start:
            raise ValueError(f"stage {interval.name} has no source frames after cropping")
        low, high = STAGE_PROGRESS[interval.name]
        progress[local_start : local_end + 1] = np.linspace(
            low,
            high,
            num=local_end - local_start + 1,
            endpoint=True,
            dtype=np.float32,
        )

    if np.isnan(progress).any():
        missing = np.flatnonzero(np.isnan(progress))
        raise ValueError(f"progress has {missing.size} unlabeled frames; first local frame={missing[0]}")
    if np.any(np.diff(progress) < -1e-7):
        first = int(np.flatnonzero(np.diff(progress) < -1e-7)[0])
        raise ValueError(f"progress is not monotonic at local frames {first}->{first + 1}")
    if not np.isclose(progress[0], 0.0) or not np.isclose(progress[-1], 1.0):
        raise ValueError(f"progress endpoints must be 0 and 1, got {progress[0]} and {progress[-1]}")
    return progress, crop_start, crop_end, final_endpoint_clamped

def _format_episode_path(pattern: str, episode_index: int, chunk_size: int, video_key: str | None = None) -> Path:
    values = {
        "episode_chunk": episode_index // chunk_size,
        "episode_index": episode_index,
        "video_key": video_key,
    }
    return Path(pattern.format(**values))


def _probe_video(path: Path, expected_frames: int, expected_fps: int, ffprobe_bin: str) -> str | None:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        streams = json.loads(result.stdout).get("streams", [])
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        return f"ffprobe failed for {path}: {error}"
    if len(streams) != 1:
        return f"Expected one video stream in {path}, got {len(streams)}"
    stream = streams[0]
    frame_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    try:
        frame_count = int(frame_value)
    except (TypeError, ValueError):
        return f"ffprobe did not report a frame count for {path}"
    if frame_count != expected_frames:
        return f"Video {path} has {frame_count} frames, expected {expected_frames}"
    try:
        fps = Fraction(stream["r_frame_rate"])
    except (KeyError, ValueError, ZeroDivisionError):
        return f"ffprobe did not report a valid frame rate for {path}"
    if fps != expected_fps:
        return f"Video {path} has fps={fps}, expected {expected_fps}"
    return None


def _validate_source_episode(
    annotation: Annotation,
    source_root: Path,
    source_info: dict[str, Any],
    source_episode: dict[str, Any],
    source_stats: dict[str, Any],
    *,
    probe_videos: bool,
    ffprobe_bin: str,
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any]]:
    del source_stats  # Presence is validated by the caller; values are used during writing.
    errors: list[str] = []
    chunk_size = int(source_info["chunks_size"])
    source_index = annotation.source_episode_index
    expected_rows = annotation.annotation_end_frame + 1
    parquet_path = source_root / _format_episode_path(source_info["data_path"], source_index, chunk_size)
    audit: dict[str, Any] = {
        "annotation_file": annotation.annotation_file,
        "source_episode_index": source_index,
        "annotation_end_frame": annotation.annotation_end_frame,
        "expected_closed_source_rows": expected_rows,
        "source_parquet": str(parquet_path),
        "source_metadata_rows": source_episode.get("length"),
        "source_parquet_rows": None,
        "source_rows_minus_expected_closed_rows": None,
        "row_relation": "unavailable",
        "annotated_training_start_frame": annotation.stages[0].start,
        "annotated_training_end_frame": annotation.stages[-1].end,
        "effective_training_start_frame": None,
        "effective_training_end_frame": None,
        "training_rows": None,
        "initial_idle_frames_removed": None,
        "trailing_idle_frames_removed": None,
        "final_return_endpoint_clamped": False,
        "adjustments": [],
        "missing_columns": [],
        "missing_videos": [],
        "status": "error",
        "errors": [],
    }
    if not parquet_path.is_file():
        errors.append(f"Missing source Parquet: {parquet_path}")
        audit["errors"] = errors
        return None, errors, audit

    try:
        import pyarrow.parquet as pq  # noqa: PLC0415 - keep annotation-only mode lightweight.

        parquet_file = pq.ParquetFile(parquet_path)
        source_length = parquet_file.metadata.num_rows
        columns = set(parquet_file.schema_arrow.names)
    except Exception as error:
        errors.append(f"Cannot read {parquet_path}: {error}")
        audit["errors"] = errors
        return None, errors, audit

    row_delta = source_length - expected_rows
    if row_delta == 0:
        row_relation = "matches_closed_annotation"
    elif row_delta == -1:
        row_relation = "source_one_row_short_of_closed_annotation"
    else:
        row_relation = "other_row_count_mismatch"
        errors.append(
            f"closed annotation expects {expected_rows} source rows, got {source_length}; "
            "only an exact match or one missing final row is supported"
        )
    audit.update(
        {
            "source_parquet_rows": source_length,
            "source_rows_minus_expected_closed_rows": row_delta,
            "row_relation": row_relation,
        }
    )

    metadata_length = source_episode.get("length")
    if metadata_length != source_length:
        errors.append(f"Source metadata length={metadata_length}, Parquet rows={source_length}")
    required_columns = {
        "actions",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    }
    missing_columns = sorted(required_columns - columns)
    audit["missing_columns"] = missing_columns
    if missing_columns:
        errors.append(f"Source Parquet is missing columns: {missing_columns}")
    if "progress" in columns:
        errors.append("Source Parquet already contains a progress column")

    try:
        progress, crop_start, crop_end, final_endpoint_clamped = make_progress(annotation, source_length)
    except ValueError as error:
        errors.append(str(error))
        progress = None
        crop_start = None
        crop_end = None
        final_endpoint_clamped = False
    if progress is not None and crop_start is not None and crop_end is not None:
        audit.update(
            {
                "effective_training_start_frame": crop_start,
                "effective_training_end_frame": crop_end,
                "training_rows": int(progress.shape[0]),
                "initial_idle_frames_removed": crop_start,
                "trailing_idle_frames_removed": source_length - crop_end - 1,
                "final_return_endpoint_clamped": final_endpoint_clamped,
            }
        )
        if final_endpoint_clamped:
            audit["adjustments"].append("clamp_return_end_to_last_source_frame")

    video_paths: list[tuple[str, Path]] = []
    missing_videos: list[str] = []
    video_keys = [key for key, feature in source_info["features"].items() if feature["dtype"] == "video"]
    for video_key in video_keys:
        relative_path = _format_episode_path(
            source_info["video_path"], source_index, chunk_size, video_key=video_key
        )
        video_path = source_root / relative_path
        video_paths.append((video_key, video_path))
        if not video_path.is_file():
            missing_videos.append(str(video_path))
            errors.append(f"Missing source video: {video_path}")
        elif probe_videos:
            probe_error = _probe_video(video_path, source_length, int(source_info["fps"]), ffprobe_bin)
            if probe_error is not None:
                errors.append(probe_error)
    audit["missing_videos"] = missing_videos
    audit["errors"] = errors

    if errors or progress is None or crop_start is None or crop_end is None:
        return None, errors, audit
    audit["status"] = "valid"
    return {
        "source_parquet": parquet_path,
        "source_videos": tuple(video_paths),
        "source_length": source_length,
        "source_start_frame": crop_start,
        "source_end_frame": crop_end,
        "length": int(progress.shape[0]),
        "final_endpoint_clamped": final_endpoint_clamped,
        "progress": progress,
    }, [], audit

def validate_source(
    annotations: list[Annotation],
    source_root: Path,
    *,
    workers: int,
    probe_videos: bool,
    ffprobe_bin: str,
) -> tuple[
    dict[str, Any],
    dict[int, dict[str, Any]],
    list[EpisodePlan],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    info_path = source_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing source metadata: {info_path}")
    source_info = _read_json(info_path)
    if int(source_info.get("fps", -1)) != EXPECTED_FPS:
        raise ValueError(f"Source fps must be {EXPECTED_FPS}, got {source_info.get('fps')}")
    action_feature = source_info.get("features", {}).get("actions")
    if action_feature is None or action_feature.get("dtype") != "float32" or action_feature.get("shape") != [14]:
        raise ValueError(f"Expected actions float32[{EXPECTED_ACTION_DIM}], got {action_feature}")
    if source_info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1, got {source_info.get('codebase_version')}")

    wanted_indices = {annotation.source_episode_index for annotation in annotations}
    episodes = _read_jsonl_subset(source_root / "meta" / "episodes.jsonl", wanted_indices)
    stats = _read_jsonl_subset(source_root / "meta" / "episodes_stats.jsonl", wanted_indices)
    missing_episodes = sorted(wanted_indices - episodes.keys())
    missing_stats = sorted(wanted_indices - stats.keys())
    if missing_episodes:
        raise ValueError(f"Source episodes.jsonl is missing {len(missing_episodes)} selected episodes: {missing_episodes[:20]}")
    if missing_stats:
        raise ValueError(
            f"Source episodes_stats.jsonl is missing {len(missing_stats)} selected episodes: {missing_stats[:20]}"
        )

    def validate_one(
        annotation: Annotation,
    ) -> tuple[Annotation, dict[str, Any] | None, list[str], dict[str, Any]]:
        result, errors, audit = _validate_source_episode(
            annotation,
            source_root,
            source_info,
            episodes[annotation.source_episode_index],
            stats[annotation.source_episode_index],
            probe_videos=probe_videos,
            ffprobe_bin=ffprobe_bin,
        )
        return annotation, result, errors, audit

    validated: list[tuple[Annotation, dict[str, Any]]] = []
    source_errors: list[dict[str, Any]] = []
    source_audits: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for position, (annotation, result, errors, audit) in enumerate(
            executor.map(validate_one, annotations), start=1
        ):
            source_audits.append(audit)
            if errors or result is None:
                source_errors.append(
                    {
                        "annotation_file": annotation.annotation_file,
                        "source_episode_index": annotation.source_episode_index,
                        "errors": errors,
                    }
                )
            else:
                validated.append((annotation, result))
            if position % 250 == 0 or position == len(annotations):
                LOGGER.info("Validated source episodes: %s/%s", position, len(annotations))

    plans: list[EpisodePlan] = []
    global_index_start = 0
    for output_index, (annotation, result) in enumerate(validated):
        plans.append(
            EpisodePlan(
                output_episode_index=output_index,
                source_episode_index=annotation.source_episode_index,
                annotation_file=annotation.annotation_file,
                length=result["length"],
                source_length=result["source_length"],
                source_start_frame=result["source_start_frame"],
                source_end_frame=result["source_end_frame"],
                final_endpoint_clamped=result["final_endpoint_clamped"],
                global_index_start=global_index_start,
                source_parquet=result["source_parquet"],
                source_videos=result["source_videos"],
                progress=result["progress"],
            )
        )
        global_index_start += result["length"]
    return source_info, stats, plans, source_errors, source_audits


def _array_stats(values: Any) -> dict[str, list[float | int]]:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Cannot compute numeric statistics for dtype={array.dtype}")
    return {
        "min": np.atleast_1d(np.min(array, axis=0)).astype(np.float64).tolist(),
        "max": np.atleast_1d(np.max(array, axis=0)).astype(np.float64).tolist(),
        "mean": np.atleast_1d(np.mean(array, axis=0)).astype(np.float64).tolist(),
        "std": np.atleast_1d(np.std(array, axis=0)).astype(np.float64).tolist(),
        "count": [int(array.shape[0])],
    }


def _output_episode_stats(
    plan: EpisodePlan,
    dataset: Any,
    source_stats_row: dict[str, Any],
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key, source_feature_stats in source_stats_row["stats"].items():
        if key in dataset.column_names:
            stats[key] = _array_stats(dataset[key])
        else:
            stats[key] = copy.deepcopy(source_feature_stats)
    stats["progress"] = _array_stats(plan.progress.reshape(-1, 1))
    return {"episode_index": plan.output_episode_index, "stats": stats}


def _rewrite_parquet(
    plan: EpisodePlan,
    output_path: Path,
    compression: str,
    source_stats_row: dict[str, Any],
) -> dict[str, Any]:
    from datasets import Dataset  # noqa: PLC0415 - only required when writing Parquet.
    from datasets import Sequence  # noqa: PLC0415 - only required when writing Parquet.
    from datasets import Value  # noqa: PLC0415 - only required when writing Parquet.

    dataset = Dataset.from_parquet(str(plan.source_parquet))
    if dataset.num_rows != plan.source_length:
        raise ValueError(
            f"{plan.source_parquet} changed during generation: expected {plan.source_length} source rows, "
            f"got {dataset.num_rows}"
        )
    dataset = dataset.select(range(plan.source_start_frame, plan.source_end_frame + 1))
    if dataset.num_rows != plan.length:
        raise ValueError(f"cropped Parquet has {dataset.num_rows} rows, expected {plan.length}")

    frame_index = np.arange(plan.length, dtype=np.int64)
    replacement_columns = {
        "timestamp": frame_index.astype(np.float32) / EXPECTED_FPS,
        "frame_index": frame_index,
        "episode_index": np.full((plan.length,), plan.output_episode_index, dtype=np.int64),
        "index": np.arange(
            plan.global_index_start,
            plan.global_index_start + plan.length,
            dtype=np.int64,
        ),
        "task_index": np.zeros((plan.length,), dtype=np.int64),
    }
    original_features = {key: dataset.features[key] for key in replacement_columns}
    dataset = dataset.remove_columns(list(replacement_columns))
    for key, values in replacement_columns.items():
        dataset = dataset.add_column(key, values, feature=original_features[key])
    dataset = dataset.add_column(
        "progress",
        plan.progress.reshape(-1, 1).tolist(),
        feature=Sequence(feature=Value("float32"), length=1),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path), compression=compression)
    return _output_episode_stats(plan, dataset, source_stats_row)


def _trim_video(
    source: Path,
    destination: Path,
    *,
    start_frame: int,
    end_frame: int,
    ffmpeg_bin: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame_count = end_frame - start_frame + 1
    video_filter = (
        f"trim=start_frame={start_frame}:end_frame={end_frame + 1},"
        "setpts=PTS-STARTPTS"
    )
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        video_filter,
        "-frames:v",
        str(frame_count),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-qp",
        "0",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = error.stderr.strip() if isinstance(error, subprocess.CalledProcessError) and error.stderr else ""
        detail = f": {stderr}" if stderr else ""
        raise RuntimeError(f"ffmpeg failed for {source}{detail}") from error
    if not destination.is_file() or destination.stat().st_size == 0:
        raise OSError(f"ffmpeg did not create a valid output file: {destination}")

def _build_output_info(source_info: dict[str, Any], plans: list[EpisodePlan]) -> dict[str, Any]:
    info = copy.deepcopy(source_info)
    total_episodes = len(plans)
    total_frames = sum(plan.length for plan in plans)
    chunk_size = int(info["chunks_size"])
    video_count = sum(1 for feature in info["features"].values() if feature["dtype"] == "video")
    info.update(
        {
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": total_episodes * video_count,
            "total_chunks": math.ceil(total_episodes / chunk_size),
            "splits": {"train": f"0:{total_episodes}"},
        }
    )
    info["features"]["progress"] = {
        "dtype": "float32",
        "shape": [1],
        "names": ["progress"],
    }
    return info


def _destination_path(info: dict[str, Any], pattern_key: str, episode_index: int, video_key: str | None = None) -> Path:
    return _format_episode_path(info[pattern_key], episode_index, int(info["chunks_size"]), video_key)


def build_dataset(
    output_root: Path,
    source_root: Path,
    annotation_dir: Path,
    source_info: dict[str, Any],
    source_stats: dict[int, dict[str, Any]],
    plans: list[EpisodePlan],
    excluded: list[ExcludedAnnotation],
    warnings: list[WarningRecord],
    *,
    compression: str,
    probe_videos: bool,
    ffmpeg_bin: str,
    ffprobe_bin: str,
) -> None:
    output_root = output_root.resolve()
    source_root = source_root.resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("Output root must not be the source root or a directory inside it")
    staging_root = output_root.with_name(f"{output_root.name}.incomplete")
    if output_root.exists():
        raise FileExistsError(f"Output directory already exists: {output_root}")
    if staging_root.exists():
        raise FileExistsError(
            f"Staging directory already exists from an earlier run: {staging_root}. Inspect or move it before retrying."
        )
    staging_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir()

    output_info = _build_output_info(source_info, plans)
    episodes_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    try:
        for position, plan in enumerate(plans, start=1):
            destination_parquet = staging_root / _destination_path(
                output_info, "data_path", plan.output_episode_index
            )
            output_stats = _rewrite_parquet(
                plan,
                destination_parquet,
                compression,
                source_stats[plan.source_episode_index],
            )
            for video_key, source_video in plan.source_videos:
                destination_video = staging_root / _destination_path(
                    output_info,
                    "video_path",
                    plan.output_episode_index,
                    video_key,
                )
                _trim_video(
                    source_video,
                    destination_video,
                    start_frame=plan.source_start_frame,
                    end_frame=plan.source_end_frame,
                    ffmpeg_bin=ffmpeg_bin,
                )
                if probe_videos:
                    probe_error = _probe_video(destination_video, plan.length, EXPECTED_FPS, ffprobe_bin)
                    if probe_error is not None:
                        raise ValueError(probe_error)

            episodes_rows.append(
                {
                    "episode_index": plan.output_episode_index,
                    "tasks": [TASK],
                    "length": plan.length,
                }
            )
            stats_rows.append(output_stats)
            mapping_rows.append(
                {
                    "episode_index": plan.output_episode_index,
                    "source_episode_index": plan.source_episode_index,
                    "annotation_file": plan.annotation_file,
                    "source_start_frame": plan.source_start_frame,
                    "source_end_frame": plan.source_end_frame,
                    "length": plan.length,
                    "final_return_endpoint_clamped": plan.final_endpoint_clamped,
                }
            )
            if position % 100 == 0 or position == len(plans):
                LOGGER.info("Generated episodes: %s/%s", position, len(plans))

        meta_dir = staging_root / "meta"
        _write_json(meta_dir / "info.json", output_info)
        _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": TASK}])
        _write_jsonl(meta_dir / "episodes.jsonl", episodes_rows)
        _write_jsonl(meta_dir / "episodes_stats.jsonl", stats_rows)
        _write_jsonl(meta_dir / "source_episode_mapping.jsonl", mapping_rows)
        _write_jsonl(meta_dir / "excluded_annotations.jsonl", [asdict(item) for item in excluded])
        _write_jsonl(meta_dir / "progress_warnings.jsonl", [asdict(item) for item in warnings])
        _write_json(
            meta_dir / "progress_generation.json",
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "source_dataset_dir": str(source_root),
                "annotation_dir": str(annotation_dir.resolve()),
                "total_annotation_files": len(plans) + len(excluded),
                "excluded_annotation_count": len(excluded),
                "output_episode_count": len(plans),
                "output_frame_count": sum(plan.length for plan in plans),

                "task": TASK,
                "stage_progress": {key: list(value) for key, value in STAGE_PROGRESS.items()},
                "stage_intervals": "closed [start_frame, end_frame]",
                "training_crop": "closed [extend.start_frame, return.end_frame]",
                "idle_frames": "removed from both ends",
                "interpolation": "numpy.linspace(low, high, num=stage_frames, endpoint=True)",
                "final_return_endpoint_clamped_count": sum(plan.final_endpoint_clamped for plan in plans),
                "video_mode": "frame-accurate ffmpeg trim; H.264 libx264 QP 0 re-encode without faststart",
                "video_frame_probe": probe_videos,
                "compression": compression,
            },
        )
        staging_root.rename(output_root)
    except Exception:
        LOGGER.error("Generation failed. Partial output was kept for inspection at: %s", staging_root)
        raise


def _annotation_summary(
    annotation_dir: Path,
    annotations: list[Annotation],
    excluded: list[ExcludedAnnotation],
    warnings: list[WarningRecord],
) -> dict[str, Any]:
    return {
        "annotation_dir": str(annotation_dir.resolve()),
        "annotation_file_count": len(annotations) + len(excluded),
        "valid_annotation_count": len(annotations),
        "excluded_annotation_count": len(excluded),
        "excluded_by_reason": dict(sorted(Counter(item.reason for item in excluded).items())),
        "warning_count": len(warnings),
        "warnings_by_code": dict(sorted(Counter(item.code for item in warnings).items())),
    }


def _check_expected_counts(summary: dict[str, Any], expected_annotations: int, expected_valid: int) -> list[str]:
    errors = []
    if expected_annotations > 0 and summary["annotation_file_count"] != expected_annotations:
        errors.append(
            f"Expected {expected_annotations} annotation files, got {summary['annotation_file_count']}"
        )
    if expected_valid > 0 and summary["valid_annotation_count"] != expected_valid:
        errors.append(f"Expected {expected_valid} valid annotations, got {summary['valid_annotation_count']}")
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Validate annotation JSON files without accessing the source dataset.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Build the output dataset. Without this flag the script is read-only.",
    )

    parser.add_argument("--compression", default="zstd", help="Parquet compression codec.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel source-validation workers.")
    parser.add_argument(
        "--probe-videos",
        action="store_true",
        help="Use ffprobe to count every selected camera video's frames. This is much slower.",
    )
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_ROOT,
        help="Root directory for timestamped per-run logs.",
    )
    parser.add_argument("--report-path", type=Path, help="Optional JSON path for the check report.")
    parser.add_argument("--expected-annotation-count", type=int, default=EXPECTED_ANNOTATION_COUNT)
    parser.add_argument("--expected-valid-episodes", type=int, default=EXPECTED_VALID_EPISODE_COUNT)
    args = parser.parse_args()
    if args.annotations_only and args.write:
        parser.error("--annotations-only and --write cannot be used together")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main() -> int:
    args = _parse_args()
    run_dir = _setup_run_logging(args.log_dir)
    LOGGER.info("Run logs: %s", run_dir)

    annotations, excluded, warnings = load_annotations(args.annotation_dir)
    summary = _annotation_summary(args.annotation_dir, annotations, excluded, warnings)
    summary.update(
        {
            "run_started_at_utc": datetime.now(UTC).isoformat(),
            "run_log_dir": str(run_dir),
            "mode": "annotations_only" if args.annotations_only else "write" if args.write else "check",
            "result": "running",
        }
    )
    annotation_audits = _annotation_audit_rows(annotations, excluded, warnings)
    _write_jsonl(run_dir / "annotation_validation.jsonl", annotation_audits)
    _write_jsonl(run_dir / "source_validation.jsonl", [])

    count_errors = _check_expected_counts(
        summary,
        args.expected_annotation_count,
        args.expected_valid_episodes,
    )
    summary["count_errors"] = count_errors
    LOGGER.info(
        "Annotations: total=%s valid=%s excluded=%s warnings=%s",
        summary["annotation_file_count"],
        summary["valid_annotation_count"],
        summary["excluded_annotation_count"],
        summary["warning_count"],
    )

    if args.annotations_only:
        summary["result"] = "failed" if count_errors else "annotations_valid"
        summary["run_finished_at_utc"] = datetime.now(UTC).isoformat()
        _persist_summary(summary, run_dir, args.report_path)
        LOGGER.info("Result: %s", summary["result"])
        return 1 if count_errors else 0

    if count_errors:
        summary["result"] = "failed"
        summary["run_finished_at_utc"] = datetime.now(UTC).isoformat()
        _persist_summary(summary, run_dir, args.report_path)
        LOGGER.error("Annotation counts did not match expectations; source validation was not run.")
        return 1

    try:
        source_info, source_stats, plans, source_errors, source_audits = validate_source(
            annotations,
            args.source_root,
            workers=args.workers,
            probe_videos=args.probe_videos,
            ffprobe_bin=args.ffprobe_bin,
        )
    except Exception as error:
        summary["source_validation_error"] = str(error)
        summary["result"] = "failed"
        summary["run_finished_at_utc"] = datetime.now(UTC).isoformat()
        _persist_summary(summary, run_dir, args.report_path)
        LOGGER.error("Source validation could not be completed: %s", error)
        return 1

    _write_jsonl(run_dir / "source_validation.jsonl", source_audits)
    row_relation_counts = dict(sorted(Counter(row["row_relation"] for row in source_audits).items()))
    source_status_counts = dict(sorted(Counter(row["status"] for row in source_audits).items()))
    summary.update(
        {
            "source_root": str(args.source_root.resolve()),
            "source_dataset_total_episodes": source_info["total_episodes"],
            "source_checked_episode_count": len(annotations),
            "source_valid_episode_count": len(plans),
            "source_error_count": len(source_errors),
            "source_row_relation_counts": row_relation_counts,
            "source_status_counts": source_status_counts,
            "source_frame_count": sum(plan.source_length for plan in plans),
            "output_frame_count": sum(plan.length for plan in plans),
            "initial_idle_frames_removed": sum(plan.source_start_frame for plan in plans),
            "trailing_idle_frames_removed": sum(
                plan.source_length - plan.source_end_frame - 1 for plan in plans
            ),
            "final_return_endpoint_clamped_count": sum(plan.final_endpoint_clamped for plan in plans),
            "video_frame_probe": args.probe_videos,
        }
    )
    LOGGER.info(
        "Source: checked=%s valid=%s errors=%s row_relations=%s clamped_return_end=%s",
        len(source_audits),
        len(plans),
        len(source_errors),
        row_relation_counts,
        sum(plan.final_endpoint_clamped for plan in plans),
    )

    if source_errors:
        summary["result"] = "failed"
        summary["run_finished_at_utc"] = datetime.now(UTC).isoformat()
        _persist_summary(summary, run_dir, args.report_path)
        LOGGER.error("Source validation failed; no output dataset was created.")
        return 1
    if not args.write:
        summary["result"] = "source_valid"
        summary["run_finished_at_utc"] = datetime.now(UTC).isoformat()
        _persist_summary(summary, run_dir, args.report_path)
        LOGGER.info("Check completed successfully. Re-run with --write to create the dataset.")
        return 0

    try:
        build_dataset(
            args.output_root,
            args.source_root,
            args.annotation_dir,
            source_info,
            source_stats,
            plans,
            excluded,
            warnings,
            compression=args.compression,
            probe_videos=args.probe_videos,
            ffmpeg_bin=args.ffmpeg_bin,
            ffprobe_bin=args.ffprobe_bin,
        )
    except Exception as error:
        summary["generation_error"] = str(error)
        summary["result"] = "failed"
        summary["run_finished_at_utc"] = datetime.now(UTC).isoformat()
        _persist_summary(summary, run_dir, args.report_path)
        LOGGER.error("Dataset generation failed: %s", error)
        return 1

    summary["result"] = "dataset_created"
    summary["output_root"] = str(args.output_root.resolve())
    summary["run_finished_at_utc"] = datetime.now(UTC).isoformat()
    _persist_summary(summary, run_dir, args.report_path)
    LOGGER.info("Created progress dataset: %s", args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
