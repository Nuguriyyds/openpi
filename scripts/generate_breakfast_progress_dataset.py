"""Add local 0-to-1 progress labels to the pre-split breakfast dataset.

The source dataset is read-only. By default this script validates all episodes;
pass ``--write`` to create a separate LeRobot v2.1 dataset. Videos are copied
byte-for-byte without links or re-encoding.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import copy
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


DEFAULT_SOURCE_ROOT = Path("/mnt/data/dataset/ei/huggingface/ljc/agilex_make_breakfast_380_subtask")
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/data/dataset/ei/huggingface/wyt/agilex_make_breakfast_380_subtask_furniturevla_progress"
)
DEFAULT_LOG_ROOT = Path(
    "/mnt/data/dataset/ei/huggingface/wyt/agilex_make_breakfast_380_subtask_furniturevla_progress_logs"
)

EXPECTED_EPISODES = 1520
EXPECTED_SOURCE_TASKS = 380
EXPECTED_SUBTASKS = 4
EXPECTED_FPS = 30
EXPECTED_ACTION_DIM = 14

LOGGER = logging.getLogger("breakfast_progress_dataset")


@dataclass(frozen=True)
class Episode:
    episode_index: int
    source_episode_index: int
    subtask_index: int
    subtask_id: str
    task: str
    length: int
    source_start_frame: int
    source_end_frame: int
    parquet_path: Path
    video_paths: tuple[tuple[str, Path], ...]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


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


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_file, destination.open("xb") as destination_file:
        shutil.copyfileobj(source_file, destination_file, length=16 * 1024 * 1024)
    if source.stat().st_size != destination.stat().st_size:
        raise OSError(f"Copied file size mismatch: {source} -> {destination}")


def _setup_logging(log_root: Path) -> Path:
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


def _format_episode_path(pattern: str, episode_index: int, chunk_size: int, video_key: str | None = None) -> Path:
    return Path(
        pattern.format(
            episode_chunk=episode_index // chunk_size,
            episode_index=episode_index,
            video_key=video_key,
        )
    )


def _indexed(rows: list[dict[str, Any]], key: str, path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Invalid {key}={value!r} in {path}")
        if value in result:
            raise ValueError(f"Duplicate {key}={value} in {path}")
        result[value] = row
    return result


def load_dataset_metadata(source_root: Path) -> tuple[dict[str, Any], list[Episode], list[dict[str, Any]]]:
    meta = source_root / "meta"
    info = _read_json(meta / "info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1, got {info.get('codebase_version')}")
    if int(info.get("fps", -1)) != EXPECTED_FPS:
        raise ValueError(f"Expected {EXPECTED_FPS} fps, got {info.get('fps')}")
    action_feature = info.get("features", {}).get("actions")
    if action_feature is None or action_feature.get("dtype") != "float32" or action_feature.get("shape") != [14]:
        raise ValueError(f"Expected actions float32[{EXPECTED_ACTION_DIM}], got {action_feature}")
    if "progress" in info.get("features", {}):
        raise ValueError("Source dataset already contains a progress feature")

    tasks_path = meta / "tasks.jsonl"
    episodes_path = meta / "episodes.jsonl"
    stats_path = meta / "episodes_stats.jsonl"
    mapping_path = meta / "source_mapping.jsonl"
    tasks = _indexed(_read_jsonl(tasks_path), "task_index", tasks_path)
    episode_rows = _indexed(_read_jsonl(episodes_path), "episode_index", episodes_path)
    stats_rows = _indexed(_read_jsonl(stats_path), "episode_index", stats_path)
    mapping_rows = _indexed(_read_jsonl(mapping_path), "episode_index", mapping_path)

    total_episodes = int(info.get("total_episodes", -1))
    expected_indices = set(range(total_episodes))
    for name, rows in (("episodes", episode_rows), ("episodes_stats", stats_rows), ("source_mapping", mapping_rows)):
        if set(rows) != expected_indices:
            missing = sorted(expected_indices - set(rows))
            extra = sorted(set(rows) - expected_indices)
            raise ValueError(f"{name} indices mismatch: missing={missing[:10]}, extra={extra[:10]}")
    if total_episodes != EXPECTED_EPISODES:
        raise ValueError(f"Expected {EXPECTED_EPISODES} episodes, got {total_episodes}")
    if len(tasks) != EXPECTED_SUBTASKS:
        raise ValueError(f"Expected {EXPECTED_SUBTASKS} tasks, got {len(tasks)}")

    chunk_size = int(info["chunks_size"])
    video_keys = tuple(key for key, feature in info["features"].items() if feature["dtype"] == "video")
    episodes: list[Episode] = []
    for episode_index in range(total_episodes):
        episode_row = episode_rows[episode_index]
        mapping = mapping_rows[episode_index]
        subtask_index = int(mapping.get("subtask_index", -1))
        source_episode_index = int(mapping.get("source_episode_index", -1))
        length = int(episode_row.get("length", -1))
        start_frame = int(mapping.get("source_start_frame", -1))
        end_frame = int(mapping.get("source_end_frame", -1))
        task = tasks.get(subtask_index, {}).get("task")
        row_tasks = episode_row.get("tasks")

        if subtask_index != episode_index % EXPECTED_SUBTASKS:
            raise ValueError(f"Episode {episode_index}: expected subtask_index={episode_index % 4}, got {subtask_index}")
        if source_episode_index != episode_index // EXPECTED_SUBTASKS:
            raise ValueError(
                f"Episode {episode_index}: expected source_episode_index={episode_index // 4}, got {source_episode_index}"
            )
        if start_frame < 0 or end_frame <= start_frame or end_frame - start_frame != length:
            raise ValueError(
                f"Episode {episode_index}: invalid half-open source interval [{start_frame},{end_frame}) for length={length}"
            )
        if length < 2:
            raise ValueError(f"Episode {episode_index}: progress requires at least two frames, got {length}")
        if row_tasks != [task] or mapping.get("task") != task:
            raise ValueError(f"Episode {episode_index}: task metadata does not match task_index={subtask_index}")
        if int(mapping.get("length", -1)) != length:
            raise ValueError(f"Episode {episode_index}: source_mapping length does not match episodes.jsonl")
        stats_count = stats_rows[episode_index].get("stats", {}).get("actions", {}).get("count")
        if stats_count != [length]:
            raise ValueError(f"Episode {episode_index}: action stats count={stats_count}, expected [{length}]")

        parquet_path = source_root / _format_episode_path(info["data_path"], episode_index, chunk_size)
        video_paths = tuple(
            (
                video_key,
                source_root
                / _format_episode_path(info["video_path"], episode_index, chunk_size, video_key=video_key),
            )
            for video_key in video_keys
        )
        episodes.append(
            Episode(
                episode_index=episode_index,
                source_episode_index=source_episode_index,
                subtask_index=subtask_index,
                subtask_id=str(mapping.get("subtask_id")),
                task=str(task),
                length=length,
                source_start_frame=start_frame,
                source_end_frame=end_frame,
                parquet_path=parquet_path,
                video_paths=video_paths,
            )
        )

    if len({episode.source_episode_index for episode in episodes}) != EXPECTED_SOURCE_TASKS:
        raise ValueError(f"Expected {EXPECTED_SOURCE_TASKS} source breakfast tasks")
    for episode_index in range(1, total_episodes):
        episode = episodes[episode_index]
        previous = episodes[episode_index - 1]
        if episode.subtask_index > 0 and previous.source_end_frame != episode.source_start_frame:
            raise ValueError(f"Episodes {episode_index - 1}->{episode_index}: non-contiguous source boundaries")
    return info, episodes, [stats_rows[index] for index in range(total_episodes)]


def validate_episode(episode: Episode) -> dict[str, Any]:
    errors: list[str] = []
    parquet_rows: int | None = None
    missing_columns: list[str] = []
    if not episode.parquet_path.is_file():
        errors.append(f"Missing Parquet: {episode.parquet_path}")
    else:
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415

            parquet = pq.ParquetFile(episode.parquet_path)
            parquet_rows = parquet.metadata.num_rows
            columns = set(parquet.schema_arrow.names)
            required = {"actions", "timestamp", "frame_index", "episode_index", "index", "task_index"}
            missing_columns = sorted(required - columns)
            if parquet_rows != episode.length:
                errors.append(f"Parquet rows={parquet_rows}, expected={episode.length}")
            if missing_columns:
                errors.append(f"Missing Parquet columns: {missing_columns}")
            if "progress" in columns:
                errors.append("Parquet already contains progress")
        except Exception as error:
            errors.append(f"Cannot read Parquet: {error}")

    missing_videos = [str(path) for _, path in episode.video_paths if not path.is_file()]
    if missing_videos:
        errors.append(f"Missing {len(missing_videos)} videos")
    return {
        "episode_index": episode.episode_index,
        "source_episode_index": episode.source_episode_index,
        "subtask_index": episode.subtask_index,
        "subtask_id": episode.subtask_id,
        "length": episode.length,
        "progress_start": 0.0,
        "progress_end": 1.0,
        "source_parquet_rows": parquet_rows,
        "missing_columns": missing_columns,
        "missing_videos": missing_videos,
        "status": "error" if errors else "valid",
        "errors": errors,
    }


def _progress(length: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, num=length, endpoint=True, dtype=np.float32)


def _progress_stats(length: int) -> dict[str, list[float | int]]:
    values = _progress(length).astype(np.float64)
    return {
        "min": [float(values.min())],
        "max": [float(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [length],
    }


def _rewrite_parquet(episode: Episode, output_path: Path, compression: str) -> None:
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    parquet = pq.ParquetFile(episode.parquet_path)
    table = parquet.read(use_threads=False)
    if table.num_rows != episode.length:
        raise ValueError(f"{episode.parquet_path} changed: rows={table.num_rows}, expected={episode.length}")
    if "progress" in table.column_names:
        raise ValueError(f"{episode.parquet_path} already contains progress")
    values = pa.array(_progress(episode.length), type=pa.float32())
    progress_column = pa.FixedSizeListArray.from_arrays(values, 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.append_column("progress", progress_column), output_path, compression=compression)


def build_dataset(
    source_root: Path,
    output_root: Path,
    source_info: dict[str, Any],
    episodes: list[Episode],
    source_stats: list[dict[str, Any]],
    *,
    compression: str,
) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("Output root must be separate from and outside the source dataset")
    staging_root = output_root.with_name(f"{output_root.name}.incomplete")
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    if staging_root.exists():
        raise FileExistsError(f"Staging output already exists: {staging_root}")
    staging_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir()

    output_info = copy.deepcopy(source_info)
    output_info["features"]["progress"] = {"dtype": "float32", "shape": [1], "names": ["progress"]}
    output_stats = copy.deepcopy(source_stats)
    for episode, stats_row in zip(episodes, output_stats, strict=True):
        stats_row["stats"]["progress"] = _progress_stats(episode.length)

    try:
        source_meta = source_root / "meta"
        output_meta = staging_root / "meta"
        for source_file in source_meta.rglob("*"):
            if not source_file.is_file():
                continue
            relative_path = source_file.relative_to(source_meta)
            if relative_path.as_posix() in {"info.json", "episodes_stats.jsonl"}:
                continue
            _copy_file(source_file, output_meta / relative_path)
        _write_json(output_meta / "info.json", output_info)
        _write_jsonl(output_meta / "episodes_stats.jsonl", output_stats)
        _write_json(
            output_meta / "progress_generation.json",
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "source_dataset_dir": str(source_root),
                "progress_scope": "local subtask completion",
                "progress": "numpy.linspace(0, 1, episode_length, endpoint=True)",
                "source_episodes": EXPECTED_SOURCE_TASKS,
                "subtasks_per_source_episode": EXPECTED_SUBTASKS,
                "video_mode": "byte copy; no links; no re-encoding",
            },
        )

        chunk_size = int(source_info["chunks_size"])

        def build_one(episode: Episode) -> None:
            output_parquet = staging_root / _format_episode_path(
                source_info["data_path"], episode.episode_index, chunk_size
            )
            _rewrite_parquet(episode, output_parquet, compression)
            for video_key, source_video in episode.video_paths:
                output_video = staging_root / _format_episode_path(
                    source_info["video_path"], episode.episode_index, chunk_size, video_key=video_key
                )
                _copy_file(source_video, output_video)

        # PyArrow can segfault when several native Parquet readers run concurrently
        # in the server environment. Keep generation sequential and disable
        # PyArrow's internal reader threads in _rewrite_parquet.
        for position, episode in enumerate(episodes, start=1):
            build_one(episode)
            if position % 100 == 0 or position == len(episodes):
                LOGGER.info("Generated episodes: %s/%s", position, len(episodes))
        staging_root.rename(output_root)
    except Exception:
        LOGGER.error("Generation failed; partial output kept at: %s", staging_root)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--write", action="store_true", help="Create the output dataset after validation.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--compression", default="zstd")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main() -> int:
    args = _parse_args()
    run_dir = _setup_logging(args.log_dir)
    LOGGER.info("Run logs: %s", run_dir)
    summary: dict[str, Any] = {
        "source_root": str(args.source_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "mode": "write" if args.write else "check",
        "result": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
    }
    try:
        source_info, episodes, source_stats = load_dataset_metadata(args.source_root)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            audits = list(executor.map(validate_episode, episodes))
        _write_jsonl(run_dir / "episode_validation.jsonl", audits)
        status_counts = dict(sorted(Counter(row["status"] for row in audits).items()))
        error_count = status_counts.get("error", 0)
        summary.update(
            {
                "episode_count": len(episodes),
                "source_task_count": len({episode.source_episode_index for episode in episodes}),
                "subtask_count": EXPECTED_SUBTASKS,
                "frame_count": sum(episode.length for episode in episodes),
                "episodes_by_subtask": dict(
                    sorted(Counter(episode.subtask_index for episode in episodes).items())
                ),
                "status_counts": status_counts,
                "error_count": error_count,
            }
        )
        LOGGER.info("Validated episodes: total=%s valid=%s errors=%s", len(audits), status_counts.get("valid", 0), error_count)
        if error_count:
            summary["result"] = "failed"
            LOGGER.error("Source validation failed; see episode_validation.jsonl")
            return 1
        if not args.write:
            summary["result"] = "source_valid"
            LOGGER.info("Check completed. Re-run with --write to create the dataset.")
            return 0
        build_dataset(
            args.source_root,
            args.output_root,
            source_info,
            episodes,
            source_stats,
            compression=args.compression,
        )
        summary["result"] = "dataset_created"
        LOGGER.info("Created breakfast progress dataset: %s", args.output_root)
        return 0
    except Exception as error:
        summary["result"] = "failed"
        summary["error"] = str(error)
        LOGGER.error("Breakfast progress generation failed: %s", error)
        return 1
    finally:
        summary["finished_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(run_dir / "summary.json", summary)


if __name__ == "__main__":
    raise SystemExit(main())
