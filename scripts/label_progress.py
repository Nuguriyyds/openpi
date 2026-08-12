"""Create a separate LeRobot dataset with float32 linear ``progress`` labels.

Each source episode is treated as one breakfast subtask. Its target is
``local_frame_index / (episode_length - 1)``; a one-frame subtask is complete
at its only frame and therefore receives ``1.0``. The source dataset is only
read. All metadata, parquet, and optional videos are written below
``--output-root``.

Pass ``--window-seconds`` (together with ``--ramp-start``) to switch to a
tail-window ramp instead: every frame before the trailing
``round(window_seconds * fps)`` frames is ``0.0``, and those trailing frames
ramp linearly from ``ramp_start`` up to ``1.0``. Episodes shorter than the
window don't fully fit it and are labeled ``0.0`` everywhere instead; the
script prints a warning listing any such episodes before writing anything
(see ``--dry-run``) so you can review them, but generation proceeds
automatically.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import shutil
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from openpi.training.completion_data import make_progress_targets
from openpi.training.completion_data import make_window_progress_targets

DEFAULT_SRC_ROOT = "/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730"
LABEL_KEY = "progress"
PROGRESS_FEATURE = {"dtype": "float32", "shape": [1], "names": None}
PROGRESS_HF_FEATURE = {"dtype": "float32", "_type": "Value"}


def load_episode_lengths(meta_dir: pathlib.Path) -> dict[int, int]:
    """Reads LeRobot ``episodes.jsonl`` into ``{episode_index: length}``."""

    lengths: dict[int, int] = {}
    for line in (meta_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            lengths[int(record["episode_index"])] = int(record["length"])
    return lengths


def read_fps(meta_dir: pathlib.Path) -> float:
    """Reads the dataset-wide frames-per-second from ``info.json``."""

    info = json.loads((meta_dir / "info.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    if fps <= 0:
        raise ValueError(f"dataset fps must be positive, got {fps}")
    return fps


def warn_episodes_shorter_than_window(episode_lengths: dict[int, int], window_frames: int) -> None:
    """Prints the episodes that don't fully fit the window; they'll be labeled all-0.

    This is informational only — generation proceeds either way. An episode
    shorter than the window is exempt from the window rule for the same
    reason the legacy full-episode ramp still has an exact target even for a
    1-frame episode, just landing on all-0 here instead: the window doesn't
    fully fit inside its own timeline, and padding it with frames borrowed
    from an adjacent episode would mean labeling frames already annotated as
    the next subtask as if they still belonged to this one.
    """

    too_short = {
        episode_id: length for episode_id, length in episode_lengths.items() if length < window_frames
    }
    if too_short:
        preview = ", ".join(f"{episode_id}:{length}" for episode_id, length in sorted(too_short.items())[:10])
        remaining = len(too_short) - 10
        suffix = f", and {remaining} more" if remaining > 0 else ""
        print(
            f"  {len(too_short)} episode(s) are shorter than window_frames={window_frames} and will be "
            f"labeled all-0.0: {preview}{suffix}"
        )


def make_progress_labels(num_frames: int) -> np.ndarray:
    """Returns the exact float32 labels required for one subtask episode."""

    return make_progress_targets(num_frames)


def make_window_progress_labels(num_frames: int, window_frames: int, ramp_start: float) -> np.ndarray:
    """Returns the exact float32 tail-window ramp labels for one subtask episode."""

    return make_window_progress_targets(num_frames, window_frames, ramp_start)


def _scalar_array(column: pa.ChunkedArray, *, name: str, episode_id: int) -> np.ndarray:
    values: list[Any] = []
    for frame_index, raw_value in enumerate(column.combine_chunks().to_pylist()):
        scalar = raw_value
        if isinstance(scalar, list | tuple):
            if len(scalar) != 1:
                raise ValueError(
                    f"episode {episode_id} field {name!r} is not scalar at frame {frame_index}: {scalar!r}"
                )
            scalar = scalar[0]
        values.append(scalar)
    return np.asarray(values)


def _validate_integer_values(values: np.ndarray, *, field: str, episode_id: int) -> np.ndarray:
    try:
        integers = values.astype(np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"episode {episode_id} has non-integer {field} values") from error
    if not np.array_equal(values, integers):
        raise ValueError(f"episode {episode_id} has non-integer {field} values: {values.tolist()}")
    return integers


def validate_source_table(table: pa.Table, *, episode_id: int, expected_length: int) -> int:
    """Fails before writing if one source file is not one contiguous subtask."""

    required = {"episode_index", "frame_index", "task_index"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"episode {episode_id} is missing required field(s): {missing}")
    if table.num_rows != expected_length:
        raise ValueError(
            f"episode {episode_id} has {table.num_rows} parquet rows but metadata length is {expected_length}"
        )
    if expected_length <= 0:
        raise ValueError(f"episode {episode_id} has non-positive metadata length {expected_length}")

    episode_values = _scalar_array(table["episode_index"], name="episode_index", episode_id=episode_id)
    frame_values = _scalar_array(table["frame_index"], name="frame_index", episode_id=episode_id)
    task_values = _scalar_array(table["task_index"], name="task_index", episode_id=episode_id)
    if not np.all(episode_values == episode_id):
        bad_rows = np.flatnonzero(episode_values != episode_id).tolist()
        raise ValueError(f"episode {episode_id} contains mismatched episode_index at rows {bad_rows}")
    integer_frames = _validate_integer_values(frame_values, field="frame_index", episode_id=episode_id)
    expected_frames = np.arange(expected_length, dtype=np.int64)
    if not np.array_equal(integer_frames, expected_frames):
        raise ValueError(
            f"episode {episode_id} frame_index must be exactly 0..{expected_length - 1}, got {integer_frames.tolist()}"
        )
    integer_tasks = _validate_integer_values(task_values, field="task_index", episode_id=episode_id)
    unique_tasks = np.unique(integer_tasks)
    if len(unique_tasks) != 1:
        raise ValueError(f"episode {episode_id} must contain exactly one task_index, got {unique_tasks.tolist()}")
    return int(unique_tasks[0])


def _validate_existing_progress_table(
    table: pa.Table,
    *,
    episode_id: int,
    expected_length: int,
    window_frames: int | None = None,
    ramp_start: float | None = None,
) -> None:
    validate_source_table(table, episode_id=episode_id, expected_length=expected_length)
    if LABEL_KEY not in table.column_names:
        raise ValueError(f"episode {episode_id} destination is missing {LABEL_KEY!r}")
    field = table.schema.field(LABEL_KEY)
    if not pa.types.is_float32(field.type):
        raise ValueError(f"episode {episode_id} destination {LABEL_KEY!r} must be float32, got {field.type}")
    labels = _scalar_array(table[LABEL_KEY], name=LABEL_KEY, episode_id=episode_id).astype(np.float32)
    if window_frames is not None:
        assert ramp_start is not None
        expected = make_window_progress_labels(expected_length, window_frames, ramp_start)
        description = "tail-window ramp target"
    else:
        expected = make_progress_labels(expected_length)
        description = "linear progress target"
    if not np.array_equal(labels, expected):
        raise ValueError(f"episode {episode_id} destination {LABEL_KEY!r} is not the expected {description}")


def update_hf_schema_metadata(table: pa.Table) -> pa.Table:
    """Updates only the Hugging Face feature metadata for the new field."""

    schema_metadata = table.schema.metadata
    if schema_metadata is None:
        hf_meta: dict[str, Any] = {"info": {"features": {LABEL_KEY: dict(PROGRESS_HF_FEATURE)}}}
    else:
        existing = {key.decode() if isinstance(key, bytes) else key: value for key, value in schema_metadata.items()}
        raw_hf_meta = existing.get("huggingface", b"{}")
        if isinstance(raw_hf_meta, bytes):
            raw_hf_meta = raw_hf_meta.decode()
        hf_meta = json.loads(raw_hf_meta)
        hf_meta.setdefault("info", {}).setdefault("features", {})[LABEL_KEY] = dict(PROGRESS_HF_FEATURE)
    new_metadata = {b"huggingface": json.dumps(hf_meta).encode()}
    if schema_metadata:
        for key, value in schema_metadata.items():
            key_as_text = key.decode() if isinstance(key, bytes) else key
            if key_as_text != "huggingface":
                new_metadata[key if isinstance(key, bytes) else key.encode()] = value
    return table.replace_schema_metadata(new_metadata)


def process_parquet(
    src_path: pathlib.Path,
    dst_path: pathlib.Path,
    expected_length: int,
    *,
    force: bool,
    resume: bool,
    window_frames: int | None = None,
    ramp_start: float | None = None,
) -> dict[str, int]:
    """Copies one parquet unchanged except for a verified float32 progress field."""

    source_table = pq.read_table(src_path)
    episode_id = int(src_path.stem.split("_")[1])
    validate_source_table(source_table, episode_id=episode_id, expected_length=expected_length)

    if dst_path.exists() and not force:
        destination_table = pq.read_table(dst_path)
        if LABEL_KEY in destination_table.column_names:
            if not resume:
                raise FileExistsError(
                    f"destination already contains {LABEL_KEY!r}: {dst_path}; use --resume to validate/skip or --force"
                )
            _validate_existing_progress_table(
                destination_table,
                episode_id=episode_id,
                expected_length=expected_length,
                window_frames=window_frames,
                ramp_start=ramp_start,
            )
            return {"skipped": 1, "written": 0}

    if window_frames is not None:
        assert ramp_start is not None
        labels = pa.array(make_window_progress_labels(expected_length, window_frames, ramp_start), type=pa.float32())
    else:
        labels = pa.array(make_progress_labels(expected_length), type=pa.float32())
    if LABEL_KEY in source_table.column_names:
        column_index = source_table.schema.get_field_index(LABEL_KEY)
        output_table = source_table.set_column(column_index, LABEL_KEY, labels)
    else:
        output_table = source_table.append_column(LABEL_KEY, labels)
    output_table = update_hf_schema_metadata(output_table)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = dst_path.with_suffix(".parquet.tmp")
    pq.write_table(output_table, temporary_path)
    os.replace(temporary_path, dst_path)
    return {"skipped": 0, "written": 1}


def _process_parquet_worker(args: tuple[str, str, int, bool, bool, int | None, float | None]) -> dict[str, Any]:
    src, dst, expected_length, force, resume, window_frames, ramp_start = args
    try:
        return process_parquet(
            pathlib.Path(src),
            pathlib.Path(dst),
            expected_length,
            force=force,
            resume=resume,
            window_frames=window_frames,
            ramp_start=ramp_start,
        )
    except Exception as error:
        return {"error": str(error), "src": src}


def collect_parquet_jobs(
    src_root: pathlib.Path,
    dst_root: pathlib.Path,
    episode_lengths: dict[int, int],
) -> list[tuple[pathlib.Path, pathlib.Path, int]]:
    """Finds every source episode parquet and its non-overlapping destination."""

    jobs: list[tuple[pathlib.Path, pathlib.Path, int]] = []
    for chunk_dir in sorted((src_root / "data").iterdir()):
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk-"):
            continue
        for src_path in sorted(chunk_dir.glob("episode_*.parquet")):
            episode_id = int(src_path.stem.split("_")[1])
            if episode_id not in episode_lengths:
                raise ValueError(f"episode {episode_id} is absent from meta/episodes.jsonl ({src_path})")
            jobs.append((src_path, dst_root / src_path.relative_to(src_root), episode_lengths[episode_id]))
    if not jobs:
        raise FileNotFoundError(f"no episode parquet files found below {src_root / 'data'}")
    return jobs


def validate_source_jobs(jobs: list[tuple[pathlib.Path, pathlib.Path, int]]) -> None:
    """Audits all source episodes before the script writes any output files."""

    for src_path, _, expected_length in tqdm.tqdm(jobs, desc="Validating source", unit="episode"):
        table = pq.read_table(src_path)
        validate_source_table(table, episode_id=int(src_path.stem.split("_")[1]), expected_length=expected_length)


def write_meta(
    src_meta: pathlib.Path,
    dst_meta: pathlib.Path,
    *,
    window_seconds: float | None = None,
    ramp_start: float | None = None,
) -> None:
    """Copies metadata and atomically adds the progress feature to info.json."""

    dst_meta.mkdir(parents=True, exist_ok=True)
    for item in src_meta.iterdir():
        if item.name == "info.json":
            continue
        destination = dst_meta / item.name
        if item.is_file():
            shutil.copy2(item, destination)
        else:
            shutil.copytree(item, destination, dirs_exist_ok=True)
    info = json.loads((src_meta / "info.json").read_text(encoding="utf-8"))
    info.setdefault("features", {})[LABEL_KEY] = dict(PROGRESS_FEATURE)
    if window_seconds is not None:
        info["window_seconds"] = window_seconds
        info["ramp_start"] = ramp_start
    else:
        info.pop("window_seconds", None)
        info.pop("ramp_start", None)
    info_path = dst_meta / "info.json"
    temporary_path = info_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary_path, info_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--src-root", type=pathlib.Path, default=pathlib.Path(DEFAULT_SRC_ROOT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="Validate and skip already-complete destination parquets."
    )
    parser.add_argument("--force", action="store_true", help="Rewrite destination parquets even when progress exists.")
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help=(
            "Switch to a tail-window ramp: frames before the trailing round(window_seconds * fps) "
            "frames are 0.0, and those trailing frames ramp from --ramp-start to 1.0. Requires "
            "--ramp-start. Unset preserves the legacy full-episode linear ramp."
        ),
    )
    parser.add_argument(
        "--ramp-start",
        type=float,
        default=None,
        help="Value the tail-window ramp starts at (must be in [0, 1)). Requires --window-seconds.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")
    if (args.window_seconds is None) != (args.ramp_start is None):
        parser.error("--window-seconds and --ramp-start must be set together")
    if args.window_seconds is not None and args.window_seconds <= 0:
        parser.error("--window-seconds must be positive")
    if args.ramp_start is not None and not 0.0 <= args.ramp_start < 1.0:
        parser.error("--ramp-start must be in [0, 1)")
    return args


def main() -> int:
    args = _parse_args()
    src_root = args.src_root.resolve()
    dst_root = args.output_root.resolve()
    if src_root == dst_root:
        raise ValueError("--output-root must differ from --src-root; the source dataset is never modified")
    if not (src_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"source dataset metadata not found: {src_root / 'meta' / 'info.json'}")

    episode_lengths = load_episode_lengths(src_root / "meta")
    jobs = collect_parquet_jobs(src_root, dst_root, episode_lengths)
    print(f"Source dataset : {src_root}")
    print(f"Output dataset : {dst_root}")
    print(f"Episodes       : {len(episode_lengths)}")
    print(f"Parquet files  : {len(jobs)}")

    window_frames: int | None = None
    if args.window_seconds is not None:
        fps = read_fps(src_root / "meta")
        window_frames = round(args.window_seconds * fps)
        if window_frames <= 0:
            raise ValueError(f"--window-seconds={args.window_seconds} at fps={fps} rounds to a non-positive window")
        lengths = sorted(episode_lengths.values())
        print(
            f"Window         : {args.window_seconds}s @ {fps} fps = {window_frames} frames, "
            f"ramp_start={args.ramp_start} "
            f"(episode lengths: min={lengths[0]}, median={lengths[len(lengths) // 2]}, max={lengths[-1]})"
        )
        # Informational: episodes shorter than the window will be labeled
        # all-0 (see module docstring); nothing here blocks generation.
        warn_episodes_shorter_than_window(episode_lengths, window_frames)

    print("Validating every source episode (frame_index and task_index) ...")
    validate_source_jobs(jobs)

    if args.dry_run:
        print("[DRY RUN] Source audit passed; no output files were written.")
        return 0

    write_meta(src_root / "meta", dst_root / "meta", window_seconds=args.window_seconds, ramp_start=args.ramp_start)
    if not args.skip_videos and (src_root / "videos").exists():
        shutil.copytree(src_root / "videos", dst_root / "videos", dirs_exist_ok=True)

    work_items = [
        (str(src), str(dst), length, args.force, args.resume, window_frames, args.ramp_start)
        for src, dst, length in jobs
    ]
    written = 0
    skipped = 0
    errors: list[dict[str, Any]] = []
    if args.workers == 1:
        results = map(_process_parquet_worker, work_items)
        for result in tqdm.tqdm(results, total=len(work_items), desc="Writing progress", unit="episode"):
            written += int(result.get("written", 0))
            skipped += int(result.get("skipped", 0))
            if "error" in result:
                errors.append(result)
    else:
        with mp.Pool(args.workers) as pool:
            for result in tqdm.tqdm(
                pool.imap_unordered(_process_parquet_worker, work_items),
                total=len(work_items),
                desc="Writing progress",
                unit="episode",
            ):
                written += int(result.get("written", 0))
                skipped += int(result.get("skipped", 0))
                if "error" in result:
                    errors.append(result)
    if errors:
        for error in errors[:10]:
            print(f"ERROR {error.get('src', '?')}: {error['error']}", file=sys.stderr)
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} additional errors", file=sys.stderr)
        return 1

    print(f"Done. Wrote {written} parquet file(s), skipped {skipped}; output: {dst_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
