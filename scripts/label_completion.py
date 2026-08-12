"""Generate a labeled copy of a LeRobot dataset with a ``completion`` column.

Each episode's final 2 frames are labeled ``1.0``; every other frame is ``0.0``.
Episodes shorter than 2 frames are labeled ``0.0`` everywhere (they cannot
satisfy the last-two-frames rule and are exempt from that part of the audit).

Pass ``--window-seconds`` to switch to a tail-window scheme instead: the last
``round(window_seconds * fps)`` frames of each episode are labeled ``1.0``,
everything before is ``0.0``. This is a wider version of the same last-N-frames
idea, meant to give the completion head a denser positive region to learn
from. Episodes shorter than the window don't fully fit it and are labeled
``0.0`` everywhere, same as the legacy 2-frame rule's exemption; the script
prints a warning listing any such episodes before writing anything (see
``--dry-run``) so you can review them, but generation proceeds automatically.

The original dataset is never modified — all output is written to
``--output-root``.  The new dataset preserves the full LeRobot v2.1 layout
(``meta/``, ``data/``, ``videos/``) so it can be used as a drop-in replacement
by pointing ``lerobot_home`` / ``repo_id`` at it.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import shutil
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from openpi.training.completion_data import make_window_completion_targets

DEFAULT_SRC_ROOT = "/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730"
COMPLETION_FEATURE = {"dtype": "float32", "shape": [1], "names": None}
COMPLETION_HF_FEATURE = {"dtype": "float32", "_type": "Value"}
LABEL_KEY = "completion"


def load_episode_lengths(meta_dir: pathlib.Path) -> dict[int, int]:
    """Reads ``episodes.jsonl`` and returns ``{episode_index: length}``."""

    lengths: dict[int, int] = {}
    episodes_path = meta_dir / "episodes.jsonl"
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
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
    reason a <2-frame episode is exempt from the legacy last-2-frames rule:
    the window doesn't fully fit inside its own timeline, and padding it with
    frames borrowed from an adjacent episode would mean labeling frames
    already annotated as the next subtask as if they still belonged to
    this one.
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


def make_completion_labels(num_frames: int) -> np.ndarray:
    """Last-2-frames = 1.0, rest = 0.0; episodes with < 2 frames are all 0.0."""

    labels = np.zeros(num_frames, dtype=np.float32)
    if num_frames >= 2:
        labels[-2:] = 1.0
    return labels


def update_hf_schema_metadata(table: pa.Table) -> pa.Table:
    """Injects the ``completion`` field into the parquet ``huggingface`` metadata."""

    schema_metadata = table.schema.metadata
    if schema_metadata is None:
        # No existing metadata — build a minimal huggingface schema entry.
        hf_meta: dict = {"info": {"features": {LABEL_KEY: dict(COMPLETION_HF_FEATURE)}}}
    else:
        existing = {
            k.decode() if isinstance(k, bytes) else k: v
            for k, v in schema_metadata.items()
        }
        raw = existing.get("huggingface", b"{}")
        if isinstance(raw, bytes):
            raw = raw.decode()
        hf_meta = json.loads(raw)
        features = hf_meta.setdefault("info", {}).setdefault("features", {})
        features[LABEL_KEY] = dict(COMPLETION_HF_FEATURE)
    new_metadata = {b"huggingface": json.dumps(hf_meta).encode()}
    if schema_metadata:
        for key, value in schema_metadata.items():
            key_str = key.decode() if isinstance(key, bytes) else key
            if key_str != "huggingface":
                new_metadata[key if isinstance(key, bytes) else key.encode()] = value
    return table.replace_schema_metadata(new_metadata)


def process_parquet(
    src_path: pathlib.Path,
    dst_path: pathlib.Path,
    expected_length: int,
    *,
    force: bool,
    window_frames: int | None = None,
) -> dict[str, int]:
    """Reads a source parquet, appends ``completion``, writes to ``dst_path``."""

    table = pq.read_table(src_path)
    num_rows = table.num_rows
    if num_rows != expected_length:
        raise ValueError(
            f"{src_path.name}: parquet has {num_rows} rows but episodes.jsonl length is {expected_length}"
        )

    # Idempotency: skip if the output already has the column (unless --force).
    if dst_path.exists():
        existing = pq.ParquetFile(dst_path)
        if LABEL_KEY in existing.schema_arrow.names and not force:
            return {"skipped": 1, "written": 0}

    if window_frames is not None:
        labels = make_window_completion_targets(num_rows, window_frames)
    else:
        labels = make_completion_labels(num_rows)
    completion_col = pa.array(labels, type=pa.float32())
    table = table.append_column(LABEL_KEY, completion_col)
    table = update_hf_schema_metadata(table)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp_path)
    os.replace(tmp_path, dst_path)
    return {"skipped": 0, "written": 1}


def _process_parquet_worker(args: tuple) -> dict[str, int]:
    """Wrapper for ``multiprocessing``."""

    src_path, dst_path, expected_length, force, window_frames = args
    try:
        return process_parquet(
            pathlib.Path(src_path),
            pathlib.Path(dst_path),
            expected_length,
            force=force,
            window_frames=window_frames,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "src": str(src_path)}


def collect_parquet_jobs(
    src_root: pathlib.Path,
    dst_root: pathlib.Path,
    episode_lengths: dict[int, int],
) -> list[tuple[pathlib.Path, pathlib.Path, int]]:
    """Builds (src, dst, expected_length) tuples for every episode parquet."""

    data_dir = src_root / "data"
    jobs: list[tuple[pathlib.Path, pathlib.Path, int]] = []
    for chunk_dir in sorted(data_dir.iterdir()):
        if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk-"):
            continue
        for parquet_path in sorted(chunk_dir.glob("episode_*.parquet")):
            # Extract episode index from filename: episode_000123.parquet -> 123
            episode_id = int(parquet_path.stem.split("_")[1])
            if episode_id not in episode_lengths:
                raise ValueError(f"episode {episode_id} not found in episodes.jsonl ({parquet_path})")
            relative = parquet_path.relative_to(src_root)
            dst_path = dst_root / relative
            jobs.append((parquet_path, dst_path, episode_lengths[episode_id]))
    return jobs


def write_meta(src_meta: pathlib.Path, dst_meta: pathlib.Path, *, window_seconds: float | None = None) -> None:
    """Copies ``meta/`` and patches ``info.json`` with the ``completion`` feature."""

    dst_meta.mkdir(parents=True, exist_ok=True)
    # Copy everything except info.json (which we patch) verbatim.
    for item in src_meta.iterdir():
        if item.name == "info.json":
            continue
        dst_item = dst_meta / item.name
        if item.is_file():
            shutil.copy2(item, dst_item)
        else:
            shutil.copytree(item, dst_item, dirs_exist_ok=True)

    info = json.loads((src_meta / "info.json").read_text(encoding="utf-8"))
    features = info.setdefault("features", {})
    if LABEL_KEY not in features:
        features[LABEL_KEY] = dict(COMPLETION_FEATURE)
    else:
        features[LABEL_KEY] = dict(COMPLETION_FEATURE)  # overwrite to ensure consistency
    if window_seconds is not None:
        info["window_seconds"] = window_seconds
    else:
        info.pop("window_seconds", None)

    info_path = dst_meta / "info.json"
    tmp_path = info_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_path, info_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a labeled dataset copy with a 'completion' column."
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        required=True,
        help="Output dataset root directory (must not be the same as --src-root).",
    )
    parser.add_argument(
        "--src-root",
        type=pathlib.Path,
        default=pathlib.Path(DEFAULT_SRC_ROOT),
        help="Source LeRobot dataset root directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and report planned actions without writing anything.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process parquet files even if the output already has the completion column.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes for parquet processing.",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Skip copying the videos/ directory (use if videos are already present).",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help=(
            "Switch to a tail-window scheme: the last round(window_seconds * fps) frames of each "
            "episode are labeled 1.0, everything before is 0.0. Unset preserves the legacy "
            "last-2-frames scheme."
        ),
    )
    args = parser.parse_args()

    if args.window_seconds is not None and args.window_seconds <= 0:
        parser.error("--window-seconds must be positive")

    src_root: pathlib.Path = args.src_root.resolve()
    dst_root: pathlib.Path = args.output_root.resolve()

    if src_root == dst_root:
        parser.error("--output-root must not be the same as --src-root (original dataset must not be modified).")

    if not (src_root / "meta" / "info.json").is_file():
        parser.error(f"source dataset not found: {src_root / 'meta' / 'info.json'}")

    print(f"Source dataset : {src_root}")
    print(f"Output dataset : {dst_root}")

    # --- Load episode metadata ---
    episode_lengths = load_episode_lengths(src_root / "meta")
    print(f"Episodes       : {len(episode_lengths)}")

    window_frames: int | None = None
    if args.window_seconds is not None:
        fps = read_fps(src_root / "meta")
        window_frames = round(args.window_seconds * fps)
        if window_frames <= 0:
            parser.error(f"--window-seconds={args.window_seconds} at fps={fps} rounds to a non-positive window")
        lengths = sorted(episode_lengths.values())
        print(
            f"Window         : {args.window_seconds}s @ {fps} fps = {window_frames} frames "
            f"(episode lengths: min={lengths[0]}, median={lengths[len(lengths) // 2]}, max={lengths[-1]})"
        )
        # Informational: episodes shorter than the window will be labeled
        # all-0 (see module docstring); nothing here blocks generation.
        warn_episodes_shorter_than_window(episode_lengths, window_frames)
    else:
        short_episodes = {eid: length for eid, length in episode_lengths.items() if length < 2}
        if short_episodes:
            print(f"  Episodes with < 2 frames (labeled all-0): {short_episodes}")

    jobs = collect_parquet_jobs(src_root, dst_root, episode_lengths)
    print(f"Parquet files  : {len(jobs)}")

    if args.dry_run:
        print("\n[DRY RUN] No files will be written.")
        print(f"  Would copy meta/ and patch info.json (add '{LABEL_KEY}' feature)")
        if not args.skip_videos:
            videos_dir = src_root / "videos"
            if videos_dir.exists():
                print(f"  Would copy videos/ ({sum(1 for _ in videos_dir.rglob('*') if _.is_file())} files)")
        print(f"  Would process {len(jobs)} parquet files (add '{LABEL_KEY}' column)")
        # Quick validation on a few files.
        for src_path, _, expected_length in jobs[:3]:
            table = pq.read_table(src_path, columns=["frame_index"])
            actual = table.num_rows
            status = "OK" if actual == expected_length else "MISMATCH"
            print(f"    {src_path.name}: {actual} rows vs {expected_length} expected [{status}]")
        return

    # --- 1. Write meta/ ---
    print("\n[1/3] Writing meta/ ...")
    write_meta(src_root / "meta", dst_root / "meta", window_seconds=args.window_seconds)
    print(f"  Patched info.json — added '{LABEL_KEY}' to features" + (
        f", window_seconds={args.window_seconds}" if args.window_seconds is not None else ""
    ))

    # --- 2. Copy videos/ ---
    if not args.skip_videos:
        src_videos = src_root / "videos"
        if src_videos.exists():
            print("\n[2/3] Copying videos/ ...")
            shutil.copytree(src_videos, dst_root / "videos", dirs_exist_ok=True)
            print("  Done.")
        else:
            print("\n[2/3] videos/ not found in source — skipping.")
    else:
        print("\n[2/3] Skipping videos/ (--skip-videos)")

    # --- 3. Process parquet files ---
    print(f"\n[3/3] Processing {len(jobs)} parquet files with {args.workers} worker(s) ...")
    total_written = 0
    total_skipped = 0
    errors: list[dict] = []

    work_items = [(str(src), str(dst), length, args.force, window_frames) for src, dst, length in jobs]

    if args.workers <= 1:
        for item in tqdm.tqdm(work_items, desc="Labeling"):
            result = _process_parquet_worker(item)
            total_written += result.get("written", 0)
            total_skipped += result.get("skipped", 0)
            if "error" in result:
                errors.append(result)
    else:
        with mp.Pool(args.workers) as pool:
            for result in tqdm.tqdm(
                pool.imap_unordered(_process_parquet_worker, work_items),
                total=len(work_items),
                desc="Labeling",
            ):
                total_written += result.get("written", 0)
                total_skipped += result.get("skipped", 0)
                if "error" in result:
                    errors.append(result)

    print(f"\n  Written : {total_written}")
    print(f"  Skipped : {total_skipped}")
    if errors:
        print(f"  Errors  : {len(errors)}")
        for err in errors[:10]:
            print(f"    {err.get('src', '?')}: {err.get('error', '?')}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
        sys.exit(1)

    print(f"\nDone. Labeled dataset at: {dst_root}")
    print(f"  To use it for S2 training, set lerobot_home to: {dst_root.parent}")
    print(f"  and repo_id to: {dst_root.name}")


if __name__ == "__main__":
    main()
