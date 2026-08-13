"""Tests for scripts/label_completion_boundary.py on a synthetic mini dataset.

Covers: boundary labels (10 positives, subtask 4 last-10), same-image /
different-prompt contrast, parquet/index/timestamp/metadata consistency,
label_audit.json counts, and cross-group copy failure.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import label_completion_boundary as lcb

# ---------------------------------------------------------------------------
# Synthetic source dataset helpers
# ---------------------------------------------------------------------------

FRAMES_PER_EP = 15
FPS = 30.0
CHUNKS_SIZE = 1000
VIDEO_KEY = "observation.image.top"


def _make_source_video(path: pathlib.Path, num_frames: int) -> None:
    """Creates a tiny solid-color video with ``num_frames`` frames at FPS."""
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = num_frames / FPS
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=red:s=64x48:r={FPS}:d={duration}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def _make_source_parquet(path: pathlib.Path, episode_id: int, num_frames: int) -> None:
    """Writes a minimal source parquet with the columns the boundary script reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "frame_index": pa.array(np.arange(num_frames, dtype=np.int64), type=pa.int64()),
            "timestamp": pa.array(np.arange(num_frames, dtype=np.float32) / FPS, type=pa.float32()),
            "episode_index": pa.array([episode_id] * num_frames, type=pa.int64()),
            "index": pa.array(np.arange(num_frames, dtype=np.int64), type=pa.int64()),
            "task_index": pa.array([episode_id % 4] * num_frames, type=pa.int64()),
            "observation.state.joint": pa.array(np.zeros(num_frames, dtype=np.float32), type=pa.float32()),
            "actions": pa.array(np.zeros(num_frames, dtype=np.float32), type=pa.float32()),
        }
    )
    pq.write_table(table, path)


def _make_source_dataset(root: pathlib.Path, num_episodes: int = 8) -> dict[int, int]:
    """Creates a synthetic source LeRobot dataset under ``root``.

    Returns ``{episode_id: length}``.
    """
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    lengths: dict[int, int] = {}
    for eid in range(num_episodes):
        n = FRAMES_PER_EP
        _make_source_parquet(
            root / "data" / f"chunk-{eid // CHUNKS_SIZE:03d}" / f"episode_{eid:06d}.parquet",
            eid,
            n,
        )
        _make_source_video(
            root / "videos" / f"chunk-{eid // CHUNKS_SIZE:03d}" / VIDEO_KEY / f"episode_{eid:06d}.mp4",
            n,
        )
        lengths[eid] = n

    info = {
        "codebase_version": "v2.1",
        "fps": FPS,
        "chunks_size": CHUNKS_SIZE,
        "total_frames": sum(lengths.values()),
        "total_episodes": num_episodes,
        "features": {
            VIDEO_KEY: {"dtype": "video", "shape": [3, 48, 64], "names": None},
            "observation.state.joint": {"dtype": "float32", "shape": [7], "names": None},
            "actions": {"dtype": "float32", "shape": [10], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    lines = [
        json.dumps({"episode_index": eid, "length": lengths[eid], "task_index": eid % 4}) for eid in range(num_episodes)
    ]
    (meta_dir / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return lengths


def _compute_new_lengths(num_episodes: int) -> dict[int, int]:
    new_lengths: dict[int, int] = {}
    for eid in range(num_episodes):
        pos = eid % 4
        new_lengths[eid] = FRAMES_PER_EP + (0 if pos == 3 else lcb.COPY_FRAMES)
    return new_lengths


def _process_all_parquets(src_root: pathlib.Path, dst_root: pathlib.Path, num_episodes: int) -> dict[int, int]:
    """Runs ``process_boundary_parquet`` for every episode and returns new lengths."""
    new_lengths = _compute_new_lengths(num_episodes)
    global_offset = 0
    for eid in range(num_episodes):
        pos = eid % 4
        task_idx = pos
        src_path = lcb.get_episode_data_path(src_root, eid, CHUNKS_SIZE)
        dst_path = lcb.get_episode_data_path(dst_root, eid, CHUNKS_SIZE)
        next_src_path = None
        if pos < 3:
            next_src_path = lcb.get_episode_data_path(src_root, eid + 1, CHUNKS_SIZE)
        lcb.process_boundary_parquet(
            src_path,
            dst_path,
            episode_id=eid,
            group_position=pos,
            task_index=task_idx,
            next_src_path=next_src_path,
            global_offset=global_offset,
            old_length=FRAMES_PER_EP,
            fps=FPS,
            force=True,
        )
        global_offset += new_lengths[eid]
    return new_lengths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_boundary_labels_subtask1_2_3_have_10_positives_with_copy(tmp_path):
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    _make_source_dataset(src_root, num_episodes=4)
    _process_all_parquets(src_root, dst_root, num_episodes=4)

    for eid in range(3):  # subtasks 1, 2, 3
        dst_path = lcb.get_episode_data_path(dst_root, eid, CHUNKS_SIZE)
        table = pq.read_table(dst_path)
        completion = table["completion"].to_numpy()
        is_copy = table["is_boundary_copy"].to_numpy()
        assert len(completion) == FRAMES_PER_EP + lcb.COPY_FRAMES  # 15 + 5 = 20
        assert int(completion.sum()) == lcb.TOTAL_POSITIVES  # 10
        assert int(is_copy.sum()) == lcb.COPY_FRAMES  # 5
        # Last 10 frames are positive.
        assert np.all(completion[-10:] == 1)
        # First 10 frames are negative.
        assert np.all(completion[:10] == 0)
        # Copy frames are the last 5.
        assert np.all(is_copy[-5:] == 1)
        assert np.all(is_copy[:-5] == 0)


def test_boundary_labels_subtask4_has_10_positives_no_copy(tmp_path):
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    _make_source_dataset(src_root, num_episodes=4)
    _process_all_parquets(src_root, dst_root, num_episodes=4)

    eid = 3  # subtask 4
    dst_path = lcb.get_episode_data_path(dst_root, eid, CHUNKS_SIZE)
    table = pq.read_table(dst_path)
    completion = table["completion"].to_numpy()
    is_copy = table["is_boundary_copy"].to_numpy()
    assert len(completion) == FRAMES_PER_EP  # 15 (no copies)
    assert int(completion.sum()) == lcb.TOTAL_POSITIVES  # 10
    assert int(is_copy.sum()) == 0
    assert np.all(completion[-10:] == 1)
    assert np.all(completion[:5] == 0)


def test_same_image_different_prompt_contrast(tmp_path):
    """Copy frames (task=k, completion=1) and next episode's original first-5
    (task=k+1, completion=0) share the same source observation."""

    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    _make_source_dataset(src_root, num_episodes=4)
    _process_all_parquets(src_root, dst_root, num_episodes=4)

    # Episode 0 (subtask 1) copy frames: source from episode 1, frames 0..4.
    ep0 = pq.read_table(lcb.get_episode_data_path(dst_root, 0, CHUNKS_SIZE))
    copy_mask = ep0["is_boundary_copy"].to_numpy().astype(bool)
    assert np.all(ep0["source_episode_index"].to_numpy()[copy_mask] == 1)
    assert np.all(ep0["source_frame_index"].to_numpy()[copy_mask] == np.arange(5))
    assert np.all(ep0["task_index"].to_numpy()[copy_mask] == 0)  # current subtask's task
    assert np.all(ep0["completion"].to_numpy()[copy_mask] == 1)  # positive

    # Episode 1 (subtask 2) original first-5: same source (self, frames 0..4).
    ep1 = pq.read_table(lcb.get_episode_data_path(dst_root, 1, CHUNKS_SIZE))
    first5_mask = np.zeros(len(ep1), dtype=bool)
    first5_mask[:5] = True
    assert np.all(ep1["source_episode_index"].to_numpy()[first5_mask] == 1)  # self
    assert np.all(ep1["source_frame_index"].to_numpy()[first5_mask] == np.arange(5))
    assert np.all(ep1["task_index"].to_numpy()[first5_mask] == 1)  # own task
    assert np.all(ep1["completion"].to_numpy()[first5_mask] == 0)  # negative

    # Same source observation (episode 1, frames 0-4), opposite label, different prompt.
    assert np.array_equal(
        ep0["source_episode_index"].to_numpy()[copy_mask],
        ep1["source_episode_index"].to_numpy()[first5_mask],
    )
    assert np.array_equal(
        ep0["source_frame_index"].to_numpy()[copy_mask],
        ep1["source_frame_index"].to_numpy()[first5_mask],
    )


def test_parquet_index_timestamp_frame_index_contiguous(tmp_path):
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    _make_source_dataset(src_root, num_episodes=4)
    new_lengths = _process_all_parquets(src_root, dst_root, num_episodes=4)

    expected_global = 0
    for eid in range(4):
        table = pq.read_table(
            lcb.get_episode_data_path(dst_root, eid, CHUNKS_SIZE),
            columns=["index", "timestamp", "frame_index", "episode_index"],
        )
        n = new_lengths[eid]
        idx = table["index"].to_numpy()
        ts = table["timestamp"].to_numpy()
        fi = table["frame_index"].to_numpy()
        ei = table["episode_index"].to_numpy()

        assert np.array_equal(fi, np.arange(n))
        assert np.array_equal(idx, np.arange(expected_global, expected_global + n))
        assert np.allclose(ts, np.arange(n) / FPS, atol=1e-5)
        assert np.all(ei == eid)
        expected_global += n


def test_write_boundary_meta_patches_info_and_episodes(tmp_path):
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    _make_source_dataset(src_root, num_episodes=4)
    new_lengths = _process_all_parquets(src_root, dst_root, num_episodes=4)
    total_frames = sum(new_lengths.values())

    lcb.write_boundary_meta(
        src_root / "meta",
        dst_root / "meta",
        new_lengths=new_lengths,
        total_frames=total_frames,
        fps=FPS,
    )

    info = json.loads((dst_root / "meta" / "info.json").read_text())
    assert info["label_scheme"] == "boundary"
    assert info["total_frames"] == total_frames
    assert info["boundary_copy_frames"] == lcb.COPY_FRAMES
    assert info["positive_tail"] == lcb.POSITIVE_TAIL
    assert info["subtask4_tail"] == lcb.SUBTASK4_TAIL
    for col in lcb.NEW_COLUMNS:
        assert col in info["features"]

    for line in (dst_root / "meta" / "episodes.jsonl").read_text().splitlines():
        rec = json.loads(line)
        assert rec["length"] == new_lengths[rec["episode_index"]]


def test_write_boundary_meta_skips_stale_stats_files(tmp_path):
    """P1-3: stats.json, episodes_stats.jsonl, and stats/ must not be copied
    (they describe the old dataset and would be stale)."""

    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    _make_source_dataset(src_root, num_episodes=4)

    # Add stale stats files to the source meta.
    src_meta = src_root / "meta"
    (src_meta / "stats.json").write_text('{"stale": true}', encoding="utf-8")
    (src_meta / "episodes_stats.jsonl").write_text('{"stale": true}\n', encoding="utf-8")
    stats_dir = src_meta / "stats"
    stats_dir.mkdir()
    (stats_dir / "observation.state.joint.json").write_text('{"stale": true}', encoding="utf-8")

    new_lengths = _compute_new_lengths(4)
    total_frames = sum(new_lengths.values())
    lcb.write_boundary_meta(
        src_root / "meta",
        dst_root / "meta",
        new_lengths=new_lengths,
        total_frames=total_frames,
        fps=FPS,
    )

    dst_meta = dst_root / "meta"
    # Stale stats files must NOT exist in the output.
    assert not (dst_meta / "stats.json").exists()
    assert not (dst_meta / "episodes_stats.jsonl").exists()
    assert not (dst_meta / "stats").exists()
    # But episodes.jsonl and info.json should be present (patched).
    assert (dst_meta / "info.json").exists()
    assert (dst_meta / "episodes.jsonl").exists()


def test_full_audit_and_label_audit_json(tmp_path):
    """End-to-end: generate parquets + meta + audit, verify label_audit.json."""

    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    num_episodes = 8  # 2 groups
    _make_source_dataset(src_root, num_episodes=num_episodes)
    new_lengths = _process_all_parquets(src_root, dst_root, num_episodes=num_episodes)
    total_frames = sum(new_lengths.values())
    lcb.write_boundary_meta(
        src_root / "meta",
        dst_root / "meta",
        new_lengths=new_lengths,
        total_frames=total_frames,
        fps=FPS,
    )

    audit = lcb.audit_boundary_dataset(
        dst_root,
        new_lengths=new_lengths,
        fps=FPS,
        chunks_size=CHUNKS_SIZE,
        video_keys=[VIDEO_KEY],
        seed=42,
        episodes_per_group=4,
        val_groups=0,
        test_groups=1,  # 1 group = 4 episodes in test, 4 in train
        repo_id=dst_root.name,
        verify_frames=False,
    )

    # Per-subtask positives: aggregated across episodes, each episode contributes 10.
    num_groups = num_episodes // 4
    for count in audit["per_subtask_positive_counts"].values():
        assert count == num_groups * lcb.TOTAL_POSITIVES

    # Per-episode: every episode must have exactly 10 positives.
    assert audit["consistency"]["per_episode_positive_all_10"] is True

    # Total frames == pos + neg across all splits.
    total_pos = audit["train_full"]["positive"] + audit["test_full"]["positive"]
    total_neg = audit["train_full"]["negative"] + audit["test_full"]["negative"]
    assert total_pos + total_neg == total_frames

    # No val episodes.
    assert audit["val_episode_count"] == 0

    # Consistency checks all pass.
    assert audit["consistency"]["all_checks_passed"] is True
    assert audit["consistency"]["index_contiguous"] is True
    assert audit["consistency"]["timestamp_correct"] is True
    assert audit["consistency"]["metadata_consistent"] is True

    # Boundary copy positives: 3 subtasks per group x 2 groups = 6 episodes x 5 copies.
    assert audit["boundary_copy_positive_count"] == 6 * lcb.COPY_FRAMES


def test_cross_group_copy_triggers_audit_failure(tmp_path):
    """Tampering a copy frame's source_episode to an out-of-group episode must
    cause the audit to fail (SystemExit) and report the episode id + length."""

    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    num_episodes = 8
    _make_source_dataset(src_root, num_episodes=num_episodes)
    new_lengths = _process_all_parquets(src_root, dst_root, num_episodes=num_episodes)
    total_frames = sum(new_lengths.values())
    lcb.write_boundary_meta(
        src_root / "meta",
        dst_root / "meta",
        new_lengths=new_lengths,
        total_frames=total_frames,
        fps=FPS,
    )

    # Tamper: episode 0's copy frames source from episode 99 (out of group 0-3).
    path = lcb.get_episode_data_path(dst_root, 0, CHUNKS_SIZE)
    table = pq.read_table(path)
    src_ep = table["source_episode_index"].to_numpy().copy()
    src_ep[-5:] = 99
    table = table.set_column(
        table.column_names.index("source_episode_index"),
        "source_episode_index",
        pa.array(src_ep, type=pa.int64()),
    )
    pq.write_table(table, path)

    with pytest.raises(SystemExit):
        lcb.audit_boundary_dataset(
            dst_root,
            new_lengths=new_lengths,
            fps=FPS,
            chunks_size=CHUNKS_SIZE,
            video_keys=[VIDEO_KEY],
            seed=42,
            episodes_per_group=4,
            val_groups=0,
            test_groups=1,
            repo_id=dst_root.name,
            verify_frames=False,
        )


def test_video_extension_matches_parquet_rows(tmp_path):
    """Extended videos for subtasks 1/2/3 have original + 5 frames; subtask 4
    videos are copied unchanged."""

    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    num_episodes = 4
    _make_source_dataset(src_root, num_episodes=num_episodes)
    new_lengths = _process_all_parquets(src_root, dst_root, num_episodes=num_episodes)

    # Process videos.
    for eid in range(num_episodes):
        pos = eid % 4
        copy_n = 0 if pos == 3 else lcb.COPY_FRAMES
        next_eid = eid + 1 if pos < 3 else None
        lcb.process_episode_videos(
            src_root,
            dst_root,
            eid,
            next_eid,
            [VIDEO_KEY],
            CHUNKS_SIZE,
            copy_n,
            force=True,
            expected_length=new_lengths[eid],
        )

    for eid in range(num_episodes):
        vpath = lcb.get_episode_video_path(dst_root, eid, VIDEO_KEY, CHUNKS_SIZE)
        assert vpath.exists()
        nb = lcb.count_video_frames(vpath)
        assert nb == new_lengths[eid], f"episode {eid}: video {nb} != parquet {new_lengths[eid]}"
