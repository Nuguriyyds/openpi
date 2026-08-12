import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import label_progress


@pytest.mark.parametrize(
    ("frame_count", "expected"),
    [
        (1, [1.0]),
        (2, [0.0, 1.0]),
        (5, [0.0, 0.25, 0.5, 0.75, 1.0]),
    ],
)
def test_make_progress_labels(frame_count, expected):
    labels = label_progress.make_progress_labels(frame_count)

    assert labels.dtype == np.float32
    np.testing.assert_array_equal(labels, np.asarray(expected, dtype=np.float32))
    assert labels[-1] == np.float32(1.0)
    if frame_count > 1:
        assert labels[0] == np.float32(0.0)
    assert np.all(np.diff(labels) >= 0.0)


def _write_source_episode(path, episode_id, frame_count, *, frame_indices=None, task_indices=None):
    pq.write_table(
        pa.table(
            {
                "episode_index": [episode_id] * frame_count,
                "frame_index": list(range(frame_count)) if frame_indices is None else frame_indices,
                "task_index": [3] * frame_count if task_indices is None else task_indices,
                "actions": [[float(index), float(index + 1)] for index in range(frame_count)],
                "observation.state": [[float(index)] for index in range(frame_count)],
            }
        ),
        path,
    )


def test_process_parquet_preserves_action_observation_and_supports_resume_force(tmp_path):
    source = tmp_path / "episode_000007.parquet"
    destination = tmp_path / "out" / source.name
    _write_source_episode(source, 7, 5)

    first = label_progress.process_parquet(source, destination, 5, force=False, resume=False)
    output = pq.read_table(destination)

    assert first == {"skipped": 0, "written": 1}
    assert output.schema.field("progress").type == pa.float32()
    np.testing.assert_array_equal(
        output["progress"].combine_chunks().to_numpy(),
        np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32),
    )
    assert output["actions"].to_pylist() == [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
    assert output["observation.state"].to_pylist() == [[0.0], [1.0], [2.0], [3.0], [4.0]]

    resumed = label_progress.process_parquet(source, destination, 5, force=False, resume=True)
    assert resumed == {"skipped": 1, "written": 0}
    with pytest.raises(FileExistsError, match="--resume"):
        label_progress.process_parquet(source, destination, 5, force=False, resume=False)
    forced = label_progress.process_parquet(source, destination, 5, force=True, resume=False)
    assert forced == {"skipped": 0, "written": 1}


def test_validate_source_table_rejects_noncontiguous_frame_indices(tmp_path):
    source = tmp_path / "episode_000008.parquet"
    _write_source_episode(source, 8, 5, frame_indices=[0, 1, 3, 4, 5])

    with pytest.raises(ValueError, match="frame_index must be exactly"):
        label_progress.validate_source_table(pq.read_table(source), episode_id=8, expected_length=5)


def test_validate_source_table_rejects_multiple_task_indices(tmp_path):
    source = tmp_path / "episode_000009.parquet"
    _write_source_episode(source, 9, 5, task_indices=[3, 3, 4, 4, 4])

    with pytest.raises(ValueError, match="exactly one task_index"):
        label_progress.validate_source_table(pq.read_table(source), episode_id=9, expected_length=5)


def test_write_meta_adds_float32_progress_feature(tmp_path):
    src_meta = tmp_path / "src_meta"
    src_meta.mkdir()
    (src_meta / "info.json").write_text(json.dumps({"features": {"actions": {"dtype": "float32"}}}))
    (src_meta / "episodes.jsonl").write_text('{"episode_index":0,"length":1}\n')
    destination = tmp_path / "dst_meta"

    label_progress.write_meta(src_meta, destination)

    info = json.loads((destination / "info.json").read_text())
    assert info["features"]["progress"] == label_progress.PROGRESS_FEATURE
    assert (destination / "episodes.jsonl").read_text() == '{"episode_index":0,"length":1}\n'


def test_make_window_progress_labels():
    labels = label_progress.make_window_progress_labels(5, 3, 0.5)

    assert labels.dtype == np.float32
    np.testing.assert_allclose(labels, np.asarray([0.0, 0.0, 0.5, 0.75, 1.0], dtype=np.float32), atol=1e-6)


def test_check_episode_lengths_fit_window_reports_offending_episodes():
    with pytest.raises(ValueError, match=r"episode\(s\) are shorter than window_frames=10.*3:5"):
        label_progress.check_episode_lengths_fit_window({1: 20, 3: 5, 7: 15}, 10)


def test_check_episode_lengths_fit_window_accepts_when_all_long_enough():
    label_progress.check_episode_lengths_fit_window({1: 20, 3: 15, 7: 30}, 10)


def test_process_parquet_writes_window_ramp_labels(tmp_path):
    source = tmp_path / "episode_000010.parquet"
    destination = tmp_path / "out" / source.name
    _write_source_episode(source, 10, 5)

    result = label_progress.process_parquet(
        source, destination, 5, force=False, resume=False, window_frames=3, ramp_start=0.5
    )
    output = pq.read_table(destination)

    assert result == {"skipped": 0, "written": 1}
    np.testing.assert_allclose(
        output["progress"].combine_chunks().to_numpy(),
        np.asarray([0.0, 0.0, 0.5, 0.75, 1.0], dtype=np.float32),
        atol=1e-6,
    )


def test_process_parquet_resume_validates_window_ramp_labels(tmp_path):
    source = tmp_path / "episode_000011.parquet"
    destination = tmp_path / "out" / source.name
    _write_source_episode(source, 11, 5)

    label_progress.process_parquet(source, destination, 5, force=False, resume=False, window_frames=3, ramp_start=0.5)
    resumed = label_progress.process_parquet(
        source, destination, 5, force=False, resume=True, window_frames=3, ramp_start=0.5
    )

    assert resumed == {"skipped": 1, "written": 0}
