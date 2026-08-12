import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.training import completion_data


def _write_episode(path, episode_id: int, labels, *, include_completion: bool = True, frame_indices=None):
    columns = {
        "episode_index": [episode_id] * len(labels),
        "frame_index": list(range(len(labels))) if frame_indices is None else frame_indices,
    }
    if include_completion:
        columns["completion"] = labels
    pq.write_table(pa.table(columns), path)


def _write_progress_episode(path, episode_id: int, labels, *, frame_indices=None, task_indices=None):
    frame_count = len(labels)
    pq.write_table(
        pa.table(
            {
                "episode_index": [episode_id] * frame_count,
                "frame_index": list(range(frame_count)) if frame_indices is None else frame_indices,
                "task_index": [17] * frame_count if task_indices is None else task_indices,
                "progress": pa.array(np.asarray(labels, dtype=np.float32), type=pa.float32()),
            }
        ),
        path,
    )


def test_four_episode_grouping_is_fixed_and_has_no_leakage(tmp_path):
    episode_ids = list(range(48))
    path = tmp_path / "split.json"
    first = completion_data.load_or_create_split_manifest(
        path,
        episode_ids,
        repo_id="org/breakfast",
        seed=42,
    )
    second = completion_data.load_or_create_split_manifest(
        path,
        episode_ids,
        repo_id="org/breakfast",
        seed=42,
    )

    assert first == second
    assert json.loads(path.read_text())["seed"] == 42
    assert len(first.splits["val"]) == 5
    assert len(first.splits["test"]) == 5
    assert len(first.splits["train"]) == 2
    split_sets = {split: set(first.episode_ids(split)) for split in completion_data.SPLIT_NAMES}
    assert split_sets["train"].isdisjoint(split_sets["val"])
    assert split_sets["train"].isdisjoint(split_sets["test"])
    assert split_sets["val"].isdisjoint(split_sets["test"])
    assert set.union(*split_sets.values()) == set(episode_ids)
    assert all(len(group.episode_ids) == 4 for groups in first.splits.values() for group in groups)


@pytest.mark.parametrize(
    ("episode_ids", "message"),
    [
        (list(range(45)), "not divisible by 4"),
        (list(range(40)), "at least 11 complete groups"),
        ([*range(7), *range(8, 45)], "must be continuous"),
    ],
)
def test_invalid_episode_layout_has_clear_error(episode_ids, message):
    with pytest.raises(ValueError, match=message):
        completion_data.create_split_manifest(episode_ids, repo_id="org/breakfast")


def test_split_manifest_requires_validation_groups():
    with pytest.raises(ValueError, match="val_groups must be positive"):
        completion_data.create_split_manifest(
            range(24),
            repo_id="org/breakfast",
            val_groups=0,
            test_groups=5,
        )


def test_episode_audit_accepts_exactly_last_two_positive(tmp_path):
    path = tmp_path / "episode_000017.parquet"
    _write_episode(path, 17, [0, 0, 0, 1, 1])

    audit = completion_data.audit_episode_parquet(path, episode_id=17, expected_length=5)

    assert audit.positive_count == 2
    assert audit.negative_count == 3


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ([0, -1, 0, 1, 1], "outside 0/1"),
        ([0, 1, 0, 1, 1], "before its last 2 frames"),
        ([0, 0, 0, 0, 1], "exactly its last 2 frames"),
    ],
)
def test_episode_audit_reports_specific_episode_for_bad_labels(tmp_path, labels, message):
    path = tmp_path / "episode_000023.parquet"
    _write_episode(path, 23, labels)

    with pytest.raises(ValueError, match=rf"episode 23.*{message}"):
        completion_data.audit_episode_parquet(path, episode_id=23, expected_length=len(labels))


def test_episode_audit_accepts_single_frame_all_zero(tmp_path):
    """Episodes shorter than 2 frames are exempt from the last-two rule and must be all-0."""

    path = tmp_path / "episode_002797.parquet"
    _write_episode(path, 2797, [0])

    audit = completion_data.audit_episode_parquet(path, episode_id=2797, expected_length=1)

    assert audit.frame_count == 1
    assert audit.positive_count == 0
    assert audit.negative_count == 1


def test_episode_audit_rejects_positive_label_on_short_episode(tmp_path):
    """A single-frame episode with label 1 must fail the audit."""

    path = tmp_path / "episode_000001.parquet"
    _write_episode(path, 1, [1])

    with pytest.raises(ValueError, match=r"episode 1.*all labels must be 0"):
        completion_data.audit_episode_parquet(path, episode_id=1, expected_length=1)


def test_episode_audit_reports_missing_completion_field(tmp_path):
    path = tmp_path / "episode_000009.parquet"
    _write_episode(path, 9, [0, 0, 1, 1], include_completion=False)

    with pytest.raises(ValueError, match=r"episode 9.*completion"):
        completion_data.audit_episode_parquet(path, episode_id=9, expected_length=4)


def test_episode_audit_rejects_fractional_frame_indices(tmp_path):
    path = tmp_path / "episode_000031.parquet"
    _write_episode(path, 31, [0, 0, 1, 1], frame_indices=[0.5, 1.5, 2.5, 3.5])

    with pytest.raises(ValueError, match=r"episode 31.*non-integer frame_index"):
        completion_data.audit_episode_parquet(path, episode_id=31, expected_length=4)


@pytest.mark.parametrize(
    ("frame_count", "expected"),
    [
        (1, [1.0]),
        (2, [0.0, 1.0]),
        (5, [0.0, 0.25, 0.5, 0.75, 1.0]),
    ],
)
def test_progress_labels_have_required_linear_endpoints_and_monotonicity(frame_count, expected):
    labels = completion_data.make_progress_targets(frame_count)

    assert labels.dtype == np.float32
    np.testing.assert_array_equal(labels, np.asarray(expected, dtype=np.float32))
    assert labels[-1] == np.float32(1.0)
    if frame_count > 1:
        assert labels[0] == np.float32(0.0)
    assert np.all(np.diff(labels) >= 0.0)


def test_progress_episode_audit_accepts_exact_float32_linear_targets(tmp_path):
    path = tmp_path / "episode_000019.parquet"
    _write_progress_episode(path, 19, completion_data.make_progress_targets(5))

    audit = completion_data.audit_progress_episode_parquet(path, episode_id=19, expected_length=5)

    assert audit.frame_count == 5
    assert audit.task_index == 17
    assert audit.positive_count == audit.negative_count == 0


def test_progress_episode_audit_rejects_noncontiguous_frame_indices(tmp_path):
    path = tmp_path / "episode_000020.parquet"
    _write_progress_episode(
        path,
        20,
        completion_data.make_progress_targets(5),
        frame_indices=[0, 1, 3, 4, 5],
    )

    with pytest.raises(ValueError, match=r"episode 20.*frame_index must be exactly"):
        completion_data.audit_progress_episode_parquet(path, episode_id=20, expected_length=5)


def test_progress_episode_audit_rejects_multiple_task_indices(tmp_path):
    path = tmp_path / "episode_000021.parquet"
    _write_progress_episode(
        path,
        21,
        completion_data.make_progress_targets(5),
        task_indices=[4, 4, 5, 5, 5],
    )

    with pytest.raises(ValueError, match=r"episode 21.*exactly one task_index"):
        completion_data.audit_progress_episode_parquet(path, episode_id=21, expected_length=5)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ([0.1, 0.25, 0.5, 0.75, 1.0], "first frame"),
        ([0.0, 0.25, 0.5, 0.75, 0.9], "last frame"),
        ([0.0, 0.5, 0.4, 0.75, 1.0], "monotonic"),
    ],
)
def test_progress_episode_audit_enforces_endpoints_and_monotonicity(tmp_path, labels, message):
    path = tmp_path / "episode_000022.parquet"
    _write_progress_episode(path, 22, labels)

    with pytest.raises(ValueError, match=message):
        completion_data.audit_progress_episode_parquet(path, episode_id=22, expected_length=5)


def test_prepare_completion_data_requires_fully_mounted_parquet_files(tmp_path):
    class Metadata:
        def __init__(self):
            self.features = {"completion": {}}
            self.episodes = {0: {"length": 4}}

        @staticmethod
        def get_data_file_path(episode_id):
            return f"data/chunk-000/episode_{episode_id:06d}.parquet"

    manifest_path = tmp_path / "split.json"
    with pytest.raises(FileNotFoundError, match="fully mounted local LeRobot dataset") as error:
        completion_data.prepare_completion_data(
            Metadata(),
            repo_id="org/breakfast",
            dataset_root=tmp_path,
            label_key="completion",
            manifest_path=manifest_path,
        )

    assert str(tmp_path) in str(error.value)
    assert "episode_000000.parquet" in str(error.value)
    assert not manifest_path.exists()


def test_prepare_split_only_does_not_require_completion_labels_or_parquet_files(tmp_path):
    class Metadata:
        def __init__(self):
            self.features = {}
            self.episodes = {episode_id: {"length": 4} for episode_id in range(44)}

        @staticmethod
        def get_data_file_path(episode_id):
            return f"data/chunk-000/episode_{episode_id:06d}.parquet"

    info = completion_data.prepare_completion_data(
        Metadata(),
        repo_id="org/breakfast",
        dataset_root=tmp_path,
        label_key="completion",
        manifest_path=tmp_path / "split.json",
        audit_labels=False,
    )

    assert info.manifest.episode_ids("train")
    assert info.episode_audits == {}
    assert info.train_positive_count is None
    assert info.train_negative_count is None
    assert info.pos_weight is None


def test_derived_label_dataset_can_reuse_source_manifest_identity(tmp_path):
    class Metadata:
        def __init__(self):
            self.features = {}
            self.episodes = {episode_id: {"length": 4} for episode_id in range(44)}

        @staticmethod
        def get_data_file_path(episode_id):
            return f"data/chunk-000/episode_{episode_id:06d}.parquet"

    manifest_path = tmp_path / "split.json"
    manifest_identity = "agilex_make_breakfast_subtask_730_frozen_head"
    completion_data.load_or_create_split_manifest(manifest_path, range(44), repo_id=manifest_identity)

    info = completion_data.prepare_completion_data(
        Metadata(),
        repo_id="agilex_make_breakfast_subtask_730_frozen_head_progress",
        dataset_root=tmp_path,
        label_key="progress",
        manifest_path=manifest_path,
        manifest_repo_id=manifest_identity,
        objective="progress",
        audit_labels=False,
    )

    assert info.manifest.repo_id == manifest_identity
    assert info.manifest.episode_ids("train")


def test_existing_manifest_rejects_dataset_drift(tmp_path):
    path = tmp_path / "split.json"
    completion_data.load_or_create_split_manifest(path, range(44), repo_id="org/breakfast")

    with pytest.raises(ValueError, match="episode IDs do not match"):
        completion_data.load_or_create_split_manifest(path, range(48), repo_id="org/breakfast")


@pytest.mark.parametrize(
    ("frame_count", "window_frames", "expected"),
    [
        (5, 3, [0.0, 0.0, 1.0, 1.0, 1.0]),
        (3, 3, [1.0, 1.0, 1.0]),
        (1, 1, [1.0]),
    ],
)
def test_window_completion_targets_flat_over_trailing_window(frame_count, window_frames, expected):
    targets = completion_data.make_window_completion_targets(frame_count, window_frames)

    assert targets.dtype == np.float32
    np.testing.assert_array_equal(targets, np.asarray(expected, dtype=np.float32))


def test_window_completion_targets_rejects_episode_shorter_than_window():
    with pytest.raises(ValueError, match="shorter than window_frames"):
        completion_data.make_window_completion_targets(2, 3)


@pytest.mark.parametrize(
    ("frame_count", "window_frames", "ramp_start", "expected"),
    [
        (5, 3, 0.5, [0.0, 0.0, 0.5, 0.75, 1.0]),
        (3, 3, 0.5, [0.5, 0.75, 1.0]),
        (1, 1, 0.5, [1.0]),
        (4, 4, 0.0, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]),
    ],
)
def test_window_progress_targets_ramp_over_trailing_window(frame_count, window_frames, ramp_start, expected):
    targets = completion_data.make_window_progress_targets(frame_count, window_frames, ramp_start)

    assert targets.dtype == np.float32
    np.testing.assert_allclose(targets, np.asarray(expected, dtype=np.float32), atol=1e-6)
    assert targets[-1] == np.float32(1.0)


def test_window_progress_targets_rejects_episode_shorter_than_window():
    with pytest.raises(ValueError, match="shorter than window_frames"):
        completion_data.make_window_progress_targets(2, 3, 0.5)


@pytest.mark.parametrize("bad_ramp_start", [-0.1, 1.0, 1.5])
def test_window_progress_targets_rejects_ramp_start_outside_unit_interval(bad_ramp_start):
    with pytest.raises(ValueError, match="ramp_start"):
        completion_data.make_window_progress_targets(5, 3, bad_ramp_start)


def test_episode_audit_accepts_custom_window_frames(tmp_path):
    path = tmp_path / "episode_000040.parquet"
    _write_episode(path, 40, [0, 0, 1, 1, 1])

    audit = completion_data.audit_episode_parquet(path, episode_id=40, expected_length=5, window_frames=3)

    assert audit.positive_count == 3
    assert audit.negative_count == 2


def test_episode_audit_with_custom_window_frames_rejects_early_positive(tmp_path):
    path = tmp_path / "episode_000041.parquet"
    _write_episode(path, 41, [0, 1, 1, 1, 1])

    with pytest.raises(ValueError, match=r"episode 41.*before its last 3 frames"):
        completion_data.audit_episode_parquet(path, episode_id=41, expected_length=5, window_frames=3)


def test_episode_audit_with_custom_window_frames_exempts_shorter_episode(tmp_path):
    path = tmp_path / "episode_000042.parquet"
    _write_episode(path, 42, [0, 0])

    audit = completion_data.audit_episode_parquet(path, episode_id=42, expected_length=2, window_frames=3)

    assert audit.positive_count == 0
    assert audit.negative_count == 2


def test_window_progress_episode_audit_accepts_partial_window(tmp_path):
    path = tmp_path / "episode_000050.parquet"
    labels = completion_data.make_window_progress_targets(5, 3, 0.5)
    _write_progress_episode(path, 50, labels)

    audit = completion_data.audit_window_progress_episode_parquet(
        path, episode_id=50, expected_length=5, window_frames=3, ramp_start=0.5
    )

    assert audit.frame_count == 5
    assert audit.task_index == 17


def test_window_progress_episode_audit_accepts_whole_episode_as_window(tmp_path):
    path = tmp_path / "episode_000051.parquet"
    labels = completion_data.make_window_progress_targets(3, 3, 0.5)
    _write_progress_episode(path, 51, labels)

    audit = completion_data.audit_window_progress_episode_parquet(
        path, episode_id=51, expected_length=3, window_frames=3, ramp_start=0.5
    )

    assert audit.frame_count == 3


def test_window_progress_episode_audit_rejects_mismatched_ramp(tmp_path):
    path = tmp_path / "episode_000052.parquet"
    # Written with the wrong ramp_start relative to what the audit expects.
    labels = completion_data.make_window_progress_targets(5, 3, 0.8)
    _write_progress_episode(path, 52, labels)

    with pytest.raises(ValueError, match=r"episode 52.*tail-window ramp"):
        completion_data.audit_window_progress_episode_parquet(
            path, episode_id=52, expected_length=5, window_frames=3, ramp_start=0.5
        )
