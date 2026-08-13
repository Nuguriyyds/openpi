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


def test_split_manifest_allows_zero_val_groups():
    """val_groups=0 is valid (disables val); val_groups<0 is rejected."""

    manifest = completion_data.create_split_manifest(
        range(48),
        repo_id="org/breakfast",
        val_groups=0,
        test_groups=10,
    )
    assert len(manifest.episode_ids("val")) == 0
    assert len(manifest.episode_ids("test")) == 40
    assert len(manifest.episode_ids("train")) == 8

    with pytest.raises(ValueError, match="val_groups must be non-negative"):
        completion_data.create_split_manifest(
            range(48),
            repo_id="org/breakfast",
            val_groups=-1,
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


def test_window_completion_targets_exempts_episode_shorter_than_window():
    targets = completion_data.make_window_completion_targets(2, 3)

    assert targets.dtype == np.float32
    np.testing.assert_array_equal(targets, np.asarray([0.0, 0.0], dtype=np.float32))


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


def test_window_progress_targets_exempts_episode_shorter_than_window():
    targets = completion_data.make_window_progress_targets(2, 3, 0.5)

    assert targets.dtype == np.float32
    np.testing.assert_array_equal(targets, np.asarray([0.0, 0.0], dtype=np.float32))


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


def test_window_progress_episode_audit_accepts_all_zero_shorter_episode(tmp_path):
    """Mirrors the real 1-frame episode 2797: too short for the window, labeled all-0."""

    path = tmp_path / "episode_002797.parquet"
    _write_progress_episode(path, 2797, [0.0])

    audit = completion_data.audit_window_progress_episode_parquet(
        path, episode_id=2797, expected_length=1, window_frames=15, ramp_start=0.8
    )

    assert audit.frame_count == 1


def test_window_progress_episode_audit_rejects_nonzero_label_on_shorter_episode(tmp_path):
    path = tmp_path / "episode_000053.parquet"
    _write_progress_episode(path, 53, [0.5])

    with pytest.raises(ValueError, match=r"episode 53.*shorter than.*all labels must be 0"):
        completion_data.audit_window_progress_episode_parquet(
            path, episode_id=53, expected_length=1, window_frames=15, ramp_start=0.8
        )


# ---------------------------------------------------------------------------
#  Boundary scheme tests
# ---------------------------------------------------------------------------

_BOUNDARY_GROUP_IDS = (0, 1, 2, 3)


def _write_boundary_episode(
    path,
    episode_id: int,
    group_position: int,
    *,
    original_length: int = 20,
    copy_frames: int | None = None,
    completion_override=None,
    source_episode_override=None,
    is_copy_override=None,
    task_index_override=None,
):
    """Writes a synthetic boundary parquet for one episode.

    Default behavior produces a *valid* boundary episode.  Override parameters
    inject deliberate errors for rejection tests.
    """

    is_subtask4 = group_position == 3
    copy_n = 0 if is_subtask4 else (copy_frames if copy_frames is not None else 5)
    new_length = original_length + copy_n
    task_index = task_index_override if task_index_override is not None else group_position

    completion = np.zeros(new_length, dtype=np.float32)
    completion[-10:] = 1.0
    if completion_override is not None:
        completion = np.asarray(completion_override, dtype=np.float32)

    source_ep = np.full(new_length, episode_id, dtype=np.int64)
    source_fr = np.arange(new_length, dtype=np.int64)
    is_copy = np.zeros(new_length, dtype=np.int8)
    if not is_subtask4 and copy_n > 0:
        source_ep[original_length:] = episode_id + 1
        source_fr[original_length:] = np.arange(copy_n, dtype=np.int64)
        is_copy[original_length:] = 1

    if source_episode_override is not None:
        source_ep = np.asarray(source_episode_override, dtype=np.int64)
    if is_copy_override is not None:
        is_copy = np.asarray(is_copy_override, dtype=np.int8)

    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([episode_id] * new_length, type=pa.int64()),
                "frame_index": pa.array(np.arange(new_length, dtype=np.int64), type=pa.int64()),
                "task_index": pa.array([task_index] * new_length, type=pa.int64()),
                "completion": pa.array(completion, type=pa.float32()),
                "source_episode_index": pa.array(source_ep, type=pa.int64()),
                "source_frame_index": pa.array(source_fr, type=pa.int64()),
                "is_boundary_copy": pa.array(is_copy, type=pa.int8()),
            }
        ),
        path,
    )
    return new_length


def test_boundary_audit_accepts_valid_subtask1(tmp_path):
    path = tmp_path / "episode_000000.parquet"
    new_len = _write_boundary_episode(path, 0, group_position=0)

    audit = completion_data.audit_boundary_completion_episode_parquet(
        path,
        episode_id=0,
        expected_length=new_len,
        group_episode_ids=_BOUNDARY_GROUP_IDS,
        group_position=0,
    )
    assert audit.positive_count == 10
    assert audit.negative_count == new_len - 10
    assert audit.boundary_copy_count == 5
    assert audit.is_subtask4 is False


def test_boundary_audit_accepts_valid_subtask4(tmp_path):
    path = tmp_path / "episode_000003.parquet"
    new_len = _write_boundary_episode(path, 3, group_position=3)

    audit = completion_data.audit_boundary_completion_episode_parquet(
        path,
        episode_id=3,
        expected_length=new_len,
        group_episode_ids=_BOUNDARY_GROUP_IDS,
        group_position=3,
    )
    assert audit.positive_count == 10
    assert audit.boundary_copy_count == 0
    assert audit.is_subtask4 is True


def test_boundary_audit_accepts_one_frame_excluded_episode_as_all_negative(tmp_path):
    path = tmp_path / "episode_002797.parquet"
    new_len = _write_boundary_episode(
        path,
        2797,
        group_position=1,
        original_length=1,
        copy_frames=0,
        completion_override=[0.0],
    )

    audit = completion_data.audit_boundary_completion_episode_parquet(
        path,
        episode_id=2797,
        expected_length=new_len,
        group_episode_ids=(2796, 2797, 2798, 2799),
        group_position=1,
        excluded_episode_ids={2797},
    )

    assert audit.positive_count == 0
    assert audit.negative_count == 1
    assert audit.boundary_copy_count == 0


def test_boundary_audit_skips_excluded_episode_and_copies_next_valid_episode(tmp_path):
    path = tmp_path / "episode_002796.parquet"
    source_episodes = np.full(25, 2796, dtype=np.int64)
    source_episodes[-5:] = 2798
    new_len = _write_boundary_episode(
        path,
        2796,
        group_position=0,
        original_length=20,
        copy_frames=5,
        source_episode_override=source_episodes,
    )

    audit = completion_data.audit_boundary_completion_episode_parquet(
        path,
        episode_id=2796,
        expected_length=new_len,
        group_episode_ids=(2796, 2797, 2798, 2799),
        group_position=0,
        excluded_episode_ids={2797},
    )

    assert audit.positive_count == 10
    assert audit.negative_count == 15
    assert audit.boundary_copy_count == 5
    assert set(audit.source_episode_indices[-5:].tolist()) == {2798}


def test_boundary_audit_rejects_non_binary_labels(tmp_path):
    path = tmp_path / "episode_000000.parquet"
    new_len = _write_boundary_episode(
        path,
        0,
        group_position=0,
        completion_override=np.array([0] * 15 + [0.5] + [1] * 9, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="outside 0/1"):
        completion_data.audit_boundary_completion_episode_parquet(
            path,
            episode_id=0,
            expected_length=new_len,
            group_episode_ids=_BOUNDARY_GROUP_IDS,
            group_position=0,
        )


def test_boundary_audit_rejects_wrong_positive_count(tmp_path):
    """Positives must occupy exactly the last 10 frames; 9 positives (one of the
    last 10 flipped to 0) is rejected because the structural check fires before
    the redundant positive-count backstop."""

    path = tmp_path / "episode_000000.parquet"
    new_len = _write_boundary_episode(
        path,
        0,
        group_position=0,
        completion_override=np.array([0] * 16 + [1] * 9, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="last 10 frames"):
        completion_data.audit_boundary_completion_episode_parquet(
            path,
            episode_id=0,
            expected_length=new_len,
            group_episode_ids=_BOUNDARY_GROUP_IDS,
            group_position=0,
        )


def test_boundary_audit_rejects_cross_group_copy(tmp_path):
    """Copy frames sourced from an episode outside the task group must fail.

    The specific ``source from next episode`` check fires before the redundant
    cross-group backstop, so we match that message — the intent (rejecting an
    out-of-group source) is the same.
    """

    path = tmp_path / "episode_000000.parquet"
    new_len = _write_boundary_episode(path, 0, group_position=0)
    # Tamper: set copy-frame source to episode 99 (outside group 0-3).
    table = pq.read_table(path)
    src = table["source_episode_index"].to_numpy().copy()
    src[-5:] = 99
    table = table.set_column(
        table.column_names.index("source_episode_index"),
        "source_episode_index",
        pa.array(src, type=pa.int64()),
    )
    pq.write_table(table, path)

    with pytest.raises(ValueError, match="source from next valid episode"):
        completion_data.audit_boundary_completion_episode_parquet(
            path,
            episode_id=0,
            expected_length=new_len,
            group_episode_ids=_BOUNDARY_GROUP_IDS,
            group_position=0,
        )


def test_boundary_audit_rejects_subtask4_with_copy_frames(tmp_path):
    path = tmp_path / "episode_000003.parquet"
    # Subtask 4 must have no copies; inject copies on the trailing rows to
    # trigger rejection.  (_write_boundary_episode normally refuses to add
    # copies for subtask 4, so we override is_boundary_copy directly.)
    new_len = _write_boundary_episode(
        path,
        3,
        group_position=3,
        is_copy_override=np.array([0] * 15 + [1] * 5, dtype=np.int8),
    )

    with pytest.raises(ValueError, match=r"subtask 4.*no boundary copies"):
        completion_data.audit_boundary_completion_episode_parquet(
            path,
            episode_id=3,
            expected_length=new_len,
            group_episode_ids=_BOUNDARY_GROUP_IDS,
            group_position=3,
        )


def test_boundary_train_sample_indices_positive_ordinary_and_forced(tmp_path):
    """Sampling keeps all positives, every-15-frame ordinary negatives, and
    forced first-5 negatives for subtasks 2/3/4."""

    path = tmp_path / "episode_000001.parquet"
    # Subtask 2 (group_position=1): 20 original + 5 copy = 25 frames.
    new_len = _write_boundary_episode(path, 1, group_position=1)
    audit = completion_data.audit_boundary_completion_episode_parquet(
        path,
        episode_id=1,
        expected_length=new_len,
        group_episode_ids=_BOUNDARY_GROUP_IDS,
        group_position=1,
    )

    positives, ordinary, forced = completion_data.boundary_train_sample_indices(
        audit,
        stride=15,
        forced_first_n=5,
    )
    # All 10 positives kept.
    assert len(positives) == 10
    # Ordinary negatives: source_frame % 15 == 0, non-copy, completion==0.
    # source_frame for original frames = 0..19; on grid: 0, 15 → but 0 is also
    # a forced negative.  Ordinary grid excludes forced set; dedup happens later.
    assert len(ordinary) >= 1
    # Forced negatives: first 5 original frames of subtask 2/3/4.
    assert len(forced) == 5
    assert set(forced.tolist()) == {0, 1, 2, 3, 4}

    # Deduped sample set.
    sample_set = completion_data.build_boundary_train_sample_set(
        audit,
        stride=15,
        forced_first_n=5,
    )
    assert len(sample_set) == len(np.unique(sample_set))
    # All positives in the sample set.
    assert set(positives.tolist()).issubset(set(sample_set.tolist()))


def test_boundary_train_sample_indices_subtask1_has_no_forced_negatives(tmp_path):
    """Subtask 1 (group_position=0) has no forced first-5 negatives."""

    path = tmp_path / "episode_000000.parquet"
    new_len = _write_boundary_episode(path, 0, group_position=0)
    audit = completion_data.audit_boundary_completion_episode_parquet(
        path,
        episode_id=0,
        expected_length=new_len,
        group_episode_ids=_BOUNDARY_GROUP_IDS,
        group_position=0,
    )
    _, _, forced = completion_data.boundary_train_sample_indices(audit, stride=15, forced_first_n=5)
    assert len(forced) == 0
