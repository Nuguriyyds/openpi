import json

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


def test_existing_manifest_rejects_dataset_drift(tmp_path):
    path = tmp_path / "split.json"
    completion_data.load_or_create_split_manifest(path, range(44), repo_id="org/breakfast")

    with pytest.raises(ValueError, match="episode IDs do not match"):
        completion_data.load_or_create_split_manifest(path, range(48), repo_id="org/breakfast")
