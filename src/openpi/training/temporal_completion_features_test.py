import dataclasses
import hashlib

import numpy as np
import pytest

from openpi.training import temporal_completion_data as temporal_data
from openpi.training import temporal_completion_features as features


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _manifest():
    episodes = [
        temporal_data.SubtaskEpisodeRecord(episode_id=index, task_index=index % 4, length=60) for index in range(12)
    ]
    groups = temporal_data.build_subtask_groups(episodes)
    full = [temporal_data.FullEpisodeRecord(episode_id=index, length=240) for index in range(3)]
    matches = [temporal_data.IdentityMatchRecord(index, index, _sha(f"evidence-{index}")) for index in range(3)]
    identities = temporal_data.build_trajectory_identities(groups, full, matches)
    return temporal_data.create_temporal_manifest(
        identities,
        source_subtask_repo_id="subtasks",
        source_subtask_root=".",
        source_full_repo_id="full",
        source_full_root=".",
        subtask_metadata_fingerprint=_sha("subtasks"),
        full_metadata_fingerprint=_sha("full"),
        task_prompts=("task 0", "task 1", "task 2", "task 3"),
    )


def test_feature_cache_round_trip_and_split_dataset(tmp_path):
    manifest = _manifest()
    rows = features.manifest_rows(manifest)
    history = np.arange(len(rows) * 3 * 4, dtype=np.float16).reshape(len(rows), 3, 4)
    path = tmp_path / "features.npz"
    features.save_temporal_feature_cache(
        path,
        manifest=manifest,
        prefix_history=history,
        checkpoint_fingerprint=_sha("checkpoint"),
        preprocess_fingerprint=_sha("preprocess"),
        model_config_name="clean",
        checkpoint_path="/checkpoint/49999",
    )
    loaded = features.load_temporal_feature_cache(
        path,
        manifest=manifest,
        expected_checkpoint_fingerprint=_sha("checkpoint"),
        expected_preprocess_fingerprint=_sha("preprocess"),
        expected_model_config_name="clean",
    )
    assert loaded.rows == rows
    np.testing.assert_array_equal(loaded.prefix_history, history)
    dataset = features.TemporalFeatureDataset(loaded, "train")
    sample_history, target = dataset[0]
    assert sample_history.shape == (3, 4)
    assert sample_history.dtype == np.float32
    assert target in (0.0, 1.0)
    assert all(row.split == "train" for row in dataset.samples)


def test_feature_cache_rejects_stale_manifest(tmp_path):
    manifest = _manifest()
    rows = features.manifest_rows(manifest)
    path = tmp_path / "features.npz"
    features.save_temporal_feature_cache(
        path,
        manifest=manifest,
        prefix_history=np.zeros((len(rows), 3, 2), dtype=np.float16),
        checkpoint_fingerprint=_sha("checkpoint"),
        preprocess_fingerprint=_sha("preprocess"),
        model_config_name="clean",
        checkpoint_path="/checkpoint/49999",
    )
    stale = dataclasses.replace(manifest, source_subtask_repo_id="different")
    with pytest.raises(ValueError, match="different temporal manifest"):
        features.load_temporal_feature_cache(path, manifest=stale)


def test_feature_cache_rejects_wrong_history_shape(tmp_path):
    manifest = _manifest()
    rows = features.manifest_rows(manifest)
    with pytest.raises(ValueError, match="shape"):
        features.save_temporal_feature_cache(
            tmp_path / "features.npz",
            manifest=manifest,
            prefix_history=np.zeros((len(rows), 2, 4), dtype=np.float16),
            checkpoint_fingerprint=_sha("checkpoint"),
            preprocess_fingerprint=_sha("preprocess"),
            model_config_name="clean",
            checkpoint_path="/checkpoint/49999",
        )


def test_row_decoder_reads_each_npz_member_once():
    rows = features.manifest_rows(_manifest())
    values = features._row_arrays(rows)  # noqa: SLF001

    class CountingArrays:
        files = tuple(values)

        def __init__(self):
            self.counts = dict.fromkeys(values, 0)

        def __getitem__(self, name):
            self.counts[name] += 1
            return values[name]

    arrays = CountingArrays()

    assert features._rows_from_arrays(arrays) == rows  # noqa: SLF001
    assert set(arrays.counts.values()) == {1}
