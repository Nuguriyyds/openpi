from __future__ import annotations

import numpy as np

from openpi.training import raw_prefix_completion_features as raw_features
from openpi.training import temporal_completion_data as temporal_data


def _manifest(tmp_path):
    episodes = tuple(
        temporal_data.SubtaskEpisodeRecord(
            episode_id=group_id * 4 + task_index,
            task_index=task_index,
            length=60,
        )
        for group_id in range(10)
        for task_index in range(4)
    )
    groups = temporal_data.build_subtask_groups(episodes)
    full = tuple(temporal_data.FullEpisodeRecord(group_id, 247) for group_id in range(10))
    matches = tuple(temporal_data.IdentityMatchRecord(group_id, group_id) for group_id in range(10))
    identities = temporal_data.build_trajectory_identities(groups, full, matches)
    return temporal_data.create_temporal_manifest(
        identities,
        source_subtask_repo_id="org/subtasks",
        source_subtask_root=tmp_path / "subtasks",
        source_full_repo_id="org/full",
        source_full_root=tmp_path / "full",
        task_prompts=("task 0", "task 1", "task 2", "task 3"),
    )


def test_raw_prefix_cache_round_trip_and_dataset_current_row(tmp_path):
    manifest = _manifest(tmp_path)
    rows = raw_features.manifest_rows(manifest)
    prefix_out = np.arange(len(rows) * 3 * 4, dtype=np.float32).reshape(len(rows), 3, 4)
    prefix_mask = np.ones((len(rows), 3), dtype=np.bool_)
    segment_ids = np.asarray([0, 1, 3], dtype=np.int32)
    position_ids = np.asarray([0, 0, 0], dtype=np.int32)
    cache_path = tmp_path / "raw-cache"

    metadata = raw_features.save_raw_prefix_cache(
        cache_path,
        manifest=manifest,
        prefix_out=prefix_out,
        prefix_mask=prefix_mask,
        prefix_segment_ids=segment_ids,
        prefix_position_ids=position_ids,
        model_config_name="clean-config",
        checkpoint_path="/clean/checkpoint",
    )
    loaded = raw_features.load_raw_prefix_cache(
        cache_path,
        manifest=manifest,
        expected_checkpoint_path="/clean/checkpoint",
        expected_model_config_name="clean-config",
    )

    assert metadata.row_count == len(rows)
    assert metadata.feature_count == len(rows)
    assert metadata.shard_count == 1
    assert loaded.prefix_out.dtype == np.float16
    assert loaded.prefix_mask.dtype == np.bool_
    assert loaded.prefix_out.shape == (len(rows), 3, 4)
    assert loaded.rows == rows
    dataset = raw_features.RawPrefixCompletionDataset(loaded, "val")
    values, mask, loaded_segments, loaded_positions, target = dataset[0]
    row = dataset.samples[0]
    assert values.shape == (3, 4)
    assert values.dtype == np.float16
    assert mask.shape == (3,)
    np.testing.assert_array_equal(loaded_segments, segment_ids)
    np.testing.assert_array_equal(loaded_positions, position_ids)
    assert target == np.float32(row.label)
    assert row.source_episode_ids[-1] >= 0
    assert row.source_frame_indices[-1] == row.logical_tick


def test_raw_prefix_cache_shards_unique_features_and_maps_rows(tmp_path):
    manifest = _manifest(tmp_path)
    rows = raw_features.manifest_rows(manifest)
    prefix_out = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    prefix_mask = np.ones((2, 3), dtype=np.bool_)
    row_feature_indices = np.arange(len(rows), dtype=np.int64) % 2
    cache_path = tmp_path / "raw-cache-sharded"

    metadata = raw_features.save_raw_prefix_cache(
        cache_path,
        manifest=manifest,
        prefix_out=prefix_out,
        prefix_mask=prefix_mask,
        prefix_segment_ids=np.asarray([0, 1, 3], dtype=np.int32),
        prefix_position_ids=np.asarray([0, 0, 0], dtype=np.int32),
        model_config_name="clean-config",
        checkpoint_path="/clean/checkpoint",
        row_feature_indices=row_feature_indices,
        max_shard_bytes=27,
    )
    loaded = raw_features.load_raw_prefix_cache(cache_path, manifest=manifest)

    assert metadata.feature_count == 2
    assert metadata.shard_count == 2
    assert loaded.prefix_out.shape == (2, 3, 4)
    np.testing.assert_array_equal(loaded.row_feature_indices, row_feature_indices)
    values, masks = loaded.features_for_rows(np.asarray([0, 1, 2], dtype=np.int64))
    np.testing.assert_array_equal(values, prefix_out[[0, 1, 0]].astype(np.float16))
    np.testing.assert_array_equal(masks, prefix_mask[[0, 1, 0]])
