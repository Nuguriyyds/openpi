from __future__ import annotations

import numpy as np

from openpi.training import raw_prefix_completion_features as raw_features
from openpi.training import temporal_completion_data as temporal_data
from openpi.training import temporal_raw_prefix_completion_features as temporal_raw_features


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


def test_temporal_raw_prefix_history_reuses_base_and_indexes_missing_extension(tmp_path):
    manifest = _manifest(tmp_path)
    rows = raw_features.manifest_rows(manifest)
    prefix_out = np.arange(len(rows) * 3 * 4, dtype=np.float32).reshape(len(rows), 3, 4)
    prefix_mask = np.ones((len(rows), 3), dtype=np.bool_)
    segment_ids = np.asarray([0, 1, 3], dtype=np.int32)
    position_ids = np.asarray([0, 0, 0], dtype=np.int32)
    base_path = tmp_path / "raw-cache"
    raw_features.save_raw_prefix_cache(
        base_path,
        manifest=manifest,
        prefix_out=prefix_out,
        prefix_mask=prefix_mask,
        prefix_segment_ids=segment_ids,
        prefix_position_ids=position_ids,
        model_config_name="clean-config",
        checkpoint_path="/clean/checkpoint",
    )
    base_cache = raw_features.load_raw_prefix_cache(base_path, manifest=manifest)
    plan = temporal_raw_features.build_temporal_raw_prefix_history_plan(manifest, base_cache)

    assert plan.row_count == len(rows)
    assert plan.history_location_kind.shape == (len(rows), 3)
    assert np.any(plan.history_location_kind == temporal_raw_features.BASE_LOCATION)
    assert np.any(plan.history_location_kind == temporal_raw_features.EXTENSION_LOCATION)
    assert plan.extension_count == len(plan.missing_keys) > 0

    extension_values = np.arange(plan.extension_count * 3 * 4, dtype=np.float32).reshape(plan.extension_count, 3, 4)
    extension_masks = np.ones((plan.extension_count, 3), dtype=np.bool_)
    history_path = tmp_path / "temporal-history"
    metadata = temporal_raw_features.save_temporal_raw_prefix_history(
        history_path,
        manifest=manifest,
        base_cache=base_cache,
        plan=plan,
        extension_prefix_out=extension_values,
        extension_prefix_mask=extension_masks,
        base_cache_path=base_path,
        max_shard_bytes=27,
    )
    loaded = temporal_raw_features.load_temporal_raw_prefix_history(
        history_path,
        manifest=manifest,
        base_cache=base_cache,
        expected_base_cache_path=base_path,
        expected_checkpoint_path="/clean/checkpoint",
        expected_model_config_name="clean-config",
    )

    assert metadata.extension_shard_count == plan.extension_count
    assert loaded.extension_count == plan.extension_count
    dataset = temporal_raw_features.TemporalRawPrefixCompletionDataset(loaded, "val")
    values, masks, loaded_segments, loaded_positions, target = dataset[0]
    cache_index = int(loaded.indices_for_split("val")[0])
    assert values.shape == (3, 3, 4)
    assert values.dtype == np.float16
    assert masks.shape == (3, 3)
    np.testing.assert_array_equal(loaded_segments, segment_ids)
    np.testing.assert_array_equal(loaded_positions, position_ids)
    assert target == np.float32(loaded.rows[cache_index].label)
    for slot in range(3):
        kind = int(loaded.history_location_kind[cache_index, slot])
        index = int(loaded.history_location_index[cache_index, slot])
        expected = prefix_out[index] if kind == temporal_raw_features.BASE_LOCATION else extension_values[index]
        np.testing.assert_array_equal(values[slot], expected.astype(np.float16))

    batch_indices = loaded.indices_for_split("val")[:2]
    batch_values, batch_masks = loaded.features_for_rows(batch_indices)
    assert batch_values.shape == (len(batch_indices), 3, 3, 4)
    assert batch_values.dtype == np.float16
    assert batch_masks.shape == (len(batch_indices), 3, 3)


def test_temporal_raw_prefix_history_writer_streams_batches(tmp_path):
    manifest = _manifest(tmp_path)
    rows = raw_features.manifest_rows(manifest)
    prefix_out = np.arange(len(rows) * 3 * 4, dtype=np.float32).reshape(len(rows), 3, 4)
    prefix_mask = np.ones((len(rows), 3), dtype=np.bool_)
    segment_ids = np.asarray([0, 1, 3], dtype=np.int32)
    position_ids = np.asarray([0, 0, 0], dtype=np.int32)
    base_path = tmp_path / "raw-cache"
    raw_features.save_raw_prefix_cache(
        base_path,
        manifest=manifest,
        prefix_out=prefix_out,
        prefix_mask=prefix_mask,
        prefix_segment_ids=segment_ids,
        prefix_position_ids=position_ids,
        model_config_name="clean-config",
        checkpoint_path="/clean/checkpoint",
    )
    base_cache = raw_features.load_raw_prefix_cache(base_path, manifest=manifest)
    plan = temporal_raw_features.build_temporal_raw_prefix_history_plan(manifest, base_cache)
    extension_values = np.arange(plan.extension_count * 3 * 4, dtype=np.float32).reshape(plan.extension_count, 3, 4)
    extension_masks = np.ones((plan.extension_count, 3), dtype=np.bool_)

    writer = temporal_raw_features.TemporalRawPrefixHistoryWriter(
        tmp_path / "temporal-history-streamed",
        manifest=manifest,
        base_cache=base_cache,
        plan=plan,
        base_cache_path=base_path,
        max_shard_bytes=27,
    )
    for start in range(0, plan.extension_count, 2):
        stop = min(start + 2, plan.extension_count)
        writer.append(
            extension_values[start:stop],
            extension_masks[start:stop],
            segment_ids,
            position_ids,
        )
    metadata = writer.finalize()

    assert metadata.extension_count == plan.extension_count
    assert metadata.extension_shard_count == plan.extension_count
    assert metadata.extension_shard_rows == (1,) * plan.extension_count
    loaded = temporal_raw_features.load_temporal_raw_prefix_history(
        tmp_path / "temporal-history-streamed",
        manifest=manifest,
        base_cache=base_cache,
        expected_base_cache_path=base_path,
    )
    values, masks = loaded.features_for_rows(np.asarray([0, 1], dtype=np.int64))
    assert values.shape == (2, 3, 3, 4)
    assert masks.shape == (2, 3, 3)
