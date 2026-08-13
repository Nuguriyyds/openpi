import dataclasses
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import completion
from openpi.training import completion_data
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_completion_target_is_returned_separately_from_observation():
    model_config = pi0_config.Pi0Config(
        action_dim=4,
        action_horizon=2,
        max_token_len=8,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    batch = {
        **model_config.fake_obs(batch_size=2).to_dict(),
        "actions": model_config.fake_act(batch_size=2),
        _data_loader.COMPLETION_TARGET_KEY: jnp.asarray([0, 1], dtype=jnp.int32),
    }
    data_config = _config.DataConfig(repo_id="fake", completion_label_key="completion")
    loader = _data_loader.DataLoaderImpl(data_config, [batch])

    observation, actions, targets = next(iter(loader))

    assert actions.shape == (2, 2, 4)
    assert targets.tolist() == [0, 1]
    assert not hasattr(observation, "completion")


def test_pad_batch_to_multiple_repeats_only_the_last_row():
    batch = {
        "values": np.arange(5, dtype=np.int32),
        "matrix": np.arange(10, dtype=np.float32).reshape(5, 2),
    }

    padded = _data_loader._pad_batch_to_multiple(batch, 4)  # noqa: SLF001

    assert padded["values"].tolist() == [0, 1, 2, 3, 4, 4, 4, 4]
    np.testing.assert_array_equal(padded["matrix"][:5], batch["matrix"])
    np.testing.assert_array_equal(padded["matrix"][5:], np.repeat(batch["matrix"][-1:], 3, axis=0))


def test_episode_subset_dataset_preserves_global_lerobot_indices():
    class Dataset:
        def __init__(self):
            self.episode_data_index = {
                "from": np.asarray([0, 2, 5, 9], dtype=np.int64),
                "to": np.asarray([2, 5, 9, 10], dtype=np.int64),
            }

        def __getitem__(self, index):
            return {"global_index": index, "episode_index": 2 if 5 <= index < 9 else 3}

    subset = _data_loader.EpisodeSubsetDataset(Dataset(), [2, 3])

    assert len(subset) == 5
    assert [subset[index]["global_index"] for index in range(len(subset))] == [5, 6, 7, 8, 9]
    assert [subset[index]["episode_index"] for index in range(len(subset))] == [2, 2, 2, 2, 3]


def test_balanced_completion_sampler_guarantees_each_batch_composition():
    audits = {
        episode_id: completion_data.EpisodeAudit(
            episode_id=episode_id,
            frame_count=30,
            positive_count=2,
            negative_count=28,
        )
        for episode_id in (7, 8)
    }
    sampler = _data_loader.BalancedCompletionSampler(
        (7, 8),
        audits,
        batch_size=8,
        positive_fraction=0.25,
        hard_negative_fraction=0.25,
        hard_negative_window=4,
        seed=123,
    )
    indices = list(sampler)
    positive = {28, 29, 58, 59}
    hard_negative = {24, 25, 26, 27, 54, 55, 56, 57}

    assert sampler.batch_composition == {"positive": 2, "hard_negative": 2, "ordinary_negative": 4}
    assert len(indices) % 8 == 0
    for start in range(0, len(indices), 8):
        batch = indices[start : start + 8]
        assert sum(index in positive for index in batch) == 2
        assert sum(index in hard_negative for index in batch) == 2
        assert sum(index not in positive | hard_negative for index in batch) == 4


def test_progress_stratified_sampler_covers_all_bins_and_is_reproducible():
    episode_ids = (7, 8)
    audits = {
        episode_id: completion_data.EpisodeAudit(
            episode_id=episode_id,
            frame_count=21,
            positive_count=0,
            negative_count=0,
            task_index=episode_id,
        )
        for episode_id in episode_ids
    }
    sampler = _data_loader.ProgressStratifiedSampler(episode_ids, audits, batch_size=20, seed=123)
    same_seed_sampler = _data_loader.ProgressStratifiedSampler(episode_ids, audits, batch_size=20, seed=123)

    indices = list(sampler)
    assert indices == list(same_seed_sampler)
    assert sampler.batch_composition == {f"bin_{index}": 2 for index in range(10)}
    assert sampler.pool_sizes.keys() == {f"bin_{index}" for index in range(10)}
    assert all(size > 0 for size in sampler.pool_sizes.values())
    assert all(size == 2 for size in sampler.episode_pool_sizes.values())

    targets = completion_data.make_progress_targets(21)
    for batch_start in range(0, len(indices), 20):
        batch = indices[batch_start : batch_start + 20]
        bin_counts = np.zeros(10, dtype=np.int32)
        for index in batch:
            local_frame = index % 21
            bin_index = min(int(targets[local_frame] * 10), 9)
            bin_counts[bin_index] += 1
        np.testing.assert_array_equal(bin_counts, np.full(10, 2, dtype=np.int32))


def test_progress_stratified_sampler_rotates_remainder_bins():
    audits = {
        episode_id: completion_data.EpisodeAudit(
            episode_id=episode_id,
            frame_count=21,
            positive_count=0,
            negative_count=0,
        )
        for episode_id in (1, 2)
    }
    sampler = _data_loader.ProgressStratifiedSampler((1, 2), audits, batch_size=24, seed=9)
    indices = list(sampler)
    targets = completion_data.make_progress_targets(21)

    first_batch_counts = np.zeros(10, dtype=np.int32)
    second_batch_counts = np.zeros(10, dtype=np.int32)
    for index in indices[:24]:
        first_batch_counts[min(int(targets[index % 21] * 10), 9)] += 1
    for index in indices[24:48]:
        second_batch_counts[min(int(targets[index % 21] * 10), 9)] += 1
    np.testing.assert_array_equal(first_batch_counts, np.asarray([3, 3, 3, 3, 2, 2, 2, 2, 2, 2]))
    np.testing.assert_array_equal(second_batch_counts, np.asarray([2, 3, 3, 3, 3, 2, 2, 2, 2, 2]))


def test_s1_and_s2_share_manifest_episodes_but_only_s2_emits_target(tmp_path, monkeypatch):
    class DataFactory:
        def create(self, assets_dirs, model):
            del assets_dirs, model
            return _config.DataConfig(repo_id="org/breakfast")

    train_episodes = (8, 9, 10, 11)
    info = types.SimpleNamespace(
        manifest=types.SimpleNamespace(episode_ids=lambda split: train_episodes if split == "train" else ())
    )
    captured = []

    def fake_create_torch_data_loader(data_config, **kwargs):
        captured.append((data_config, kwargs))
        return object()

    monkeypatch.setattr(_data_loader, "create_torch_data_loader", fake_create_torch_data_loader)
    for stage, expected_label in (("action", None), ("head", "completion")):
        config = types.SimpleNamespace(
            data=DataFactory(),
            assets_dirs=tmp_path,
            model=types.SimpleNamespace(action_horizon=2),
            completion=completion.CompletionTrainingConfig(
                stage=stage,
                split_manifest_path="manifest.json",
            ),
            batch_size=2,
            num_workers=0,
            seed=42,
        )
        _data_loader.create_data_loader(config, completion_data_info=info, split="train")

        data_config, kwargs = captured[-1]
        assert data_config.episodes == train_episodes
        assert data_config.completion_label_key == expected_label
        assert kwargs["repeat"]
        assert kwargs["drop_last"]


def test_s1_prepares_split_without_auditing_completion_labels(tmp_path, monkeypatch):
    class DataFactory:
        def create(self, assets_dirs, model):
            del assets_dirs, model
            return _config.DataConfig(repo_id="org/breakfast", lerobot_home=str(tmp_path))

    metadata = types.SimpleNamespace(
        root=tmp_path / "org" / "breakfast",
        features={},
        episodes={episode_id: {"length": 4} for episode_id in range(44)},
    )
    captured = []

    def fake_prepare_completion_data(*args, **kwargs):
        del args
        captured.append(kwargs["audit_labels"])
        return types.SimpleNamespace(
            manifest=types.SimpleNamespace(episode_ids=lambda split: (0, 1, 2, 3) if split == "train" else ()),
            pos_weight=None,
            train_positive_count=None,
            train_negative_count=None,
        )

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(
        _data_loader._completion_data,  # noqa: SLF001
        "prepare_completion_data",
        fake_prepare_completion_data,
    )

    for stage, expected_audit in (("action", False), ("head", True)):
        config = types.SimpleNamespace(
            data=DataFactory(),
            assets_dirs=tmp_path,
            model=types.SimpleNamespace(),
            completion=completion.CompletionTrainingConfig(
                stage=stage,
                split_manifest_path=str(tmp_path / "split.json"),
            ),
        )
        _data_loader.prepare_completion_data(config)
        assert captured[-1] is expected_audit


# ---------------------------------------------------------------------------
#  BoundaryCompletionSampler tests
# ---------------------------------------------------------------------------


def _make_boundary_audit(
    episode_id: int,
    group_position: int,
    *,
    original_length: int = 15,
) -> completion_data.BoundaryEpisodeAudit:
    """Builds a minimal valid BoundaryEpisodeAudit for sampler tests.

    Subtasks 1-3 have ``original_length + 5`` frames (5 copy frames); subtask 4
    has ``original_length`` frames.  The last 10 frames are positive.
    """

    is_subtask4 = group_position == 3
    copy_n = 0 if is_subtask4 else 5
    new_length = original_length + copy_n
    completion = np.zeros(new_length, dtype=np.int8)
    completion[-10:] = 1
    is_copy = np.zeros(new_length, dtype=np.int8)
    source_frame = np.arange(new_length, dtype=np.int64)
    source_episode = np.full(new_length, episode_id, dtype=np.int64)
    if not is_subtask4:
        is_copy[original_length:] = 1
        source_frame[original_length:] = np.arange(copy_n, dtype=np.int64)
        source_episode[original_length:] = episode_id + 1
    return completion_data.BoundaryEpisodeAudit(
        episode_id=episode_id,
        group_position=group_position,
        task_index=group_position,
        frame_count=new_length,
        positive_count=10,
        negative_count=new_length - 10,
        boundary_copy_count=copy_n,
        is_subtask4=is_subtask4,
        completion=completion,
        source_episode_indices=source_episode,
        source_frame_indices=source_frame,
        is_boundary_copy=is_copy,
    )


def test_boundary_sampler_sample_set_composition():
    """Sample set = all positives + stride-grid ordinary negatives + forced
    first-5 negatives (subtasks 2/3/4 only)."""

    audits = {
        0: _make_boundary_audit(0, group_position=0),  # subtask 1: no forced
        1: _make_boundary_audit(1, group_position=1),  # subtask 2: forced first-5
    }
    sampler = _data_loader.BoundaryCompletionSampler(
        (0, 1),
        audits,
        batch_size=8,
        seed=42,
        stride=15,
        forced_first_n=5,
    )

    sample_set = sampler.sample_set
    # Episode 0 (20 frames, offset 0): positives 10-19, ordinary {0}, no forced.
    ep0_expected = {0, *range(10, 20)}
    # Episode 1 (20 frames, offset 20): positives 30-39, ordinary {20}, forced {20..24}.
    ep1_expected = {20, *range(21, 25), *range(30, 40)}
    expected = ep0_expected | ep1_expected
    assert set(sample_set.tolist()) == expected
    # No duplicates.
    assert len(sample_set) == len(np.unique(sample_set))


def test_boundary_sampler_steps_per_epoch_and_padded_last_batch():
    audits = {0: _make_boundary_audit(0, group_position=0)}
    batch_size = 8
    sampler = _data_loader.BoundaryCompletionSampler(
        (0,),
        audits,
        batch_size=batch_size,
        seed=42,
    )
    # Sample set = {0, 10..19} = 11 indices → steps = ceil(11/8) = 2.
    assert sampler.steps_per_epoch == 2
    assert sampler.num_samples == 11

    indices = list(sampler)
    # Total yielded = steps * batch_size = 16 (padded).
    assert len(indices) == 2 * batch_size
    # Every batch is full-sized.
    # All original 11 samples appear at least once.
    unique_yielded = set(indices)
    assert set(sampler.sample_set.tolist()).issubset(unique_yielded)


def test_boundary_sampler_is_deterministic_same_seed():
    audits = {
        0: _make_boundary_audit(0, group_position=0),
        1: _make_boundary_audit(1, group_position=1),
    }
    s1 = _data_loader.BoundaryCompletionSampler((0, 1), audits, batch_size=8, seed=99)
    s2 = _data_loader.BoundaryCompletionSampler((0, 1), audits, batch_size=8, seed=99)
    assert list(s1) == list(s2)


def test_boundary_sampler_different_seed_different_order():
    audits = {0: _make_boundary_audit(0, group_position=0)}
    s1 = _data_loader.BoundaryCompletionSampler((0,), audits, batch_size=8, seed=1)
    s2 = _data_loader.BoundaryCompletionSampler((0,), audits, batch_size=8, seed=2)
    assert list(s1) != list(s2)


def test_boundary_sampler_resume_fast_forwards_and_continues():
    """set_epoch + set_skip_batches reconstructs the same epoch and skips
    already-consumed batches."""

    audits = {0: _make_boundary_audit(0, group_position=0)}
    batch_size = 4
    full_sampler = _data_loader.BoundaryCompletionSampler((0,), audits, batch_size=batch_size, seed=7)
    full_indices = list(full_sampler)

    # Simulate resume after 1 batch of epoch 0.
    resume_sampler = _data_loader.BoundaryCompletionSampler((0,), audits, batch_size=batch_size, seed=7)
    resume_sampler.set_epoch(0)
    resume_sampler.set_skip_batches(1)
    resumed = list(resume_sampler)

    # Remaining batches should match the tail of the full epoch.
    expected_tail = full_indices[batch_size:]
    assert resumed == expected_tail


def test_boundary_sampler_multi_epoch_advances():
    """Iterating twice yields two different epoch shuffles (auto-advance)."""

    audits = {0: _make_boundary_audit(0, group_position=0)}
    sampler = _data_loader.BoundaryCompletionSampler((0,), audits, batch_size=8, seed=5)
    epoch0 = list(sampler)
    epoch1 = list(sampler)
    # Same sample set, different shuffle (extremely likely with different epoch seeds).
    assert set(epoch0) == set(epoch1)
    assert epoch0 != epoch1


def test_boundary_sampler_rejects_empty_episodes():
    with pytest.raises(ValueError, match="at least one episode"):
        _data_loader.BoundaryCompletionSampler((), {}, batch_size=8, seed=42)
