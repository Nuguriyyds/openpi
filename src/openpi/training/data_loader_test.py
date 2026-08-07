import dataclasses
import types

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_config
from openpi.training import completion
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
