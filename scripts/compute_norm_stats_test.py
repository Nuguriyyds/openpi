import types

import numpy as np
import pytest

from openpi.shared import normalize
from openpi.training import completion
from openpi.training import config as _config
from openpi.training import data_loader

from . import compute_norm_stats


class _DataFactory:
    def __init__(self, data_config, assets_dir):
        self._data_config = data_config
        self._assets_dir = assets_dir

    def create(self, assets_dirs, model):
        del assets_dirs, model
        return self._data_config

    def resolve_assets_dir(self, default_assets_dir):
        del default_assets_dir
        return self._assets_dir


def test_completion_norm_stats_main_uses_train_split_and_configured_asset_path(tmp_path, monkeypatch):
    raw_data_config = _config.DataConfig(repo_id="org/breakfast", asset_id="breakfast-assets")
    external_assets = tmp_path / "external-assets"
    factory = _DataFactory(raw_data_config, external_assets)
    model = types.SimpleNamespace(completion_head=types.SimpleNamespace(enabled=False), action_horizon=2)
    train_episode_ids = (8, 9, 10, 11)
    manifest = types.SimpleNamespace(episode_ids=lambda split: train_episode_ids if split == "train" else ())
    completion_info = types.SimpleNamespace(manifest=manifest)
    config = types.SimpleNamespace(
        data=factory,
        assets_dirs=tmp_path / "default-assets",
        model=model,
        completion=completion.CompletionTrainingConfig(
            stage="action",
            split_manifest_path="manifest.json",
        ),
        batch_size=2,
        num_workers=0,
    )
    captured = {}

    def create_dataloader(data_config, action_horizon, batch_size, model_config, num_workers, max_frames):
        del action_horizon, batch_size, model_config, num_workers, max_frames
        captured["episodes"] = data_config.episodes
        return [
            {
                "state": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "actions": np.asarray([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
            }
        ], 1

    monkeypatch.setattr(_config, "get_config", lambda _: config)
    monkeypatch.setattr(data_loader, "prepare_completion_data", lambda _: completion_info)
    monkeypatch.setattr(compute_norm_stats, "create_torch_dataloader", create_dataloader)

    compute_norm_stats.main("completion")

    output_path = external_assets / "breakfast-assets"
    saved_stats = normalize.load(output_path)
    assert captured["episodes"] == train_episode_ids
    assert (output_path / "norm_stats.json").is_file()
    assert not (config.assets_dirs / raw_data_config.repo_id / "norm_stats.json").exists()
    np.testing.assert_allclose(saved_stats["state"].mean, [2.0, 3.0])
    np.testing.assert_allclose(saved_stats["actions"].mean, [4.0, 6.0])


def test_norm_stats_output_rejects_remote_asset_path(tmp_path):
    data_config = _config.DataConfig(repo_id="org/data", asset_id="data-assets")
    factory = _DataFactory(data_config, "gs://bucket/assets")
    config = types.SimpleNamespace(data=factory, assets_dirs=tmp_path, model=types.SimpleNamespace())

    with pytest.raises(ValueError, match="only be written to a local assets_dir"):
        compute_norm_stats.norm_stats_output_path(config, data_config)
