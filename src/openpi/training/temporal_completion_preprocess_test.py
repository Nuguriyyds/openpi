from __future__ import annotations

import dataclasses
import hashlib
from types import SimpleNamespace

from openpi.training import temporal_completion_preprocess as preprocess


@dataclasses.dataclass(frozen=True)
class _DataConfig:
    repo_id: str


@dataclasses.dataclass(frozen=True)
class _ModelConfig:
    name: str


@dataclasses.dataclass(frozen=True)
class _DataFactory:
    repo_id: str

    def create(self, assets_dirs, model_config):
        del assets_dirs, model_config
        return _DataConfig(self.repo_id)


def _source_config():
    return SimpleNamespace(
        data=_DataFactory("breakfast"),
        assets_dirs="unused",
        model=_ModelConfig(name="clean-pi05"),
    )


def test_expected_preprocess_fingerprint_is_shared_and_binds_prompts_assets_and_contract(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    assets = checkpoint / "assets"
    assets.mkdir(parents=True)
    asset = assets / "norm.json"
    asset.write_text("v1", encoding="utf-8")
    code_hash = hashlib.sha256(b"implementation").hexdigest()
    monkeypatch.setattr(preprocess, "implementation_fingerprint", lambda repo_root=None: code_hash)
    manifest = SimpleNamespace(task_prompts=("task 0", "task 1", "task 2", "task 3"))

    first = preprocess.expected_preprocess_fingerprint(
        source_train_config=_source_config(),
        manifest=manifest,
        checkpoint_path=checkpoint,
    )
    assert first == preprocess.expected_preprocess_fingerprint(
        source_train_config=_source_config(),
        manifest=manifest,
        checkpoint_path=checkpoint,
    )

    asset.write_text("v2", encoding="utf-8")
    changed_asset = preprocess.expected_preprocess_fingerprint(
        source_train_config=_source_config(),
        manifest=manifest,
        checkpoint_path=checkpoint,
    )
    changed_prompt = preprocess.expected_preprocess_fingerprint(
        source_train_config=_source_config(),
        manifest=SimpleNamespace(task_prompts=("changed", "task 1", "task 2", "task 3")),
        checkpoint_path=checkpoint,
    )

    assert first != changed_asset
    assert changed_asset != changed_prompt
    assert preprocess.describe_preprocess_contract() == '{"pooling_method":"masked_mean_fp32","protocol_version":1}'
