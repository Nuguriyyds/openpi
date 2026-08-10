import flax.nnx as nnx
import pytest

from openpi.models import pi0_config
from openpi.training import completion as completion_training
from openpi.training import config


def test_existing_ttrtc_config_keeps_completion_disabled():
    existing = config.get_config("pi05_agilex_empty_the_box_all_470_ttrtc")

    assert existing.training_time_rtc.enabled
    assert not existing.model.completion_head.enabled
    assert existing.completion.stage == "disabled"
    assert isinstance(existing.freeze_filter, nnx.Nothing)
    assert existing.data.repo_id == "modanqing/agilex_empty_the_box_all_470"


def test_breakfast_two_stage_configs_are_explicit_and_share_the_split():
    s1 = config.get_config("pi05_agilex_breakfast_ttrtc_s1_action")
    s2 = config.get_config("pi05_agilex_breakfast_ttrtc_s2_completion_head")

    assert s1.training_time_rtc.enabled
    assert s2.training_time_rtc.enabled
    assert s1.completion.stage == "action"
    assert not s1.model.completion_head.enabled
    assert s2.completion.stage == "head"
    assert s2.model.completion_head.enabled
    assert s1.completion.label_key == s2.completion.label_key == "completion"
    assert s1.completion.split_manifest_path == s2.completion.split_manifest_path
    assert s1.completion.split_seed == s2.completion.split_seed == 42
    assert s1.completion.val_groups == s2.completion.val_groups == 5
    assert s1.completion.test_groups == s2.completion.test_groups == 5
    assert s1.data.repo_id == s2.data.repo_id
    assert s1.data.assets.assets_dir == s2.data.assets.assets_dir
    assert s1.data.assets.asset_id == s2.data.assets.asset_id
    assert s1.data.base_config.lerobot_home == s2.data.base_config.lerobot_home
    assert s1.num_train_steps == s2.num_train_steps == 50_000
    assert s2.completion.warmup_steps == 500
    assert s2.completion.peak_lr == 1e-4
    assert s2.completion.decay_lr == 1e-5
    assert s2.completion.weight_decay == 1e-4
    assert s2.weight_loader.params_path == "/path/to/s1_checkpoint/params"
    assert s2.weight_loader.missing_regex == r"completion_head/.*"
    assert s1.data.repo_id != config.get_config("pi05_agilex_empty_the_box_all_470_ttrtc").data.repo_id
    assert not s1.data.repo_id.startswith("/")
    assert not s1.data.assets.asset_id.startswith("/")
    assert not s2.data.repo_id.startswith("/")
    assert not s2.data.assets.asset_id.startswith("/")


def test_completion_overfit_config_uses_balanced_unweighted_bce():
    overfit = config.get_config("pi05_agilex_breakfast_frozen_head_s2_completion_overfit")

    assert overfit.completion.stage == "head"
    assert overfit.completion.balanced_sampling
    assert overfit.completion.train_episode_limit == 20
    assert overfit.completion.bce_pos_weight_override == 1.0
    assert not overfit.completion.uses_focal_loss
    assert overfit.batch_size == 64
    assert overfit.ema_decay is None


@pytest.mark.parametrize(
    ("stage", "head_enabled", "pi05", "error"),
    [
        ("disabled", True, True, "must be disabled"),
        ("action", True, True, "must be disabled"),
        ("head", False, True, "must be enabled"),
        ("action", False, False, "only supported for pi0.5"),
    ],
)
def test_train_config_rejects_incoherent_completion_stages(stage, head_enabled, pi05, error):
    with pytest.raises(ValueError, match=error):
        config.TrainConfig(
            name="invalid-stage",
            model=pi0_config.Pi0Config(
                pi05=pi05,
                completion_head=pi0_config.CompletionHeadConfig(enabled=head_enabled),
            ),
            completion=completion_training.CompletionTrainingConfig(
                stage=stage,
                split_manifest_path="manifest.json",
            ),
        )


def test_staged_completion_requires_manifest():
    with pytest.raises(ValueError, match="split_manifest_path"):
        config.TrainConfig(
            name="missing-manifest",
            model=pi0_config.Pi0Config(pi05=True),
            completion=completion_training.CompletionTrainingConfig(stage="action"),
        )


def test_completion_training_requires_validation_groups():
    with pytest.raises(ValueError, match="val_groups must be positive"):
        completion_training.CompletionTrainingConfig(val_groups=0)

    assert completion_training.CompletionTrainingConfig(test_groups=0).test_groups == 0
