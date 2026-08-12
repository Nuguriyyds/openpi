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


def test_binary_breakfast_two_stage_configs_are_explicit_and_share_the_split():
    s1 = config.get_config("pi05_agilex_breakfast_frozen_head_s1_action")
    s2 = config.get_config("pi05_agilex_breakfast_frozen_head_s2_completion_head")

    assert s1.training_time_rtc.enabled
    assert s2.training_time_rtc.enabled
    assert s1.completion.stage == "action"
    assert not s1.model.completion_head.enabled
    assert s2.completion.stage == "head"
    assert s2.model.completion_head.enabled
    assert s1.completion.label_key == s2.completion.label_key == "completion"
    assert s1.completion.split_manifest_path == s2.completion.split_manifest_path
    assert s1.completion.split_manifest_repo_id == s2.completion.split_manifest_repo_id == s2.data.repo_id
    assert s1.completion.split_seed == s2.completion.split_seed == 42
    assert s1.completion.val_groups == s2.completion.val_groups == 5
    assert s1.completion.test_groups == s2.completion.test_groups == 5
    assert s1.data.repo_id == "modanqing/agilex_make_breakfast_subtask_730"
    assert s2.data.repo_id == "agilex_make_breakfast_subtask_730_frozen_head"
    assert s1.data.assets.assets_dir == s2.data.assets.assets_dir
    assert s1.data.assets.asset_id == s2.data.assets.asset_id
    assert s1.data.base_config.lerobot_home == "/mnt/data/dataset/ei/huggingface"
    assert s2.data.base_config.lerobot_home == "/mnt/data/models/wyt/data"
    assert s1.num_train_steps == 50_000
    assert s2.num_train_steps == 2_000
    assert s2.completion.warmup_steps == 50
    assert s2.completion.peak_lr == 3e-5
    assert s2.completion.decay_lr == 3e-6
    assert s2.completion.weight_decay == 1e-4
    assert s2.weight_loader.params_path.endswith("pi05_agilex_breakfast_frozen_head_s1_action/s1_action/49999/params")
    assert s2.weight_loader.missing_regex == r"completion_head/.*"
    assert s1.data.repo_id != config.get_config("pi05_agilex_empty_the_box_all_470_ttrtc").data.repo_id


def test_completion_overfit_config_uses_balanced_unweighted_bce():
    overfit = config.get_config("pi05_agilex_breakfast_frozen_head_s2_completion_overfit")

    assert overfit.completion.stage == "head"
    assert overfit.completion.balanced_sampling
    assert overfit.completion.train_episode_limit == 20
    assert overfit.completion.bce_pos_weight_override == 1.0
    assert not overfit.completion.uses_focal_loss
    assert overfit.batch_size == 64
    assert overfit.ema_decay is None


def test_progress_configs_use_frozen_head_huber_and_stratified_sampling():
    full = config.get_config("pi05_agilex_breakfast_frozen_head_s2_progress_head")
    overfit = config.get_config("pi05_agilex_breakfast_frozen_head_s2_progress_overfit")
    binary = config.get_config("pi05_agilex_breakfast_frozen_head_s2_completion_head")

    for progress_config in (full, overfit):
        assert progress_config.completion.stage == "head"
        assert progress_config.completion.objective == "progress"
        assert progress_config.completion.label_key == "progress"
        assert progress_config.completion.uses_progress_objective
        assert progress_config.completion.uses_progress_stratified_sampling
        assert not progress_config.completion.balanced_sampling
        assert not progress_config.completion.uses_focal_loss
        assert progress_config.completion.bce_pos_weight_override is None
        assert progress_config.model.completion_head.enabled
        assert progress_config.model.completion_head.dropout_rate == 0.0
        assert isinstance(progress_config.freeze_filter, pi0_config.FreezeAllExceptCompletionFilter)
        assert progress_config.model.action_dim == binary.model.action_dim
        assert progress_config.training_time_rtc == binary.training_time_rtc
        assert progress_config.completion.split_manifest_path == binary.completion.split_manifest_path
        assert progress_config.completion.split_manifest_repo_id == binary.completion.split_manifest_repo_id
        assert progress_config.weight_loader.params_path == binary.weight_loader.params_path
        assert progress_config.weight_loader.missing_regex == r"completion_head/.*"

    assert full.batch_size == overfit.batch_size == 64
    assert overfit.completion.train_episode_limit == 20
    assert full.completion.train_episode_limit is None

    # The overfit diagnostic intentionally diverges from the full run: a
    # larger huber_delta keeps gradients from saturating on the typically
    # large errors seen when memorizing a tiny 20-episode set, a higher
    # peak/decay LR and more steps give it room to converge, and zero weight
    # decay removes regularization that would otherwise fight overfitting on
    # purpose.
    assert full.completion.huber_delta == 0.1
    assert overfit.completion.huber_delta == 0.5
    assert overfit.completion.weight_decay == 0.0
    assert overfit.completion.peak_lr == 1e-4
    assert overfit.completion.decay_lr == 1e-5
    assert overfit.num_train_steps == 2_000


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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"focal_gamma": 1.0}, "focal loss"),
        ({"balanced_sampling": True}, "progress-stratified"),
        ({"bce_pos_weight_override": 1.0}, "bce_pos_weight_override"),
    ],
)
def test_progress_objective_rejects_binary_loss_and_sampler_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        completion_training.CompletionTrainingConfig(
            stage="head",
            objective="progress",
            **kwargs,
        )
