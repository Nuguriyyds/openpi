import ast
import dataclasses
import inspect
import os
import pathlib
import textwrap
from types import SimpleNamespace

os.environ["JAX_PLATFORMS"] = "cpu"

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.training import completion as _completion
from openpi.training import config as _config

from . import train


def test_completion_validation_runs_inside_mesh_context():
    tree = ast.parse(textwrap.dedent(inspect.getsource(train.main)))

    def is_mesh_context(node):
        context = node.items[0].context_expr
        return (
            isinstance(context, ast.Call)
            and isinstance(context.func, ast.Attribute)
            and isinstance(context.func.value, ast.Name)
            and context.func.value.id == "sharding"
            and context.func.attr == "set_mesh"
        )

    def contains_completion_eval(node):
        return any(
            isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "evaluate_completion"
            for child in ast.walk(node)
        )

    assert any(
        isinstance(node, ast.With) and is_mesh_context(node) and contains_completion_eval(node)
        for node in ast.walk(tree)
    )


def test_s2_dtype_cast_keeps_loaded_action_params_float32():
    model_config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        completion_head=pi0_config.CompletionHeadConfig(enabled=True),
    )
    config = _config.TrainConfig(
        name="s2-dtype",
        model=model_config,
        completion=_completion.CompletionTrainingConfig(
            stage="head",
            split_manifest_path="manifest.json",
        ),
        freeze_filter=model_config.get_completion_head_only_freeze_filter(),
        num_train_steps=1_000,
    )
    model = model_config.create(jax.random.key(0))
    params = nnx.state(model)

    cast_params = nnx_utils.state_map(
        params,
        train._frozen_param_dtype_filter(config),  # noqa: SLF001
        lambda p: p.replace(p.value.astype(jnp.bfloat16)),
    )

    dtypes_by_group = {"vlm": [], "action": [], "completion": []}
    for path, variable in cast_params.filter(nnx.Param).flat_state().items():
        dtypes_by_group[pi0_config.classify_parameter_path(path)].append(variable.value.dtype)

    assert dtypes_by_group["vlm"]
    assert dtypes_by_group["action"]
    assert dtypes_by_group["completion"]
    assert set(dtypes_by_group["vlm"]) == {jnp.bfloat16}
    assert set(dtypes_by_group["action"]) == {jnp.float32}
    assert set(dtypes_by_group["completion"]) == {jnp.float32}


def test_train_step_dispatches_action_and_head_stages_separately():
    tree = ast.parse(textwrap.dedent(inspect.getsource(train.train_step)))

    head_branch = None
    action_branch = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        compare = node.test
        if (
            isinstance(compare, ast.Compare)
            and isinstance(compare.left, ast.Name)
            and compare.left.id == "completion_stage"
            and len(compare.ops) == 1
            and isinstance(compare.ops[0], ast.Eq)
            and len(compare.comparators) == 1
            and isinstance(compare.comparators[0], ast.Constant)
            and compare.comparators[0].value == "head"
        ):
            head_branch = node.body
            action_branch = node.orelse
            break

    assert head_branch is not None
    assert action_branch is not None

    def branch_calls(branch, name):
        return any(
            isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == name
            for statement in branch
            for child in ast.walk(statement)
        )

    assert branch_calls(head_branch, "compute_completion_logits")
    assert not branch_calls(head_branch, "compute_loss")
    assert branch_calls(action_branch, "compute_loss")
    assert not branch_calls(action_branch, "compute_completion_logits")


def test_joint_action_completion_api_is_not_present():
    assert not hasattr(train, "compute_action_loss_and_completion_logits")


def test_progress_validation_metrics_are_continuous_and_threshold_free():
    logits = np.asarray([-5.0, -1.0, 0.0, 1.0, 5.0], dtype=np.float32)
    targets = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

    metrics = train.progress_validation_metrics(logits, targets, huber_delta=0.1, prefix="progress_val")

    assert set(metrics) >= {
        "progress_val/loss",
        "progress_val/mae",
        "progress_val/rmse",
        "progress_val/pearson",
        "progress_val/spearman",
        "progress_val/prediction_mean",
        "progress_val/prediction_std",
        "progress_val/prediction_min",
        "progress_val/prediction_max",
        "progress_val/target_mean",
        "progress_val/target_std",
        "progress_val/target_min",
        "progress_val/target_max",
        "progress_val/early_mae",
        "progress_val/late_mae",
    }
    assert metrics["progress_val/frame_count"] == 5.0
    assert metrics["progress_val/prediction_min"] >= 0.0
    assert metrics["progress_val/prediction_max"] <= 1.0
    assert metrics["progress_val/pearson"] > 0.9
    assert metrics["progress_val/spearman"] > 0.9


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)


# ---------------------------------------------------------------------------
#  Boundary scheme: pure-function, config, and AST-guard tests
# ---------------------------------------------------------------------------

BOUNDARY_CONFIG_NAME = "pi05_agilex_breakfast_frozen_head_s2_completion_boundary"


def test_compute_epoch_total_steps_1_and_2_epochs():
    assert train.compute_epoch_total_steps(10, 1) == 10
    assert train.compute_epoch_total_steps(10, 2) == 20
    assert train.compute_epoch_total_steps(7, 1) == 7
    assert train.compute_epoch_total_steps(7, 2) == 14


def test_compute_epoch_total_steps_rejects_over_2_epochs():
    with pytest.raises(ValueError, match="epochs must be 1 or 2"):
        train.compute_epoch_total_steps(10, 3)
    with pytest.raises(ValueError, match="epochs must be 1 or 2"):
        train.compute_epoch_total_steps(10, 0)


def test_compute_epoch_total_steps_rejects_non_positive_steps_per_epoch():
    with pytest.raises(ValueError, match="steps_per_epoch must be positive"):
        train.compute_epoch_total_steps(0, 1)


def test_compute_boundary_total_steps_accepts_explicit_partial_epoch_budget():
    assert train.compute_boundary_total_steps(1385, epochs=None, train_steps=3000) == 3000
    assert train.compute_boundary_total_steps(1385, epochs=2, train_steps=None) == 2770


def test_compute_boundary_total_steps_requires_exactly_one_budget():
    with pytest.raises(ValueError, match="exactly one"):
        train.compute_boundary_total_steps(1385, epochs=None, train_steps=None)
    with pytest.raises(ValueError, match="exactly one"):
        train.compute_boundary_total_steps(1385, epochs=1, train_steps=3000)


def test_should_save_epoch_checkpoint_every_200_and_epoch_end():
    """Saves at multiples of save_interval and at epoch boundaries."""

    # 200 completed → save (multiple of 200).
    assert train.should_save_epoch_checkpoint(200, steps_per_epoch=500, total_steps=500) is True
    # 100 completed → no save.
    assert train.should_save_epoch_checkpoint(100, steps_per_epoch=500, total_steps=500) is False
    # 500 completed == total_steps → save (final step).
    assert train.should_save_epoch_checkpoint(500, steps_per_epoch=500, total_steps=500) is True
    # 500 completed == epoch boundary in a 2-epoch run → save.
    assert train.should_save_epoch_checkpoint(500, steps_per_epoch=500, total_steps=1000) is True
    # 501 → no save.
    assert train.should_save_epoch_checkpoint(501, steps_per_epoch=500, total_steps=1000) is False


def test_should_save_epoch_checkpoint_zero_or_negative_never_saves():
    assert train.should_save_epoch_checkpoint(0, steps_per_epoch=500, total_steps=500) is False
    assert train.should_save_epoch_checkpoint(-1, steps_per_epoch=500, total_steps=500) is False


def test_should_save_epoch_checkpoint_no_off_by_one():
    """The checkpoint fires at ``completed == steps_per_epoch`` (the epoch
    boundary), and the dir name will equal ``completed`` == ``train_state.step``,
    so resume starts at ``completed`` without redoing or skipping a step."""

    sp = 300
    total = sp * 2  # 2 epochs
    # At exactly the epoch boundary (300), we save — this is the last completed
    # step of epoch 0.  Resume from 300 continues with range(300, 600).
    assert train.should_save_epoch_checkpoint(sp, steps_per_epoch=sp, total_steps=total) is True
    # One step before the boundary does NOT save.
    assert train.should_save_epoch_checkpoint(sp - 1, steps_per_epoch=sp, total_steps=total) is False


def test_temporal_validation_rank_uses_metric_report_keys():
    metrics = {
        "val/temporal/boundary_top1_rate": 0.8,
        "val/temporal/hard_local/auprc": 0.7,
        "val/temporal/margin/hard_local_median": 0.2,
        "val/temporal/natural/auprc": 0.9,
    }

    assert train.temporal_validation_rank(metrics) == (0.8, 0.7, 0.2, 0.9)


def test_temporal_input_mode_keeps_shape_and_removes_only_history_slots():
    source = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(2, 3, 4)

    history = _completion.apply_temporal_input_mode(source, "history")
    current_only = _completion.apply_temporal_input_mode(source, "current_only")

    np.testing.assert_array_equal(np.asarray(history), np.asarray(source))
    assert current_only.shape == source.shape
    np.testing.assert_array_equal(np.asarray(current_only[:, :2, :]), np.zeros((2, 2, 4), dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(current_only[:, 2, :]), np.asarray(source[:, 2, :]))
    # The transform is functional; cached source features remain immutable.
    np.testing.assert_array_equal(np.asarray(source), np.arange(24, dtype=np.float32).reshape(2, 3, 4))


def test_temporal_train_and_validation_both_apply_configured_input_mode():
    assert "apply_temporal_input_mode" in inspect.getsource(train.train_step)
    assert "apply_temporal_input_mode" in inspect.getsource(train.temporal_completion_eval_step)


def test_temporal_selection_binding_seals_input_mode():
    data_info = SimpleNamespace(
        manifest=SimpleNamespace(),
        cache=SimpleNamespace(
            metadata=SimpleNamespace(
                schema_version=2,
                model_config_name="clean",
                checkpoint_path="/checkpoint/49999",
                row_count=100,
                feature_dim=2048,
            )
        ),
    )

    binding = train._temporal_cache_binding(data_info, input_mode="current_only")  # noqa: SLF001

    assert binding["temporal_input_mode"] == "current_only"
    assert binding["feature_cache_checkpoint_path"] == "/checkpoint/49999"


def test_temporal_resume_rejects_stale_existing_validation_selection():
    assert (
        train.require_current_temporal_validation_progress(
            {"last_validated_checkpoint_step": 400},
            resumed_step=400,
        )
        == 400
    )

    with pytest.raises(ValueError, match="stale for the resumed checkpoint"):
        train.require_current_temporal_validation_progress(
            {"last_validated_checkpoint_step": 200},
            resumed_step=400,
        )


def test_boundary_config_uses_plain_unweighted_bce():
    """Boundary config: focal_gamma==0 (no focal), pos_weight==1.0 (no weight)."""

    config = _config.get_config(BOUNDARY_CONFIG_NAME)
    assert config.completion.focal_gamma == 0.0
    assert config.completion.bce_pos_weight_override == 1.0
    assert config.completion.objective == "binary"
    assert config.completion.stage == "head"


def test_boundary_config_epoch_and_sampling_settings():
    config = _config.get_config(BOUNDARY_CONFIG_NAME)
    assert config.completion.epochs is None
    assert config.completion.train_steps == 3_000
    assert config.completion.eval_checkpoint_step == 3_000
    assert config.num_train_steps == 3_000
    assert config.completion.boundary_sampling is True
    assert config.completion.negative_stride == 15
    assert config.completion.boundary_copy_frames == 5
    assert config.completion.val_groups == 0


def test_boundary_config_uses_head_only_runtime_optimizations():
    config = _config.get_config(BOUNDARY_CONFIG_NAME)
    assert config.ema_decay is None
    assert config.fsdp_devices == 1
    assert config.num_workers == 16


def test_boundary_config_rejects_epochs_over_2():
    with pytest.raises(ValueError, match="epochs"):
        _completion.CompletionTrainingConfig(stage="head", boundary_sampling=True, epochs=3)


def test_boundary_config_rejects_conflicting_or_invalid_step_budgets():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _completion.CompletionTrainingConfig(
            stage="head",
            boundary_sampling=True,
            epochs=1,
            train_steps=3000,
        )
    with pytest.raises(ValueError, match="train_steps must be positive"):
        _completion.CompletionTrainingConfig(stage="head", boundary_sampling=True, train_steps=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        _completion.CompletionTrainingConfig(
            stage="head",
            boundary_sampling=True,
            train_steps=3000,
            eval_checkpoint_step=4000,
        )


def test_boundary_config_freezes_all_except_completion_head():
    """The freeze filter must be the completion-head-only filter so VLM and
    action-expert params receive zero gradient."""

    config = _config.get_config(BOUNDARY_CONFIG_NAME)
    model_config = pi0_config.Pi0Config(
        pi05=True,
        completion_head=pi0_config.CompletionHeadConfig(enabled=True),
    )
    expected_filter = model_config.get_completion_head_only_freeze_filter()
    assert config.freeze_filter == expected_filter


def test_epoch_based_training_loop_has_no_test_or_val_loader():
    """AST guard: ``train.main`` must never reference a test data loader, and
    every ``evaluate_completion`` call must be inside a branch guarded by
    ``not is_epoch_based`` so the epoch-based path never runs val/test eval."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(train.main)))

    # 1. No test loader identifier anywhere in main.
    all_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "test_data_loader" not in all_names
    assert "test_loader" not in all_names

    # 2. Every evaluate_completion call is inside an `if` whose test references
    #    `not is_epoch_based` (i.e. the epoch-based path never evaluates).
    def node_calls_evaluate_completion(node):
        return any(
            isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "evaluate_completion"
            for child in ast.walk(node)
        )

    def test_checks_not_epoch_based(if_node):
        """Returns True if the if-test contains ``not is_epoch_based``."""

        for child in ast.walk(if_node.test):
            if (
                isinstance(child, ast.UnaryOp)
                and isinstance(child.op, ast.Not)
                and isinstance(child.operand, ast.Name)
                and child.operand.id == "is_epoch_based"
            ):
                return True
        return False

    eval_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "evaluate_completion"
    ]
    assert eval_calls, "expected at least one evaluate_completion call in main"

    for _call in eval_calls:
        # Walk up: find the enclosing If and verify its guard.
        # Since ast.walk doesn't give parents, we check all If nodes that
        # contain this call and have the not-is_epoch_based guard.
        guarded = False
        for if_node in ast.walk(tree):
            if not isinstance(if_node, ast.If):
                continue
            if node_calls_evaluate_completion(if_node) and test_checks_not_epoch_based(if_node):
                guarded = True
                break
        assert guarded, "evaluate_completion call is not guarded by `not is_epoch_based`"


def test_epoch_based_wandb_log_has_no_val_or_test_keys():
    """AST guard: in the epoch-based branch, wandb.log string-literal keys must
    not start with ``val/`` or ``test/``."""

    tree = ast.parse(textwrap.dedent(inspect.getsource(train.main)))

    def is_is_epoch_based_branch(if_node):
        test = if_node.test
        return isinstance(test, ast.Name) and test.id == "is_epoch_based"

    def collect_log_string_keys(node):
        """Collect string-literal keys from wandb.log({...}) calls under node."""
        keys = set()
        for child in ast.walk(node):
            if not (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)):
                continue
            if not (
                child.func.attr == "log" and isinstance(child.func.value, ast.Name) and child.func.value.id == "wandb"
            ):
                continue
            if child.args and isinstance(child.args[0], ast.Dict):
                for key in child.args[0].keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
        return keys

    # Find the epoch-based logging branch: `if is_epoch_based:` inside the loop.
    epoch_branch_keys = set()
    for if_node in ast.walk(tree):
        if isinstance(if_node, ast.If) and is_is_epoch_based_branch(if_node):
            epoch_branch_keys |= collect_log_string_keys(if_node)

    # The epoch-based branch must log something.
    assert epoch_branch_keys, "expected wandb.log keys in the epoch-based branch"
    # No val/ or test/ keys.
    bad = {k for k in epoch_branch_keys if k.startswith(("val/", "test/"))}
    assert not bad, f"epoch-based wandb.log contains val/test keys: {bad}"


def test_epoch_based_protects_eval_checkpoint():
    """AST guard (P1-1 + P1-A): when a predeclared eval checkpoint is not the
    final step, train.main must copy the checkpoint to a protected
    ``eval_checkpoint/`` dir **with a step marker** (``_protected_step.json``),
    and at training end must assert the eval checkpoint still exists."""

    source_text = textwrap.dedent(inspect.getsource(train.main))
    tree = ast.parse(source_text)

    # 1. shutil.copytree is called (copies the managed checkpoint to a protected dir).
    copytree_calls = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "copytree"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "shutil"
        )
    ]
    assert copytree_calls, "expected shutil.copytree call for eval checkpoint protection"

    # 2. The string "eval_checkpoint" appears as a literal in main (used as the
    #    protected copy target and in the end-of-training assertion).
    eval_checkpoint_strings = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "eval_checkpoint" in node.value
    ]
    assert eval_checkpoint_strings, "expected 'eval_checkpoint' string literal in train.main"

    # 3. P1-A: A _protected_step.json marker is written inside the protected copy.
    protected_step_strings = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and "_protected_step" in node.value
    ]
    assert protected_step_strings, "expected '_protected_step.json' marker in train.main"

    # 4. The protected directory is namespaced by the actual completed step,
    # so even a stale copy cannot be resolved as a different checkpoint.
    assert '"eval_checkpoint" / str(completed)' in source_text
    assert '"eval_checkpoint" / str(eval_checkpoint_step)' in source_text

    # 5. The end-of-training assertion references the checkpoint manager
    #    (max_to_keep=1) as the deletion cause.
    assert "max_to_keep" in source_text or "checkpoint manager" in source_text.lower()
