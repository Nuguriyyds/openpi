import ast
import dataclasses
import inspect
import os
import pathlib
import textwrap

os.environ["JAX_PLATFORMS"] = "cpu"

import flax.nnx as nnx
import jax
import jax.numpy as jnp
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
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == name
            for statement in branch
            for child in ast.walk(statement)
        )

    assert branch_calls(head_branch, "compute_completion_logits")
    assert not branch_calls(head_branch, "compute_loss")
    assert branch_calls(action_branch, "compute_loss")
    assert not branch_calls(action_branch, "compute_completion_logits")


def test_joint_action_completion_api_is_not_present():
    assert not hasattr(train, "compute_action_loss_and_completion_logits")


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
