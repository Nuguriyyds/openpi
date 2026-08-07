import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.training import completion
from openpi.training import optimizer


def test_completion_head_optimizer_accepts_head_only_tree_and_produces_finite_updates():
    params = {"completion_head": {"kernel": jnp.asarray([0.0, 0.0], dtype=jnp.float32)}}
    grads = {"completion_head": {"kernel": jnp.asarray([3.0, 4.0], dtype=jnp.float32)}}
    tx = optimizer.create_completion_head_optimizer(
        completion.CompletionTrainingConfig(stage="head", warmup_steps=1),
        decay_steps=10,
    )

    updates, _ = tx.update(grads, tx.init(params), params)

    assert jax.tree.all(jax.tree.map(lambda value: jnp.all(jnp.isfinite(value)), updates))
    assert set(updates) == {"completion_head"}


def test_action_and_head_stage_optimizers_have_disjoint_state_and_updates():
    action_params = {"action_in_proj": {"kernel": jnp.asarray([0.0, 0.0], dtype=jnp.float32)}}
    head_params = {"completion_head": {"kernel": jnp.asarray([0.0, 0.0], dtype=jnp.float32)}}
    action_grads = jax.tree.map(jnp.ones_like, action_params)
    head_grads = jax.tree.map(lambda value: jnp.full_like(value, 1.0e6), head_params)
    action_tx = optimizer.create_optimizer(
        optimizer.AdamW(clip_gradient_norm=1.0),
        optimizer.CosineDecaySchedule(warmup_steps=1, decay_steps=10),
    )
    head_tx = optimizer.create_completion_head_optimizer(
        completion.CompletionTrainingConfig(stage="head", warmup_steps=1, gradient_clip_norm=1.0),
        decay_steps=10,
    )

    action_updates, action_state = action_tx.update(action_grads, action_tx.init(action_params), action_params)
    head_updates, head_state = head_tx.update(head_grads, head_tx.init(head_params), head_params)

    assert set(action_updates) == {"action_in_proj"}
    assert set(head_updates) == {"completion_head"}
    assert action_state is not head_state
    np.testing.assert_allclose(
        jax.tree.leaves(action_updates)[0],
        jax.tree.leaves(action_tx.update(action_grads, action_tx.init(action_params), action_params)[0])[0],
    )


def test_completion_head_optimizer_requires_steps_after_warmup():
    with pytest.raises(ValueError, match="must exceed completion warmup_steps"):
        optimizer.create_completion_head_optimizer(
            completion.CompletionTrainingConfig(stage="head", warmup_steps=10),
            decay_steps=10,
        )
