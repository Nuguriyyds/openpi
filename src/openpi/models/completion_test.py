import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import completion
from openpi.training import completion as training_completion


def test_completion_head_output_shape_and_finite_logits():
    head = completion.CompletionHead(
        8,
        completion.CompletionHeadConfig(enabled=True),
        rngs=nnx.Rngs(0),
    )
    tokens = jax.random.normal(jax.random.key(1), (3, 5, 8), dtype=jnp.float32)
    mask = jnp.ones((3, 5), dtype=jnp.bool_)
    logits = head(tokens, mask, train=False)

    assert logits.shape == (3,)
    assert logits.dtype == jnp.float32
    assert np.all(np.isfinite(logits))
    assert all(variable.value.dtype == jnp.float32 for variable in nnx.state(head, nnx.Param).flat_state().values())


def test_masked_attention_pool_ignores_padding_tokens():
    valid_tokens = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]]])
    padding_a = jnp.asarray([[[10.0, 20.0], [30.0, 40.0]]])
    padding_b = jnp.asarray([[[-1.0e6, 1.0e6], [1.0e7, -1.0e7]]])
    mask = jnp.asarray([[True, True, False, False]])
    query = jnp.asarray([0.25, -0.5])

    pooled_a = completion.masked_attention_pool(jnp.concatenate([valid_tokens, padding_a], axis=1), mask, query)
    pooled_b = completion.masked_attention_pool(jnp.concatenate([valid_tokens, padding_b], axis=1), mask, query)

    np.testing.assert_allclose(pooled_a, pooled_b, rtol=0.0, atol=0.0)


def test_logits_and_weighted_loss_are_finite_for_extreme_values():
    logits = jnp.asarray([-1.0e6, 1.0e6, 0.0], dtype=jnp.float32)
    targets = jnp.asarray([1.0, 0.0, 1.0], dtype=jnp.float32)
    loss = training_completion.weighted_bce_with_logits(logits, targets, pos_weight=50.0)

    assert np.all(np.isfinite(loss))


def test_progress_huber_loss_matches_sigmoid_prediction_formula():
    logits = jnp.asarray([0.0, 0.0], dtype=jnp.float32)
    targets = jnp.asarray([0.4, 0.0], dtype=jnp.float32)

    loss = training_completion.progress_huber_loss(logits, targets, delta=0.1)

    np.testing.assert_allclose(loss, jnp.asarray([0.005, 0.045], dtype=jnp.float32), rtol=1e-6, atol=1e-6)


def test_progress_predictions_are_float32_and_bounded():
    predictions = training_completion.progress_predictions_from_logits(
        jnp.asarray([-1.0e6, 0.0, 1.0e6], dtype=jnp.float32)
    )

    assert predictions.dtype == jnp.float32
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)
    np.testing.assert_array_equal(predictions, jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float32))


def test_completion_head_stops_gradient_to_prefix():
    head = completion.CompletionHead(
        4,
        completion.CompletionHeadConfig(enabled=True, dropout_rate=0.0),
        rngs=nnx.Rngs(0),
    )
    mask = jnp.ones((2, 3), dtype=jnp.bool_)

    def loss(prefix):
        return jnp.sum(head(prefix, mask, train=False))

    prefix = jnp.ones((2, 3, 4), dtype=jnp.float32)
    np.testing.assert_array_equal(jax.grad(loss)(prefix), jnp.zeros_like(prefix))


class _ToyBranches(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.vlm = nnx.Linear(4, 4, rngs=rngs)
        self.action_expert = nnx.Linear(4, 4, rngs=rngs)
        self.completion_head = completion.CompletionHead(
            4,
            completion.CompletionHeadConfig(enabled=True, projection_dim=4, dropout_rate=0.0),
            rngs=rngs,
        )

    def action_loss(self, inputs):
        return jnp.sum(jnp.square(self.action_expert(inputs)))

    def completion_loss(self, prefix, mask):
        prefix_out = self.vlm(prefix)
        logits = self.completion_head(prefix_out, mask, train=False)
        targets = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
        return jnp.sum(training_completion.progress_huber_loss(logits, targets, delta=0.1))


def test_progress_loss_has_zero_vlm_and_action_expert_gradient_but_updates_head():
    model = _ToyBranches(nnx.Rngs(0))
    prefix = jax.random.normal(jax.random.key(1), (2, 3, 4), dtype=jnp.float32)
    mask = jnp.ones((2, 3), dtype=jnp.bool_)
    grads = nnx.grad(lambda module: module.completion_loss(prefix, mask))(model)

    assert optax_global_norm(grads.vlm) == 0.0
    assert optax_global_norm(grads.action_expert) == 0.0
    assert optax_global_norm(grads.completion_head) > 0.0


def test_action_loss_has_zero_completion_head_gradient():
    model = _ToyBranches(nnx.Rngs(0))
    inputs = jnp.ones((2, 4), dtype=jnp.float32)
    grads = nnx.grad(lambda module: module.action_loss(inputs))(model)

    assert optax_global_norm(grads.action_expert) > 0.0
    assert optax_global_norm(grads.vlm) == 0.0
    assert optax_global_norm(grads.completion_head) == 0.0


def test_agilex_action_output_dimension_stays_fourteen():
    from openpi.policies import agilex_policy  # noqa: PLC0415

    actions = np.zeros((3, 32), dtype=np.float32)

    output = agilex_policy.AgileXOutputs()({"actions": actions})

    assert output["actions"].shape == (3, 14)


def optax_global_norm(tree) -> float:
    leaves = [jnp.asarray(leaf) for leaf in jax.tree.leaves(tree)]
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)))
