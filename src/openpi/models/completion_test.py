import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def test_masked_mean_pool_is_fp32_and_ignores_padding_tokens():
    valid_tokens = jnp.asarray([[[1.0, 2.0], [3.0, 6.0]]], dtype=jnp.float16)
    padding_a = jnp.asarray([[[10.0, 20.0], [30.0, 40.0]]], dtype=jnp.float16)
    padding_b = jnp.asarray([[[-1000.0, 1000.0], [1000.0, -1000.0]]], dtype=jnp.float16)
    mask = jnp.asarray([[True, True, False, False]])

    pooled_a = completion.masked_mean_pool(jnp.concatenate([valid_tokens, padding_a], axis=1), mask)
    pooled_b = completion.masked_mean_pool(jnp.concatenate([valid_tokens, padding_b], axis=1), mask)

    assert pooled_a.dtype == jnp.float32
    np.testing.assert_allclose(pooled_a, jnp.asarray([[2.0, 4.0]], dtype=jnp.float32))
    np.testing.assert_array_equal(pooled_a, pooled_b)


def test_masked_mean_pool_all_padding_returns_zero():
    tokens = jnp.ones((2, 3, 4), dtype=jnp.float16)
    pooled = completion.masked_mean_pool(tokens, jnp.zeros((2, 3), dtype=jnp.bool_))

    np.testing.assert_array_equal(pooled, jnp.zeros((2, 4), dtype=jnp.float32))


def test_masked_mean_pool_stops_gradient_to_prefix_tokens():
    mask = jnp.ones((2, 3), dtype=jnp.bool_)
    tokens = jnp.ones((2, 3, 4), dtype=jnp.float32)

    gradient = jax.grad(lambda value: jnp.sum(completion.masked_mean_pool(value, mask)))(tokens)

    np.testing.assert_array_equal(gradient, jnp.zeros_like(tokens))


@pytest.mark.parametrize(
    ("tokens", "mask"),
    [
        (jnp.ones((2, 4)), jnp.ones((2,), dtype=jnp.bool_)),
        (jnp.ones((2, 3, 4)), jnp.ones((2, 4), dtype=jnp.bool_)),
        (jnp.ones((2, 3, 4)), jnp.ones((2, 3, 1), dtype=jnp.bool_)),
    ],
)
def test_masked_mean_pool_rejects_malformed_shapes(tokens, mask):
    with pytest.raises(ValueError, match=r"completion (tokens|mask)"):
        completion.masked_mean_pool(tokens, mask)


def test_temporal_completion_config_is_explicit_and_strict():
    config = completion.CompletionHeadConfig(enabled=True, variant="temporal_mlp")

    assert config.temporal_steps == 3
    assert config.hidden_dim == 128
    assert config.resolved_pooling == "masked_mean"
    with pytest.raises(ValueError, match="exactly three"):
        completion.CompletionHeadConfig(enabled=True, variant="temporal_mlp", temporal_steps=2)
    with pytest.raises(ValueError, match="requires pooling"):
        completion.CompletionHeadConfig(
            enabled=True,
            variant="temporal_mlp",
            pooling="masked_attention",
        )
    with pytest.raises(ValueError, match="variant='legacy_attention'"):
        completion.CompletionHead(
            4,
            completion.CompletionHeadConfig(enabled=True, variant="temporal_mlp"),
            rngs=nnx.Rngs(0),
        )


def test_temporal_completion_head_output_shape_dtype_and_shared_layer_norm():
    head = completion.TemporalCompletionHead(
        8,
        completion.CompletionHeadConfig(enabled=True, variant="temporal_mlp", hidden_dim=16),
        rngs=nnx.Rngs(0),
    )
    history = jax.random.normal(jax.random.key(1), (4, 3, 8), dtype=jnp.float16)
    logits = head(history, train=False)
    param_state = nnx.state(head, nnx.Param).flat_state()
    param_paths = ["/".join(str(part) for part in path) for path in param_state]

    assert logits.shape == (4,)
    assert logits.dtype == jnp.float32
    assert np.all(np.isfinite(logits))
    assert param_paths.count("layer_norm/scale") == 1
    assert param_paths.count("layer_norm/bias") == 1
    assert all(variable.value.dtype == jnp.float32 for variable in param_state.values())


def test_temporal_completion_head_preserves_oldest_to_newest_concat_order():
    head = completion.TemporalCompletionHead(
        2,
        completion.CompletionHeadConfig(
            enabled=True,
            variant="temporal_mlp",
            hidden_dim=1,
            dropout_rate=0.0,
        ),
        rngs=nnx.Rngs(0),
    )
    head.projection.kernel.value = jnp.arange(1.0, 7.0, dtype=jnp.float32)[:, None]
    head.projection.bias.value = jnp.zeros((1,), dtype=jnp.float32)
    head.output.kernel.value = jnp.ones((1, 1), dtype=jnp.float32)
    head.output.bias.value = jnp.zeros((1,), dtype=jnp.float32)
    history = jnp.asarray([[[1.0, 3.0], [5.0, 2.0], [-4.0, 8.0]]], dtype=jnp.float32)

    normalized = head.layer_norm(history)
    expected_concat = jnp.concatenate([normalized[:, 0], normalized[:, 1], normalized[:, 2]], axis=-1)
    expected = jax.nn.gelu(expected_concat @ head.projection.kernel.value)[:, 0]

    np.testing.assert_allclose(head(history, train=False), expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", [(2, 8), (2, 2, 8), (2, 3, 7), (2, 4, 8)])
def test_temporal_completion_head_rejects_malformed_history(shape):
    head = completion.TemporalCompletionHead(
        8,
        completion.CompletionHeadConfig(enabled=True, variant="temporal_mlp"),
        rngs=nnx.Rngs(0),
    )

    with pytest.raises(ValueError, match="prefix_history"):
        head(jnp.ones(shape), train=False)


def test_temporal_completion_head_stops_gradient_to_history():
    head = completion.TemporalCompletionHead(
        4,
        completion.CompletionHeadConfig(
            enabled=True,
            variant="temporal_mlp",
            dropout_rate=0.0,
        ),
        rngs=nnx.Rngs(0),
    )
    history = jnp.ones((2, 3, 4), dtype=jnp.float32)

    gradient = jax.grad(lambda value: jnp.sum(head(value, train=False)))(history)

    np.testing.assert_array_equal(gradient, jnp.zeros_like(history))


def test_token_query_head_output_mask_and_gradient_contract():
    head = completion.TokenQueryCompletionHead(
        8,
        completion.CompletionHeadConfig(
            enabled=True,
            variant="token_query_attention",
            hidden_dim=12,
            dropout_rate=0.0,
        ),
        rngs=nnx.Rngs(0),
    )
    history = jax.random.normal(jax.random.key(1), (2, 3, 5, 8), dtype=jnp.float16)
    mask = jnp.asarray([[[1, 1, 1, 0, 0]] * 3, [[1, 1, 1, 1, 0]] * 3], dtype=jnp.bool_)

    logits = head(history, mask, train=False)
    gradient = jax.grad(lambda value: jnp.sum(head(value, mask, train=False)))(history)

    assert logits.shape == (2,)
    assert logits.dtype == jnp.float32
    assert np.all(np.isfinite(logits))
    np.testing.assert_array_equal(gradient, jnp.zeros_like(history))


def test_token_query_head_parameter_count_matches_qwen_scale():
    head = completion.TokenQueryCompletionHead(
        2048,
        completion.CompletionHeadConfig(
            enabled=True,
            variant="token_query_attention",
            hidden_dim=768,
        ),
        rngs=nnx.Rngs(0),
    )
    parameter_count = sum(
        int(np.prod(variable.value.shape)) for variable in nnx.state(head, nnx.Param).flat_state().values()
    )

    assert 30_000_000 <= parameter_count <= 33_000_000


def test_token_query_head_supports_fewer_queries_and_layers():
    head = completion.TokenQueryCompletionHead(
        2048,
        completion.CompletionHeadConfig(
            enabled=True,
            variant="token_query_attention",
            hidden_dim=512,
            query_count=8,
            attention_heads=8,
            temporal_layers=2,
        ),
        rngs=nnx.Rngs(0),
    )
    parameter_count = sum(
        int(np.prod(variable.value.shape)) for variable in nnx.state(head, nnx.Param).flat_state().values()
    )

    assert head.learned_queries.value.shape == (8, 512)
    assert tuple(head.temporal_blocks) == ("layer_0", "layer_1")
    assert parameter_count == 11_041_281


def test_temporal_completion_dropout_requires_rng_only_during_training():
    head = completion.TemporalCompletionHead(
        4,
        completion.CompletionHeadConfig(enabled=True, variant="temporal_mlp", dropout_rate=0.5),
        rngs=nnx.Rngs(0),
    )
    history = jnp.ones((2, 3, 4), dtype=jnp.float32)

    np.testing.assert_array_equal(head(history, train=False), head(history, train=False))
    with pytest.raises(ValueError, match="requires an RNG"):
        head(history, train=True)


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
    from openpi.policies import agilex_policy

    actions = np.zeros((3, 32), dtype=np.float32)

    output = agilex_policy.AgileXOutputs()({"actions": actions})

    assert output["actions"].shape == (3, 14)


def optax_global_norm(tree) -> float:
    leaves = [jnp.asarray(leaf) for leaf in jax.tree.leaves(tree)]
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)))
