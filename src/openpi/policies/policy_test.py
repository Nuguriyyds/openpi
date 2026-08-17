import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi.policies import aloha_policy
from openpi.policies import policy as policy_module
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)


class _FakeTemporalModel(nnx.Module):
    def sample_actions(self, rng, observation):
        del rng, observation
        return jnp.zeros((1, 1, 1), dtype=jnp.float32)

    def compute_temporal_completion_logits(self, rng, prefix_history, *, train=False):
        del rng
        # This intentionally uses a Python branch.  Passing ``train`` as a
        # dynamic jitted argument would reproduce TracerBoolConversionError.
        if train:
            raise AssertionError("completion policy scoring must use eval mode")
        return jnp.sum(prefix_history, axis=(1, 2))


def test_temporal_scoring_is_jittable_and_does_not_advance_action_rng():
    action_rng = jax.random.key(123)
    policy = policy_module.Policy(_FakeTemporalModel(), rng=action_rng)
    history = np.ones((3, 4), dtype=np.float32)

    before = np.asarray(jax.random.key_data(policy._rng)).copy()  # noqa: SLF001
    logit = policy.score_temporal_completion(history, return_logit=True)
    after = np.asarray(jax.random.key_data(policy._rng)).copy()  # noqa: SLF001

    assert logit == 12.0
    np.testing.assert_array_equal(after, before)
    assert policy.score_temporal_completion(history) == float(jax.nn.sigmoid(jnp.float32(12.0)))
