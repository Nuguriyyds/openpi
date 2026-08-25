import jax
import jax.numpy as jnp
from jax.tree_util import DictKey

from openpi.models import model as model_api
from openpi.policies import policy_config
from openpi.training import config


def test_temporal_policy_restore_preserves_native_mixed_precision_checkpoint():
    temporal = config.get_config("pi05_agilex_breakfast_temporal_completion_head")
    legacy = config.get_config("pi05_730_breakfast_subtasks")

    temporal_dtype = policy_config._jax_checkpoint_restore_dtype(temporal)  # noqa: SLF001
    assert callable(temporal_dtype)
    assert temporal_dtype((DictKey("params"), DictKey("completion_head"), DictKey("output"))) == jnp.float32
    assert temporal_dtype((DictKey("params"), DictKey("PaliGemma"), DictKey("llm"))) == jnp.bfloat16
    assert policy_config._jax_checkpoint_restore_dtype(legacy) == jnp.bfloat16  # noqa: SLF001

    raw = config.get_config("pi05_agilex_breakfast_raw_prefix_completion_head")
    raw_dtype = policy_config._jax_checkpoint_restore_dtype(raw)  # noqa: SLF001
    assert callable(raw_dtype)
    assert raw_dtype((DictKey("params"), DictKey("completion_head"), DictKey("output"))) == jnp.float32
    assert raw_dtype((DictKey("params"), DictKey("PaliGemma"), DictKey("llm"))) == jnp.bfloat16

    restore_args = model_api._restore_args_tree(  # noqa: SLF001
        {"params": {"completion_head": {"kernel": 0}, "PaliGemma": {"kernel": 0}}},
        sharding=None,
        restore_type=jax.Array,
        dtype=temporal_dtype,
    )
    assert restore_args["params"]["completion_head"]["kernel"].dtype == jnp.float32
    assert restore_args["params"]["PaliGemma"]["kernel"].dtype == jnp.bfloat16

    constant_dtype_args = model_api._restore_args_tree(  # noqa: SLF001
        {"params": {"kernel": 0}},
        sharding=None,
        restore_type=jax.Array,
        dtype=jnp.bfloat16,
    )
    assert constant_dtype_args["params"]["kernel"].dtype == jnp.bfloat16
