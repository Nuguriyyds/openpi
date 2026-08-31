import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from openpi.models import completion
from openpi.models import progress_done


def test_token_query_progress_done_head_returns_two_logits_without_prefix_gradients():
    head = progress_done.TokenQueryProgressDoneHead(
        8,
        completion.CompletionHeadConfig(
            enabled=True,
            variant="token_query_attention",
            hidden_dim=12,
            dropout_rate=0.0,
        ),
        rngs=nnx.Rngs(0),
    )
    history = jnp.ones((2, 3, 5, 8), dtype=jnp.float16)
    mask = jnp.ones((2, 3, 5), dtype=jnp.bool_)

    done_logits, progress_logits = head(history, mask, train=False)

    assert done_logits.shape == progress_logits.shape == (2,)
    assert done_logits.dtype == progress_logits.dtype == jnp.float32
    assert np.all(np.isfinite(done_logits))
    assert np.all(np.isfinite(progress_logits))
