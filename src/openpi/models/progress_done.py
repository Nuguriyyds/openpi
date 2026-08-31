"""Standalone joint progress/done head built on the token-query adapter."""

from __future__ import annotations

import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models.completion import CompletionHeadConfig
from openpi.models.completion import TokenQueryCompletionHead
from openpi.models.completion import _dropout


class TokenQueryProgressDoneHead(TokenQueryCompletionHead):
    """Shares token-query features and predicts separate done/progress logits."""

    def __init__(self, input_dim: int, config: CompletionHeadConfig, *, rngs: nnx.Rngs):
        super().__init__(input_dim, config, rngs=rngs)
        self.progress_output = nnx.Linear(
            config.hidden_dim,
            1,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(
        self,
        prefix_history: jax.Array,
        prefix_mask_history: jax.Array,
        *,
        rng: jax.Array | None = None,
        train: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        prefix_history = jax.lax.stop_gradient(jnp.asarray(prefix_history, dtype=jnp.float32))
        prefix_mask_history = jnp.asarray(prefix_mask_history, dtype=jnp.bool_)
        if prefix_history.ndim != 4 or prefix_history.shape[1] != self.temporal_steps:
            raise ValueError(f"token prefix_history must have shape [batch, 3, tokens, D], got {prefix_history.shape}")
        if prefix_history.shape[-1] != self.input_dim:
            raise ValueError(f"token prefix feature dimension must be {self.input_dim}")
        if prefix_mask_history.shape != prefix_history.shape[:3]:
            raise ValueError("token prefix mask must match [batch, time, tokens]")
        if train and self.dropout_rate and rng is None:
            raise ValueError("token-query head requires an RNG when dropout is enabled during training")

        normalized = self.token_norm(prefix_history)
        keys = self.key_projection(normalized) + self.time_embedding.value[None, :, None, :]
        values = self.value_projection(normalized) + self.time_embedding.value[None, :, None, :]
        batch_size = prefix_history.shape[0]
        keys = keys.reshape(batch_size, -1, keys.shape[-1])
        values = values.reshape(batch_size, -1, values.shape[-1])
        mask = prefix_mask_history.reshape(batch_size, -1)
        queries = jnp.broadcast_to(
            self.learned_queries.value[None, ...],
            (batch_size, self.query_count, self.learned_queries.value.shape[-1]),
        )
        scores = jnp.einsum("bqd,bkd->bqk", self.query_projection(queries), keys) / math.sqrt(keys.shape[-1])
        scores = jnp.where(mask[:, None, :], scores, jnp.finfo(jnp.float32).min)
        weights = jax.nn.softmax(scores, axis=-1)
        cross_output = self.cross_output(jnp.einsum("bqk,bkd->bqd", weights, values))
        if train and self.dropout_rate:
            rng, cross_rng = jax.random.split(rng)
            cross_output = _dropout(cross_output, cross_rng, self.dropout_rate)
        queries = self.cross_norm(queries + cross_output)
        cross_ffn = self.cross_ffn_out(jax.nn.gelu(self.cross_ffn_in(self.cross_ffn_norm(queries))))
        if train and self.dropout_rate:
            rng, cross_ffn_rng = jax.random.split(rng)
            cross_ffn = _dropout(cross_ffn, cross_ffn_rng, self.dropout_rate)
        queries = queries + cross_ffn
        block_rngs = (None,) * self.temporal_layers
        if train and self.dropout_rate:
            block_rngs = tuple(jax.random.split(rng, self.temporal_layers))
        for index, block_rng in enumerate(block_rngs):
            queries = self.temporal_blocks[f"layer_{index}"](queries, rng=block_rng, train=train)
        hidden = self.output_norm(queries[:, 0])
        return (
            jnp.asarray(self.output(hidden)[..., 0], dtype=jnp.float32),
            jnp.asarray(self.progress_output(hidden)[..., 0], dtype=jnp.float32),
        )
