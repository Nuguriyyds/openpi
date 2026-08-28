"""Frozen-prefix completion head used by pi0.5 training."""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp

CompletionHeadVariant = Literal[
    "legacy_attention",
    "temporal_mlp",
    "token_query_attention",
]
CompletionPooling = Literal["masked_attention", "masked_mean"]


@dataclasses.dataclass(frozen=True)
class CompletionHeadConfig:
    """Configuration for the optional VLM completion prediction head.

    ``input_dim`` deliberately does not live in this config. It is read from
    the selected PaliGemma model configuration when the model is constructed.
    """

    enabled: bool = False
    variant: CompletionHeadVariant = "legacy_attention"
    pooling: CompletionPooling | None = None
    projection_dim: int = 256
    temporal_steps: int = 3
    hidden_dim: int = 128
    query_count: int = 32
    attention_heads: int = 12
    temporal_layers: int = 3
    dropout_rate: float = 0.1

    def __post_init__(self) -> None:
        if self.variant not in ("legacy_attention", "temporal_mlp", "token_query_attention"):
            raise ValueError(f"unknown completion_head.variant: {self.variant!r}")
        if self.pooling not in (None, "masked_attention", "masked_mean"):
            raise ValueError(f"unknown completion_head.pooling: {self.pooling!r}")
        if self.projection_dim <= 0:
            raise ValueError("completion_head.projection_dim must be positive")
        if self.temporal_steps <= 0:
            raise ValueError("completion_head.temporal_steps must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("completion_head.hidden_dim must be positive")
        if self.query_count <= 0:
            raise ValueError("completion_head.query_count must be positive")
        if self.attention_heads <= 0:
            raise ValueError("completion_head.attention_heads must be positive")
        if self.temporal_layers <= 0:
            raise ValueError("completion_head.temporal_layers must be positive")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("completion_head.dropout_rate must be in [0, 1)")
        if self.variant in ("temporal_mlp", "token_query_attention") and self.temporal_steps != 3:
            raise ValueError("temporal completion requires exactly three prefix time steps")

        expected_pooling = (
            "masked_mean"
            if self.variant in ("temporal_mlp", "token_query_attention")
            else "masked_attention"
        )
        if self.pooling is not None and self.pooling != expected_pooling:
            raise ValueError(
                f"completion_head variant {self.variant!r} requires pooling={expected_pooling!r}, got {self.pooling!r}"
            )

    @property
    def resolved_pooling(self) -> CompletionPooling:
        """Returns the pooling operation implied by the explicit head variant."""

        return (
            "masked_mean"
            if self.variant in ("temporal_mlp", "token_query_attention")
            else "masked_attention"
        )


def masked_mean_pool(tokens: jax.Array, mask: jax.Array) -> jax.Array:
    """FP32 masked mean of frozen prefix tokens.

    The stop-gradient is intentional: cached features and features produced
    directly from the VLM have identical completion-training semantics.
    """

    tokens = jax.lax.stop_gradient(jnp.asarray(tokens, dtype=jnp.float32))
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    if tokens.ndim != 3:
        raise ValueError(f"completion tokens must have rank 3, got shape {tokens.shape}")
    if mask.ndim != 2 or mask.shape != tokens.shape[:2]:
        raise ValueError(f"completion mask shape {mask.shape} does not match token shape {tokens.shape[:2]}")
    if tokens.shape[-1] <= 0:
        raise ValueError("completion tokens must have a non-empty hidden dimension")

    valid_tokens = jnp.where(mask[..., None], tokens, jnp.zeros((), dtype=jnp.float32))
    valid_count = jnp.sum(mask, axis=1, keepdims=True, dtype=jnp.float32)
    return jnp.sum(valid_tokens, axis=1, dtype=jnp.float32) / jnp.maximum(valid_count, 1.0)


def masked_attention_pool(
    tokens: jax.Array,
    mask: jax.Array,
    query: jax.Array,
) -> jax.Array:
    """Pools tokens with a learned query while assigning padding zero weight."""

    tokens = jnp.asarray(tokens, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    query = jnp.asarray(query, dtype=jnp.float32)
    if tokens.ndim != 3:
        raise ValueError(f"completion tokens must have rank 3, got shape {tokens.shape}")
    if mask.shape != tokens.shape[:2]:
        raise ValueError(f"completion mask shape {mask.shape} does not match token shape {tokens.shape[:2]}")
    if query.shape != (tokens.shape[-1],):
        raise ValueError(
            f"completion attention query shape {query.shape} does not match hidden size {tokens.shape[-1]}"
        )

    scores = jnp.einsum("btd,d->bt", tokens, query) / math.sqrt(tokens.shape[-1])
    # Avoid a softmax over -inf for a malformed all-padding example. The zero
    # denominator fallback yields a finite all-zero pooled vector; the normal
    # data path always has at least one valid image or prompt token.
    masked_scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    score_max = jnp.max(masked_scores, axis=-1, keepdims=True)
    unnormalized = jnp.exp(masked_scores - score_max) * mask.astype(jnp.float32)
    weights = unnormalized / jnp.maximum(jnp.sum(unnormalized, axis=-1, keepdims=True), 1.0e-8)
    return jnp.einsum("bt,btd->bd", weights, tokens)


class CompletionHead(nnx.Module):
    """FP32 learned-attention head over frozen VLM prefix outputs."""

    def __init__(self, input_dim: int, config: CompletionHeadConfig, *, rngs: nnx.Rngs):
        if input_dim <= 0:
            raise ValueError("completion head input_dim must be positive")
        if config.variant != "legacy_attention":
            raise ValueError("CompletionHead requires completion_head.variant='legacy_attention'")
        if config.resolved_pooling != "masked_attention":
            raise ValueError("CompletionHead requires masked_attention pooling")
        self.input_dim = input_dim
        self.dropout_rate = config.dropout_rate
        query_key = rngs.params()
        self.attention_query = nnx.Param(
            jax.random.normal(query_key, (input_dim,), dtype=jnp.float32) / math.sqrt(input_dim)
        )
        self.layer_norm = nnx.LayerNorm(
            input_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.projection = nnx.Linear(
            input_dim,
            config.projection_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.output = nnx.Linear(
            config.projection_dim,
            1,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(
        self,
        prefix_out: jax.Array,
        prefix_mask: jax.Array,
        *,
        rng: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        # The completion objective must never update the frozen VLM or the
        # action expert through the shared prefix representation.
        prefix_out = jax.lax.stop_gradient(jnp.asarray(prefix_out, dtype=jnp.float32))
        pooled = masked_attention_pool(prefix_out, prefix_mask, self.attention_query.value)
        hidden = self.layer_norm(pooled)
        hidden = self.projection(hidden)
        hidden = jax.nn.gelu(hidden)
        if train and self.dropout_rate:
            if rng is None:
                raise ValueError("completion head requires an RNG when dropout is enabled during training")
            keep_probability = 1.0 - self.dropout_rate
            keep = jax.random.bernoulli(rng, keep_probability, hidden.shape)
            hidden = jnp.where(keep, hidden / keep_probability, 0.0)
        logits = self.output(hidden)
        return jnp.asarray(logits[..., 0], dtype=jnp.float32)


class TemporalCompletionHead(nnx.Module):
    """FP32 MLP over three frozen prefix features in causal time order."""

    def __init__(self, input_dim: int, config: CompletionHeadConfig, *, rngs: nnx.Rngs):
        if input_dim <= 0:
            raise ValueError("temporal completion head input_dim must be positive")
        if config.variant != "temporal_mlp":
            raise ValueError("TemporalCompletionHead requires completion_head.variant='temporal_mlp'")
        if config.resolved_pooling != "masked_mean":
            raise ValueError("TemporalCompletionHead requires masked_mean pooling")

        self.input_dim = input_dim
        self.temporal_steps = config.temporal_steps
        self.dropout_rate = config.dropout_rate
        # A single LayerNorm module is intentionally applied across the whole
        # [batch, time, hidden] tensor, sharing parameters at all time steps.
        self.layer_norm = nnx.LayerNorm(
            input_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.projection = nnx.Linear(
            config.temporal_steps * input_dim,
            config.hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.output = nnx.Linear(
            config.hidden_dim,
            1,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(
        self,
        prefix_history: jax.Array,
        *,
        rng: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        prefix_history = jax.lax.stop_gradient(jnp.asarray(prefix_history, dtype=jnp.float32))
        expected_suffix = (self.temporal_steps, self.input_dim)
        if prefix_history.ndim != 3 or prefix_history.shape[1:] != expected_suffix:
            raise ValueError(
                "temporal completion prefix_history must have shape "
                f"[batch, {self.temporal_steps}, {self.input_dim}], got {prefix_history.shape}"
            )

        normalized = self.layer_norm(prefix_history)
        # The input contract is oldest-to-newest. C-order reshape preserves
        # [z_(t-2), z_(t-1), z_t] as consecutive feature blocks.
        concatenated = jnp.reshape(
            normalized,
            (prefix_history.shape[0], self.temporal_steps * self.input_dim),
        )
        hidden = jax.nn.gelu(self.projection(concatenated))
        if train and self.dropout_rate:
            if rng is None:
                raise ValueError("temporal completion head requires an RNG when dropout is enabled during training")
            keep_probability = 1.0 - self.dropout_rate
            keep = jax.random.bernoulli(rng, keep_probability, hidden.shape)
            hidden = jnp.where(keep, hidden / keep_probability, 0.0)
        logits = self.output(hidden)
        return jnp.asarray(logits[..., 0], dtype=jnp.float32)


def _dropout(values: jax.Array, rng: jax.Array, rate: float) -> jax.Array:
    keep_probability = 1.0 - rate
    keep = jax.random.bernoulli(rng, keep_probability, values.shape)
    return jnp.where(keep, values / keep_probability, 0.0)


class _TokenTransformerBlock(nnx.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout_rate: float, *, rngs: nnx.Rngs):
        self.dropout_rate = dropout_rate
        self.attention_norm = nnx.LayerNorm(hidden_dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            dropout_rate=0.0,
            decode=False,
            rngs=rngs,
        )
        self.ffn_norm = nnx.LayerNorm(hidden_dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.ffn_in = nnx.Linear(
            hidden_dim,
            4 * hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.ffn_out = nnx.Linear(
            4 * hidden_dim,
            hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )

    def __call__(self, tokens: jax.Array, *, rng: jax.Array | None, train: bool) -> jax.Array:
        if train and self.dropout_rate and rng is None:
            raise ValueError("token transformer block requires an RNG during training")
        attention_output = self.attention(self.attention_norm(tokens), deterministic=True)
        if train and self.dropout_rate:
            rng, attention_rng = jax.random.split(rng)
            attention_output = _dropout(attention_output, attention_rng, self.dropout_rate)
        tokens = tokens + attention_output
        ffn_output = self.ffn_out(jax.nn.gelu(self.ffn_in(self.ffn_norm(tokens))))
        if train and self.dropout_rate:
            assert rng is not None
            ffn_output = _dropout(ffn_output, rng, self.dropout_rate)
        return tokens + ffn_output


class TokenQueryCompletionHead(nnx.Module):
    """Q-Former-style done adapter over three complete VLM token sequences."""

    def __init__(self, input_dim: int, config: CompletionHeadConfig, *, rngs: nnx.Rngs):
        if input_dim <= 0:
            raise ValueError("token-query head input_dim must be positive")
        if config.variant != "token_query_attention":
            raise ValueError("TokenQueryCompletionHead requires variant='token_query_attention'")
        if config.hidden_dim % config.attention_heads:
            raise ValueError(f"token-query hidden_dim must be divisible by {config.attention_heads}")

        hidden_dim = config.hidden_dim
        self.input_dim = input_dim
        self.temporal_steps = config.temporal_steps
        self.query_count = config.query_count
        self.attention_heads = config.attention_heads
        self.temporal_layers = config.temporal_layers
        self.dropout_rate = config.dropout_rate
        self.token_norm = nnx.LayerNorm(input_dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.learned_queries = nnx.Param(
            jax.random.normal(rngs.params(), (self.query_count, hidden_dim), dtype=jnp.float32) * 0.02
        )
        self.time_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (config.temporal_steps, hidden_dim), dtype=jnp.float32) * 0.02
        )
        self.query_projection = nnx.Linear(
            hidden_dim, hidden_dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs
        )
        self.key_projection = nnx.Linear(input_dim, hidden_dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.value_projection = nnx.Linear(input_dim, hidden_dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.cross_output = nnx.Linear(hidden_dim, hidden_dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.cross_norm = nnx.LayerNorm(hidden_dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.cross_ffn_norm = nnx.LayerNorm(hidden_dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.cross_ffn_in = nnx.Linear(
            hidden_dim,
            4 * hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.cross_ffn_out = nnx.Linear(
            4 * hidden_dim,
            hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.temporal_blocks = nnx.Dict(
            {
                f"layer_{index}": _TokenTransformerBlock(
                    hidden_dim,
                    self.attention_heads,
                    config.dropout_rate,
                    rngs=rngs,
                )
                for index in range(self.temporal_layers)
            }
        )
        self.output_norm = nnx.LayerNorm(hidden_dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.output = nnx.Linear(hidden_dim, 1, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)

    def __call__(
        self,
        prefix_history: jax.Array,
        prefix_mask_history: jax.Array,
        *,
        rng: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
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
        projected_queries = self.query_projection(queries)
        scores = jnp.einsum("bqd,bkd->bqk", projected_queries, keys) / math.sqrt(keys.shape[-1])
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
        logits = self.output(self.output_norm(queries[:, 0]))
        return jnp.asarray(logits[..., 0], dtype=jnp.float32)
