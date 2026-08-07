"""Frozen-prefix completion head used by pi0.5 training."""

from __future__ import annotations

import dataclasses
import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class CompletionHeadConfig:
    """Configuration for the optional VLM completion classifier.

    ``input_dim`` deliberately does not live in this config. It is read from
    the selected PaliGemma model configuration when the model is constructed.
    """

    enabled: bool = False
    projection_dim: int = 256
    dropout_rate: float = 0.1

    def __post_init__(self) -> None:
        if self.projection_dim <= 0:
            raise ValueError("completion_head.projection_dim must be positive")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("completion_head.dropout_rate must be in [0, 1)")


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
    """FP32 learned-attention classifier over frozen VLM prefix outputs."""

    def __init__(self, input_dim: int, config: CompletionHeadConfig, *, rngs: nnx.Rngs):
        if input_dim <= 0:
            raise ValueError("completion head input_dim must be positive")
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
