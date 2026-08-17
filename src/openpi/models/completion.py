"""Frozen-prefix completion head used by pi0.5 training."""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp

CompletionHeadVariant = Literal["legacy_attention", "temporal_mlp"]
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
    dropout_rate: float = 0.1

    def __post_init__(self) -> None:
        if self.variant not in ("legacy_attention", "temporal_mlp"):
            raise ValueError(f"unknown completion_head.variant: {self.variant!r}")
        if self.pooling not in (None, "masked_attention", "masked_mean"):
            raise ValueError(f"unknown completion_head.pooling: {self.pooling!r}")
        if self.projection_dim <= 0:
            raise ValueError("completion_head.projection_dim must be positive")
        if self.temporal_steps <= 0:
            raise ValueError("completion_head.temporal_steps must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("completion_head.hidden_dim must be positive")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("completion_head.dropout_rate must be in [0, 1)")
        if self.variant == "temporal_mlp" and self.temporal_steps != 3:
            raise ValueError("temporal completion requires exactly three prefix time steps")

        expected_pooling = "masked_mean" if self.variant == "temporal_mlp" else "masked_attention"
        if self.pooling is not None and self.pooling != expected_pooling:
            raise ValueError(
                f"completion_head variant {self.variant!r} requires pooling={expected_pooling!r}, got {self.pooling!r}"
            )

    @property
    def resolved_pooling(self) -> CompletionPooling:
        """Returns the pooling operation implied by the explicit head variant."""

        return "masked_mean" if self.variant == "temporal_mlp" else "masked_attention"


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
