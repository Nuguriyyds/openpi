"""Frozen-prefix completion head used by pi0.5 training."""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp

CompletionHeadVariant = Literal["legacy_attention", "temporal_mlp", "raw_prefix_decoder"]
CompletionPooling = Literal["masked_attention", "masked_mean", "raw_prefix"]
RAW_PREFIX_SEGMENT_COUNT = 4
RAW_PREFIX_MAX_WITHIN_SEGMENT_POSITION = 256
RAW_PREFIX_MAX_LANGUAGE_POSITION = 200


def build_raw_prefix_layout(
    image_names: tuple[str, ...] | list[str],
    image_token_counts: tuple[int, ...] | list[int],
    language_token_count: int,
) -> tuple[jax.Array, jax.Array]:
    """Builds layout ids in the exact order used by :meth:`Pi0.embed_prefix`.

    The names are accepted as an audit/debug input; ordering is deliberately
    taken from the actual image iteration order rather than inferred from
    token values.  Segment ids 0, 1, and 2 correspond to the first three image
    blocks (top, left wrist, right wrist), while 3 is prompt/state language.
    """

    if len(image_names) != len(image_token_counts) or len(image_names) > RAW_PREFIX_SEGMENT_COUNT - 1:
        raise ValueError("raw-prefix layout requires at most three image blocks with matching names and counts")
    if language_token_count < 0:
        raise ValueError("raw-prefix language token count must be non-negative")
    segment_ids: list[int] = []
    position_ids: list[int] = []
    for segment_id, (name, count) in enumerate(zip(image_names, image_token_counts, strict=True)):
        del name
        if count < 0 or count > RAW_PREFIX_MAX_WITHIN_SEGMENT_POSITION:
            raise ValueError("raw-prefix image blocks must contain between 0 and 256 tokens")
        segment_ids.extend([segment_id] * count)
        position_ids.extend(range(count))
    if language_token_count > RAW_PREFIX_MAX_LANGUAGE_POSITION:
        raise ValueError("raw-prefix prompt/state block must contain at most 200 tokens")
    segment_ids.extend([RAW_PREFIX_SEGMENT_COUNT - 1] * language_token_count)
    position_ids.extend(range(language_token_count))
    return jnp.asarray(segment_ids, dtype=jnp.int32), jnp.asarray(position_ids, dtype=jnp.int32)


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

    # ``raw_prefix_decoder`` deliberately keeps its own structural settings
    # instead of overloading the legacy projection/MLP fields.  The input
    # width is still supplied by the selected PaliGemma configuration.
    decoder_dim: int = 256
    decoder_num_queries: int = 16
    decoder_num_layers: int = 4
    decoder_num_heads: int = 8
    decoder_ffn_dim: int = 1024

    def __post_init__(self) -> None:
        if self.variant not in ("legacy_attention", "temporal_mlp", "raw_prefix_decoder"):
            raise ValueError(f"unknown completion_head.variant: {self.variant!r}")
        if self.pooling not in (None, "masked_attention", "masked_mean", "raw_prefix"):
            raise ValueError(f"unknown completion_head.pooling: {self.pooling!r}")
        if self.projection_dim <= 0:
            raise ValueError("completion_head.projection_dim must be positive")
        if self.temporal_steps <= 0:
            raise ValueError("completion_head.temporal_steps must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("completion_head.hidden_dim must be positive")
        for field_name in (
            "decoder_dim",
            "decoder_num_queries",
            "decoder_num_layers",
            "decoder_num_heads",
            "decoder_ffn_dim",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"completion_head.{field_name} must be positive")
        if self.decoder_dim % self.decoder_num_heads:
            raise ValueError("completion_head.decoder_dim must be divisible by decoder_num_heads")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("completion_head.dropout_rate must be in [0, 1)")
        if self.variant == "temporal_mlp" and self.temporal_steps != 3:
            raise ValueError("temporal completion requires exactly three prefix time steps")

        expected_pooling = {
            "legacy_attention": "masked_attention",
            "temporal_mlp": "masked_mean",
            "raw_prefix_decoder": "raw_prefix",
        }[self.variant]
        if self.pooling is not None and self.pooling != expected_pooling:
            raise ValueError(
                f"completion_head variant {self.variant!r} requires pooling={expected_pooling!r}, got {self.pooling!r}"
            )

    @property
    def resolved_pooling(self) -> CompletionPooling:
        """Returns the pooling operation implied by the explicit head variant."""

        return {
            "legacy_attention": "masked_attention",
            "temporal_mlp": "masked_mean",
            "raw_prefix_decoder": "raw_prefix",
        }[self.variant]


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


def _dropout(value: jax.Array, rate: float, *, rng: jax.Array | None, train: bool) -> jax.Array:
    """Applies the small head-local dropout used by the raw decoder."""

    if not train or rate == 0.0:
        return value
    if rng is None:
        raise ValueError("raw-prefix completion head requires an RNG when dropout is enabled during training")
    keep_probability = 1.0 - rate
    keep = jax.random.bernoulli(rng, keep_probability, value.shape)
    return jnp.where(keep, value / keep_probability, 0.0)


class _RawPrefixMultiHeadAttention(nnx.Module):
    """A compact FP32 multi-head attention module for the raw prefix decoder."""

    def __init__(self, dim: int, num_heads: int, *, rngs: nnx.Rngs):
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("raw-prefix attention dimension must be divisible by a positive head count")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nnx.Linear(dim, dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.key = nnx.Linear(dim, dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.value = nnx.Linear(dim, dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.output = nnx.Linear(dim, dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)

    def _split_heads(self, values: jax.Array) -> jax.Array:
        values = jnp.reshape(values, (values.shape[0], values.shape[1], self.num_heads, self.head_dim))
        return jnp.transpose(values, (0, 2, 1, 3))

    def _merge_heads(self, values: jax.Array) -> jax.Array:
        values = jnp.transpose(values, (0, 2, 1, 3))
        return jnp.reshape(values, (values.shape[0], values.shape[1], self.dim))

    def __call__(self, query: jax.Array, memory: jax.Array, *, memory_mask: jax.Array | None = None) -> jax.Array:
        query = jnp.asarray(query, dtype=jnp.float32)
        memory = jnp.asarray(memory, dtype=jnp.float32)
        if memory_mask is None:
            valid = jnp.ones((memory.shape[0], memory.shape[1]), dtype=jnp.bool_)
        else:
            valid = jnp.asarray(memory_mask, dtype=jnp.bool_)
            if valid.shape != memory.shape[:2]:
                raise ValueError(
                    f"raw-prefix cross-attention mask shape {valid.shape} does not match memory {memory.shape[:2]}"
                )
            # Zero masked K/V rows before projection as well as masking their
            # attention weights.  This keeps arbitrary padding contents,
            # including non-finite sentinels, from reaching the dot products.
            memory = jnp.where(valid[:, :, None], memory, jnp.zeros((), dtype=jnp.float32))
        q = self._split_heads(self.query(query))
        k = self._split_heads(self.key(memory))
        v = self._split_heads(self.value(memory))
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k, precision=jax.lax.Precision.HIGHEST)
        scores = scores / math.sqrt(self.head_dim)

        # The explicit multiply after exp handles an all-padding row without
        # ever allowing a masked token to contribute a NaN or a large value.
        masked_scores = jnp.where(valid[:, None, None, :], scores, jnp.asarray(-1.0e30, dtype=jnp.float32))
        score_max = jnp.max(masked_scores, axis=-1, keepdims=True)
        weights = jnp.exp(masked_scores - score_max) * valid[:, None, None, :].astype(jnp.float32)
        weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1.0e-8)
        attended = jnp.einsum("bhqk,bhkd->bhqd", weights, v, precision=jax.lax.Precision.HIGHEST)
        return self.output(self._merge_heads(attended))


class _RawPrefixDecoderBlock(nnx.Module):
    """One pre-norm self/cross-attention decoder block."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int, *, dropout_rate: float, rngs: nnx.Rngs):
        self.self_norm = nnx.LayerNorm(dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.cross_norm = nnx.LayerNorm(dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.ffn_norm = nnx.LayerNorm(dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.self_attention = _RawPrefixMultiHeadAttention(dim, num_heads, rngs=rngs)
        self.cross_attention = _RawPrefixMultiHeadAttention(dim, num_heads, rngs=rngs)
        self.ffn_in = nnx.Linear(dim, ffn_dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.ffn_out = nnx.Linear(ffn_dim, dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.dropout_rate = dropout_rate

    def __call__(
        self,
        queries: jax.Array,
        memory: jax.Array,
        prefix_mask: jax.Array,
        *,
        rngs: tuple[jax.Array, jax.Array, jax.Array] | None,
        train: bool,
    ) -> jax.Array:
        if train and self.dropout_rate:
            if rngs is None:
                raise ValueError("raw-prefix decoder requires dropout RNGs during training")
            self_rng, cross_rng, ffn_rng = rngs
        else:
            self_rng = cross_rng = ffn_rng = None

        normalized_queries = self.self_norm(queries)
        self_attended = self.self_attention(normalized_queries, normalized_queries)
        queries = queries + _dropout(self_attended, self.dropout_rate, rng=self_rng, train=train)
        cross_attended = self.cross_attention(
            self.cross_norm(queries),
            memory,
            memory_mask=prefix_mask,
        )
        queries = queries + _dropout(cross_attended, self.dropout_rate, rng=cross_rng, train=train)
        ffn_hidden = jax.nn.gelu(self.ffn_in(self.ffn_norm(queries)))
        # Keep the configured FFN order explicit: Linear -> GELU -> Dropout
        # -> Linear.  The self- and cross-attention residual branches apply
        # their dropout after attention output above.
        ffn_hidden = _dropout(ffn_hidden, self.dropout_rate, rng=ffn_rng, train=train)
        queries = queries + self.ffn_out(ffn_hidden)
        return jnp.asarray(queries, dtype=jnp.float32)


class _RawPrefixDecoderStack(nnx.Module):
    """Old-NNX-compatible decoder container with string-only state paths."""

    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        *,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ) -> None:
        self.num_layers = num_layers
        for layer_index in range(num_layers):
            setattr(
                self,
                f"block_{layer_index}",
                _RawPrefixDecoderBlock(
                    dim,
                    num_heads,
                    ffn_dim,
                    dropout_rate=dropout_rate,
                    rngs=rngs,
                ),
            )

    def block(self, layer_index: int) -> _RawPrefixDecoderBlock:
        if layer_index < 0 or layer_index >= self.num_layers:
            raise IndexError(layer_index)
        return getattr(self, f"block_{layer_index}")


class RawPrefixCompletionHead(nnx.Module):
    """Current-frame decoder over every frozen raw prefix token.

    The head performs only a per-token memory projection before the decoder;
    the sequence length is preserved at every layer.  All head parameters are
    explicitly FP32, while inputs may be BF16, FP16, or FP32.
    """

    _SEGMENT_COUNT = 4
    _MAX_WITHIN_SEGMENT_POSITION = 256

    def __init__(self, input_dim: int, config: CompletionHeadConfig, *, rngs: nnx.Rngs):
        if input_dim <= 0:
            raise ValueError("raw-prefix completion head input_dim must be positive")
        if config.variant != "raw_prefix_decoder":
            raise ValueError("RawPrefixCompletionHead requires completion_head.variant='raw_prefix_decoder'")
        if config.resolved_pooling != "raw_prefix":
            raise ValueError("RawPrefixCompletionHead does not use a pooling operation")

        self.input_dim = input_dim
        self.decoder_dim = config.decoder_dim
        self.decoder_num_queries = config.decoder_num_queries
        self.decoder_num_layers = config.decoder_num_layers
        self.decoder_num_heads = config.decoder_num_heads
        self.decoder_ffn_dim = config.decoder_ffn_dim
        self.dropout_rate = config.dropout_rate

        self.memory_norm = nnx.LayerNorm(input_dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.memory_projection = nnx.Linear(
            input_dim,
            config.decoder_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.segment_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (self._SEGMENT_COUNT, config.decoder_dim), dtype=jnp.float32) * 0.02
        )
        self.position_embedding = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (self._MAX_WITHIN_SEGMENT_POSITION, config.decoder_dim),
                dtype=jnp.float32,
            )
            * 0.02
        )
        self.completion_queries = nnx.Param(
            jax.random.normal(
                rngs.params(),
                (config.decoder_num_queries, config.decoder_dim),
                dtype=jnp.float32,
            )
            / math.sqrt(config.decoder_dim)
        )
        self.decoder_blocks = _RawPrefixDecoderStack(
            config.decoder_num_layers,
            config.decoder_dim,
            config.decoder_num_heads,
            config.decoder_ffn_dim,
            dropout_rate=config.dropout_rate,
            rngs=rngs,
        )
        self.output_norm = nnx.LayerNorm(config.decoder_dim, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)
        self.output = nnx.Linear(config.decoder_dim, 1, dtype=jnp.float32, param_dtype=jnp.float32, rngs=rngs)

    def __call__(
        self,
        prefix_out: jax.Array,
        prefix_mask: jax.Array,
        prefix_segment_ids: jax.Array,
        prefix_position_ids: jax.Array,
        *,
        rng: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        # This stop-gradient is at the public head boundary so a changed
        # caller cannot accidentally unfreeze the VLM through this objective.
        prefix_out = jax.lax.stop_gradient(jnp.asarray(prefix_out, dtype=jnp.float32))
        prefix_mask = jnp.asarray(prefix_mask, dtype=jnp.bool_)
        segment_ids = jnp.asarray(prefix_segment_ids, dtype=jnp.int32)
        position_ids = jnp.asarray(prefix_position_ids, dtype=jnp.int32)
        if prefix_out.ndim != 3:
            raise ValueError(f"raw-prefix completion tokens must have shape [B, S, D], got {prefix_out.shape}")
        if prefix_out.shape[1] <= 0:
            raise ValueError("raw-prefix completion tokens must contain at least one prefix token")
        if prefix_mask.shape != prefix_out.shape[:2]:
            raise ValueError(
                f"raw-prefix completion mask shape {prefix_mask.shape} does not match tokens {prefix_out.shape[:2]}"
            )
        # The cache stores one shared [S] layout.  A generic PyTorch collate
        # function may repeat it to [B, S], so accept and collapse that
        # harmless representation at the head boundary as well.
        if segment_ids.ndim == 2 and segment_ids.shape == prefix_out.shape[:2]:
            segment_ids = segment_ids[0]
        if position_ids.ndim == 2 and position_ids.shape == prefix_out.shape[:2]:
            position_ids = position_ids[0]
        if segment_ids.shape != (prefix_out.shape[1],) or position_ids.shape != (prefix_out.shape[1],):
            raise ValueError(
                "raw-prefix layout ids must each have shape [S], "
                f"got {segment_ids.shape} and {position_ids.shape} for S={prefix_out.shape[1]}"
            )
        memory = self.memory_projection(self.memory_norm(prefix_out))
        memory = memory + self.segment_embedding.value[segment_ids][None, :, :]
        memory = memory + self.position_embedding.value[position_ids][None, :, :]
        queries = jnp.broadcast_to(
            self.completion_queries.value[None, :, :],
            (prefix_out.shape[0], self.decoder_num_queries, self.decoder_dim),
        )
        if train and self.dropout_rate:
            if rng is None:
                raise ValueError("raw-prefix completion head requires an RNG when dropout is enabled during training")
            layer_rngs = jax.random.split(rng, self.decoder_num_layers * 3)
        else:
            layer_rngs = None
        for layer_index in range(self.decoder_num_layers):
            block = self.decoder_blocks.block(layer_index)
            block_rngs = None
            if layer_rngs is not None:
                start = layer_index * 3
                block_rngs = tuple(layer_rngs[start : start + 3])  # type: ignore[assignment]
            queries = block(queries, memory, prefix_mask, rngs=block_rngs, train=train)

        cls = self.output_norm(queries[:, 0, :])
        logits = self.output(cls)
        return jnp.asarray(logits[:, 0], dtype=jnp.float32)
