"""Runtime for the frozen token-query completion head.

The deployment head intentionally lives outside ``src/openpi``.  It mirrors the
``token_query_attention`` implementation from the audited
``completion_head_frozen`` branch and consumes the unpooled Pi0.5 prefix output
directly.  The runtime keeps only the three in-memory prefix frames required by
the head; it never creates a feature cache on disk.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import time
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model


@dataclasses.dataclass(frozen=True)
class CompletionHeadConfig:
    """Exact structural configuration used by the frozen token-query head."""

    enabled: bool = True
    variant: str = "token_query_attention"
    pooling: str = "masked_mean"
    projection_dim: int = 256
    temporal_steps: int = 3
    hidden_dim: int = 768
    query_count: int = 32
    attention_heads: int = 12
    temporal_layers: int = 3
    dropout_rate: float = 0.1

    def __post_init__(self) -> None:
        if self.variant != "token_query_attention":
            raise ValueError(f"unsupported deployment completion variant: {self.variant!r}")
        if self.pooling != "masked_mean":
            raise ValueError("token_query_attention requires pooling='masked_mean'")
        if self.temporal_steps != 3:
            raise ValueError("the deployment head requires exactly three temporal steps")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.dropout_rate < 0.0 or self.dropout_rate >= 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")


@dataclasses.dataclass(frozen=True)
class HeadMetadata:
    """Validated metadata needed to interpret a completion-head checkpoint."""

    variant: str
    temporal_steps: int
    feature_dim: int
    token_count: int
    hidden_dim: int
    query_count: int
    attention_heads: int
    temporal_layers: int
    dropout_rate: float
    history_times_seconds: tuple[float, ...]
    source_path: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _metadata_value(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    head = metadata.get("head")
    if isinstance(head, dict) and key in head:
        return head[key]
    return metadata.get(key, default)


def load_head_metadata(head_dir: pathlib.Path | str) -> tuple[HeadMetadata, dict[str, Any]]:
    """Load and validate the fixed deployment metadata without comparing hashes."""

    metadata_path = pathlib.Path(head_dir) / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    cache = raw.get("cache_metadata") if isinstance(raw.get("cache_metadata"), dict) else {}
    history = tuple(float(value) for value in cache.get("history_times_seconds", (-1.0, -0.5, 0.0)))
    parsed = HeadMetadata(
        variant=str(_metadata_value(raw, "variant")),
        temporal_steps=int(_metadata_value(raw, "temporal_steps", len(history))),
        feature_dim=int(cache.get("feature_dim", raw.get("feature_dim", 2048))),
        token_count=int(cache.get("token_count", raw.get("token_count", 968))),
        hidden_dim=int(_metadata_value(raw, "hidden_dim", 768)),
        query_count=int(_metadata_value(raw, "query_count", 32)),
        attention_heads=int(_metadata_value(raw, "attention_heads", 12)),
        temporal_layers=int(_metadata_value(raw, "temporal_layers", 3)),
        dropout_rate=float(_metadata_value(raw, "dropout_rate", 0.1)),
        history_times_seconds=history,
        source_path=str(metadata_path),
    )
    expected = HeadMetadata(
        variant="token_query_attention",
        temporal_steps=3,
        feature_dim=2048,
        token_count=968,
        hidden_dim=768,
        query_count=32,
        attention_heads=12,
        temporal_layers=3,
        dropout_rate=0.1,
        history_times_seconds=(-1.0, -0.5, 0.0),
        source_path=str(metadata_path),
    )
    for field in dataclasses.fields(HeadMetadata):
        name = field.name
        if name == "source_path":
            continue
        actual = getattr(parsed, name)
        wanted = getattr(expected, name)
        if isinstance(wanted, float):
            matches = math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1.0e-8)
        else:
            matches = actual == wanted
        if not matches:
            raise ValueError(f"head metadata {name}={actual!r} does not match required value {wanted!r}")
    return parsed, raw


def _dropout(values: jax.Array, rng: jax.Array | None, rate: float) -> jax.Array:
    """The exact branch-local inverted dropout implementation."""

    keep_probability = 1.0 - rate
    if rng is None:
        raise ValueError("token-query completion head requires an RNG when dropout is enabled")
    keep = jax.random.bernoulli(rng, keep_probability, values.shape)
    return jnp.where(keep, values / keep_probability, 0.0)


class _TokenTransformerBlock(nnx.Module):
    """One exact pre-norm token transformer block from the frozen branch."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout_rate: float, *, rngs: nnx.Rngs):
        self.dropout_rate = dropout_rate
        self.attention_norm = nnx.LayerNorm(
            hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            dropout_rate=0.0,
            decode=False,
            rngs=rngs,
        )
        self.ffn_norm = nnx.LayerNorm(
            hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
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
            raise ValueError("token transformer requires an RNG when dropout is enabled during training")
        attention_output = self.attention(self.attention_norm(tokens), deterministic=True)
        if train and self.dropout_rate:
            assert rng is not None
            attention_rng, ffn_rng = jax.random.split(rng)
            attention_output = _dropout(attention_output, attention_rng, self.dropout_rate)
        else:
            ffn_rng = None
        tokens = tokens + attention_output
        ffn_output = self.ffn_out(jax.nn.gelu(self.ffn_in(self.ffn_norm(tokens))))
        if train and self.dropout_rate:
            ffn_output = _dropout(ffn_output, ffn_rng, self.dropout_rate)
        return tokens + ffn_output


class TokenQueryCompletionHead(nnx.Module):
    """Token-query attention head matching the frozen branch parameter layout."""

    def __init__(self, input_dim: int, config: CompletionHeadConfig, *, rngs: nnx.Rngs):
        if input_dim <= 0:
            raise ValueError("completion head input_dim must be positive")
        if config.variant != "token_query_attention":
            raise ValueError("TokenQueryCompletionHead requires token_query_attention")
        if config.hidden_dim % config.attention_heads:
            raise ValueError("completion head hidden_dim must be divisible by attention_heads")
        self.input_dim = input_dim
        self.temporal_steps = config.temporal_steps
        self.query_count = config.query_count
        self.attention_heads = config.attention_heads
        self.temporal_layers = config.temporal_layers
        self.dropout_rate = config.dropout_rate

        self.token_norm = nnx.LayerNorm(
            input_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.learned_queries = nnx.Param(
            jax.random.normal(rngs.params(), (config.query_count, config.hidden_dim), dtype=jnp.float32) * 0.02
        )
        self.time_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (config.temporal_steps, config.hidden_dim), dtype=jnp.float32) * 0.02
        )
        self.query_projection = nnx.Linear(
            config.hidden_dim,
            config.hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.key_projection = nnx.Linear(
            input_dim,
            config.hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.value_projection = nnx.Linear(
            input_dim,
            config.hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.cross_output = nnx.Linear(
            config.hidden_dim,
            config.hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.cross_norm = nnx.LayerNorm(
            config.hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.cross_ffn_norm = nnx.LayerNorm(
            config.hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.cross_ffn_in = nnx.Linear(
            config.hidden_dim,
            4 * config.hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.cross_ffn_out = nnx.Linear(
            4 * config.hidden_dim,
            config.hidden_dim,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        self.temporal_blocks = nnx.Dict(
            {
                f"layer_{index}": _TokenTransformerBlock(
                    config.hidden_dim,
                    config.attention_heads,
                    config.dropout_rate,
                    rngs=rngs,
                )
                for index in range(config.temporal_layers)
            }
        )
        self.output_norm = nnx.LayerNorm(
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
        prefix_mask: jax.Array,
        *,
        rng: jax.Array | None = None,
        train: bool = False,
    ) -> jax.Array:
        prefix_history = jax.lax.stop_gradient(jnp.asarray(prefix_history, dtype=jnp.float32))
        prefix_mask = jnp.asarray(prefix_mask, dtype=jnp.bool_)
        if prefix_history.ndim != 4 or prefix_history.shape[1] != self.temporal_steps:
            raise ValueError(
                "token-query prefix_history must have shape "
                f"[batch, {self.temporal_steps}, tokens, {self.input_dim}], got {prefix_history.shape}"
            )
        if prefix_history.shape[-1] != self.input_dim:
            raise ValueError(f"token-query prefix feature dim must be {self.input_dim}, got {prefix_history.shape[-1]}")
        if prefix_mask.shape != prefix_history.shape[:3]:
            raise ValueError(
                f"token-query prefix_mask shape {prefix_mask.shape} does not match {prefix_history.shape[:3]}"
            )
        if train and self.dropout_rate and rng is None:
            raise ValueError("token-query completion head requires an RNG during training")

        normalized = self.token_norm(prefix_history)
        keys = self.key_projection(normalized) + self.time_embedding.value[None, :, None, :]
        values = self.value_projection(normalized) + self.time_embedding.value[None, :, None, :]
        batch_size = prefix_history.shape[0]
        keys = keys.reshape(batch_size, -1, keys.shape[-1])
        values = values.reshape(batch_size, -1, values.shape[-1])
        flat_mask = prefix_mask.reshape(batch_size, -1)
        queries = jnp.broadcast_to(
            self.learned_queries.value[None, :, :], (batch_size, self.query_count, self.learned_queries.value.shape[-1])
        )
        projected_queries = self.query_projection(queries)
        scores = jnp.einsum("bqd,bkd->bqk", projected_queries, keys) / math.sqrt(keys.shape[-1])
        scores = jnp.where(flat_mask[:, None, :], scores, jnp.finfo(jnp.float32).min)
        weights = jax.nn.softmax(scores, axis=-1)
        cross_output = self.cross_output(jnp.einsum("bqk,bkd->bqd", weights, values))
        if train and self.dropout_rate:
            assert rng is not None
            cross_rng, ffn_rng, *block_rngs_list = jax.random.split(rng, 2 + self.temporal_layers)
            cross_output = _dropout(cross_output, cross_rng, self.dropout_rate)
            block_rngs = tuple(block_rngs_list)
        else:
            ffn_rng = None
            block_rngs = (None,) * self.temporal_layers
        queries = self.cross_norm(queries + cross_output)
        cross_ffn = self.cross_ffn_out(jax.nn.gelu(self.cross_ffn_in(self.cross_ffn_norm(queries))))
        if train and self.dropout_rate:
            cross_ffn = _dropout(cross_ffn, ffn_rng, self.dropout_rate)
        queries = queries + cross_ffn
        for index, block_rng in enumerate(block_rngs):
            queries = self.temporal_blocks[f"layer_{index}"](queries, rng=block_rng, train=train)
        logits = self.output(self.output_norm(queries[:, 0]))
        return jnp.asarray(logits[..., 0], dtype=jnp.float32)


class CompletionHeadRuntime:
    """Loads and scores the standalone FP32 head in deterministic eval mode."""

    def __init__(self, head_dir: pathlib.Path | str, *, input_dim: int = 2048, token_count: int = 968):
        self.head_dir = pathlib.Path(head_dir)
        self.metadata, self.raw_metadata = load_head_metadata(self.head_dir)
        if input_dim != self.metadata.feature_dim or token_count != self.metadata.token_count:
            raise ValueError("deployment head input shape does not match its metadata")
        config = CompletionHeadConfig(
            enabled=True,
            variant="token_query_attention",
            pooling="masked_mean",
            temporal_steps=3,
            hidden_dim=768,
            query_count=32,
            attention_heads=12,
            temporal_layers=3,
            dropout_rate=0.1,
        )
        head = TokenQueryCompletionHead(input_dim, config, rngs=nnx.Rngs(jax.random.key(0)))
        params = _model.restore_params(self.head_dir / "params", dtype=jnp.float32)
        if set(params) != {"completion_head"}:
            raise ValueError(f"expected head-only params rooted at completion_head, got {sorted(params)}")
        graphdef, state = nnx.split(head)
        state.replace_by_pure_dict(params["completion_head"])
        head = nnx.merge(graphdef, state)
        self._graphdef, self._state = nnx.split(head)

        def score_fn(state_value: nnx.State, history: jax.Array, mask: jax.Array) -> jax.Array:
            module = nnx.merge(self._graphdef, state_value)
            return module(history, mask, train=False)

        self._score_jit = jax.jit(score_fn)

    def score(self, prefix_history: np.ndarray, prefix_mask: np.ndarray) -> tuple[float, float, float]:
        history = np.asarray(prefix_history, dtype=np.float32)
        mask = np.asarray(prefix_mask, dtype=np.bool_)
        return self.score_device(jnp.asarray(history), jnp.asarray(mask))

    def score_device(self, prefix_history: jax.Array, prefix_mask: jax.Array) -> tuple[float, float, float]:
        """Score three device-resident prefixes without moving the tokens to CPU."""

        expected_history = (3, self.metadata.token_count, self.metadata.feature_dim)
        if prefix_history.shape != expected_history:
            raise ValueError(f"prefix_history must have shape {expected_history}, got {prefix_history.shape}")
        if prefix_mask.shape != expected_history[:2]:
            raise ValueError(f"prefix_mask must have shape {expected_history[:2]}, got {prefix_mask.shape}")
        start = time.monotonic()
        logits = self._score_jit(
            self._state,
            jnp.asarray(prefix_history)[None, ...],
            jnp.asarray(prefix_mask)[None, ...],
        )
        logits = jax.block_until_ready(logits)
        logit = float(np.asarray(logits)[0])
        score = float(np.asarray(jax.nn.sigmoid(jnp.asarray(logit, dtype=jnp.float32))))
        return score, logit, (time.monotonic() - start) * 1000.0


class PrefixTokenRuntime:
    """Extracts the same unpooled prefix output used by Pi0.5 action inference."""

    def __init__(self, policy: Any, *, token_count: int = 968, feature_dim: int = 2048):
        self.policy = policy
        self.model = policy._model  # noqa: SLF001 - reuse the action policy's exact VLA.
        self.token_count = token_count
        self.feature_dim = feature_dim
        graphdef, state = nnx.split(self.model)
        self._graphdef = graphdef

        def prefix_fn(state_value: nnx.State, inputs: dict[str, Any]) -> tuple[jax.Array, jax.Array]:
            module = nnx.merge(graphdef, state_value)
            observation = _model.Observation.from_dict(inputs)
            observation = _model.preprocess_observation(None, observation, train=False)
            prefix_tokens, prefix_mask, prefix_ar_mask = module.embed_prefix(observation)
            prefix_attn_mask = _make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), _ = module.PaliGemma.llm(
                [prefix_tokens, None],
                mask=prefix_attn_mask,
                positions=positions,
            )
            if prefix_out is None:
                raise ValueError("Pi0.5 returned no prefix output")
            return jax.lax.stop_gradient(prefix_out), prefix_mask

        self._state = state
        self._prefix_jit = jax.jit(prefix_fn)

    def extract(self, raw_observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
        start = time.monotonic()
        clean_observation = {key: value for key, value in raw_observation.items() if not key.startswith("_")}
        transformed = self.policy._input_transform(  # noqa: SLF001 - identical transform as action inference.
            jax.tree.map(lambda value: value, clean_observation)
        )
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
        prefix_out, prefix_mask = self._prefix_jit(self._state, batched)
        prefix_out, prefix_mask = jax.block_until_ready((prefix_out, prefix_mask))
        features = np.asarray(prefix_out[0], dtype=np.float32)
        mask = np.asarray(prefix_mask[0], dtype=np.bool_)
        if features.shape != (self.token_count, self.feature_dim):
            raise ValueError(f"Pi0.5 prefix shape must be {(self.token_count, self.feature_dim)}, got {features.shape}")
        if mask.shape != (self.token_count,):
            raise ValueError(f"Pi0.5 prefix mask shape must be {(self.token_count,)}, got {mask.shape}")
        return features, mask, (time.monotonic() - start) * 1000.0


def _make_attn_mask(input_mask: jax.Array, mask_ar: jax.Array) -> jax.Array:
    """Pi0 prefix attention mask, kept local so no model source file changes."""

    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@dataclasses.dataclass
class _History:
    generation: int | None = None
    frames: list[np.ndarray] = dataclasses.field(default_factory=list)
    masks: list[np.ndarray] = dataclasses.field(default_factory=list)

    def append(self, generation: int, features: np.ndarray, mask: np.ndarray) -> None:
        if self.generation != generation:
            self.generation = generation
            self.frames.clear()
            self.masks.clear()
        self.frames.append(np.asarray(features, dtype=np.float32))
        self.masks.append(np.asarray(mask, dtype=np.bool_))
        del self.frames[:-3]
        del self.masks[:-3]

    def clear(self) -> None:
        self.generation = None
        self.frames.clear()
        self.masks.clear()


class CompletionSession:
    """One WebSocket connection's three-frame history."""

    def __init__(self, prefix: PrefixTokenRuntime, head: CompletionHeadRuntime):
        self.head = head
        self.prefix = prefix
        self.history = _History()

    def reset(self) -> None:
        self.history.clear()

    def compute(self, request: dict[str, Any]) -> dict[str, Any]:
        generation = int(request.get("_completion_prompt_generation", 0))
        task_index = int(request.get("_completion_task_index", 0))
        client_step = int(request.get("_completion_client_step", -1))
        history_reset = bool(request.get("_completion_reset_history", False))
        if history_reset:
            self.history.clear()
        started = time.monotonic()
        features, mask, prefix_ms = self.prefix.extract(request)
        self.history.append(generation, features, mask)
        history_size = len(self.history.frames)
        score = None
        logit = None
        head_ms = 0.0
        if history_size == 3:
            history = np.stack(self.history.frames, axis=0)
            masks = np.stack(self.history.masks, axis=0)
            score, logit, head_ms = self.head.score(history, masks)
        finished = time.monotonic()
        return {
            "score": score,
            "logit": logit,
            "history_ready": history_size == 3,
            "history_size": history_size,
            "task_index": task_index,
            "prompt_generation": generation,
            "client_step": client_step,
            "history_reset": history_reset,
            "scheduled_tick_time": request.get("_completion_scheduled_tick_time"),
            "server_timing": {
                "completion_total_ms": (finished - started) * 1000.0,
                "prefix_extract_ms": prefix_ms,
                "head_score_ms": head_ms,
                "server_received_monotonic_s": started,
                "server_finished_monotonic_s": finished,
            },
        }


class CompletionRuntime:
    """Shared prefix/head runtime that creates isolated per-connection sessions."""

    def __init__(self, policy: Any, head_dir: pathlib.Path | str):
        self.head = CompletionHeadRuntime(head_dir)
        self.prefix = PrefixTokenRuntime(
            policy,
            token_count=self.head.metadata.token_count,
            feature_dim=self.head.metadata.feature_dim,
        )

    def new_session(self) -> CompletionSession:
        return CompletionSession(self.prefix, self.head)

    def compute(self, request: dict[str, Any]) -> dict[str, Any]:
        """Compatibility shortcut for callers that use one implicit session."""

        if not hasattr(self, "_session"):
            self._session = self.new_session()
        return self._session.compute(request)
