"""Train the 0.03B token-query done adapter from sharded prefix-token caches."""

# ruff: noqa: E402, I001 -- deterministic XLA flags must be set before JAX imports.

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any


_DETERMINISTIC_XLA_FLAGS = (
    "--xla_gpu_deterministic_ops=true",
    "--xla_gpu_exclude_nondeterministic_ops=true",
)


def _with_default_deterministic_xla_flags(flags: str) -> str:
    tokens = flags.split()
    configured = {token.split("=", maxsplit=1)[0] for token in tokens if token.startswith("--")}
    for flag in _DETERMINISTIC_XLA_FLAGS:
        if flag.split("=", maxsplit=1)[0] not in configured:
            tokens.append(flag)
    return " ".join(tokens)


# XLA reads these flags when JAX initializes, so configure them before importing
# Flax/JAX. An explicitly supplied value still takes precedence.
os.environ["XLA_FLAGS"] = _with_default_deterministic_xla_flags(os.environ.get("XLA_FLAGS", ""))

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from openpi.models import completion as completion_model

DEFAULT_TOKEN_CACHE = Path("/home/geek/share3/vla_done/v2/qwen_done_v2/qwen_style_done_tokens_v2_shards")
DEFAULT_OUTPUT = Path("/home/geek/share3/vla_done/v2/done_head_h768_deterministic_full_seed42_20260828")


@dataclasses.dataclass(frozen=True)
class TrainArgs:
    token_cache: Path = DEFAULT_TOKEN_CACHE
    output: Path = DEFAULT_OUTPUT
    epochs: int = 2
    batch_size: int = 48
    eval_batch_size: int = 16
    eval_steps: int = 200
    learning_rate: float = 5.0e-5
    min_learning_rate: float = 1.0e-6
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    hidden_dim: int = 768
    query_count: int = 32
    attention_heads: int = 12
    temporal_layers: int = 3
    dropout_rate: float = 0.1
    seed: int = 42
    overwrite: bool = False


@dataclasses.dataclass(frozen=True)
class _TokenShard:
    start: int
    stop: int
    tokens: np.ndarray
    masks: np.ndarray


@dataclasses.dataclass(frozen=True)
class TokenCache:
    metadata: dict[str, Any]
    shards: tuple[_TokenShard, ...]
    history_indices: np.ndarray
    labels: np.ndarray
    splits: np.ndarray

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.splits == split)

    def histories(self, row_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feature_indices = np.asarray(self.history_indices[row_indices], dtype=np.int64)
        flat_indices = feature_indices.reshape(-1)
        token_count = int(self.metadata["token_count"])
        feature_dim = int(self.metadata["feature_dim"])
        tokens = np.empty((len(flat_indices), token_count, feature_dim), dtype=np.float16)
        masks = np.empty((len(flat_indices), token_count), dtype=np.bool_)
        assigned = np.zeros(len(flat_indices), dtype=np.bool_)
        for shard in self.shards:
            positions = np.flatnonzero((flat_indices >= shard.start) & (flat_indices < shard.stop))
            if not len(positions):
                continue
            local_indices = flat_indices[positions] - shard.start
            tokens[positions] = shard.tokens[local_indices]
            masks[positions] = shard.masks[local_indices]
            assigned[positions] = True
        if not assigned.all():
            raise ValueError("history references a feature outside the token shards")
        batch_shape = (*feature_indices.shape, token_count)
        return tokens.reshape(*batch_shape, feature_dim), masks.reshape(batch_shape)


def _batches(indices: np.ndarray, batch_size: int) -> tuple[np.ndarray, ...]:
    return tuple(indices[start : start + batch_size] for start in range(0, len(indices), batch_size))


def binary_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    targets = np.asarray(labels, dtype=np.int32)
    predictions = probabilities >= 0.5
    true_positive = int(np.sum(predictions & (targets == 1)))
    false_positive = int(np.sum(predictions & (targets == 0)))
    false_negative = int(np.sum(~predictions & (targets == 1)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "accuracy": float(np.mean(predictions == targets)),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, np.finfo(np.float64).eps),
    }


def _save_head(path: Path, params: nnx.State, metadata: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=False)
    pure_params = jax.tree.map(np.asarray, params.to_pure_dict())
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path / "params", {"params": {"completion_head": pure_params}})
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_token_cache(root: Path) -> TokenCache:
    shard_roots = sorted(root.resolve().glob("shard_*"), key=lambda path: int(path.name.split("_")[-1]))
    if not shard_roots:
        raise FileNotFoundError(f"no token shards found under {root}")
    metadata_items = [json.loads((path / "metadata.json").read_text(encoding="utf-8")) for path in shard_roots]
    first = metadata_items[0]
    shard_count = int(first["feature_shard"]["count"])
    if len(shard_roots) != shard_count:
        raise ValueError(f"expected {shard_count} token shards, found {len(shard_roots)}")
    for key in ("feature_plan_sha256", "feature_count", "feature_dim", "token_count", "row_count"):
        if any(item[key] != first[key] for item in metadata_items):
            raise ValueError(f"token shard metadata disagrees on {key}")

    shards: list[_TokenShard] = []
    expected_start = 0
    for path, metadata in zip(shard_roots, metadata_items, strict=True):
        shard = metadata["feature_shard"]
        start, stop = int(shard["start"]), int(shard["stop"])
        if int(shard["index"]) != len(shards) or start != expected_start or stop <= start:
            raise ValueError("token shards are not contiguous and ordered")
        tokens = np.load(path / "tokens.npy", allow_pickle=False, mmap_mode="r")
        masks = np.load(path / "masks.npy", allow_pickle=False, mmap_mode="r")
        expected_shape = (stop - start, int(first["token_count"]), int(first["feature_dim"]))
        if tokens.shape != expected_shape or masks.shape != expected_shape[:2]:
            raise ValueError(f"token shard {path.name} has unexpected array shapes")
        shards.append(_TokenShard(start, stop, tokens, masks))
        expected_start = stop
    if expected_start != int(first["feature_count"]):
        raise ValueError("token shards do not cover all features")

    history_indices = np.load(shard_roots[0] / "history_indices.npy", allow_pickle=False, mmap_mode="r")
    labels = np.load(shard_roots[0] / "labels.npy", allow_pickle=False, mmap_mode="r")
    splits = np.load(shard_roots[0] / "splits.npy", allow_pickle=False, mmap_mode="r")
    if history_indices.shape != (int(first["row_count"]), 3):
        raise ValueError("token cache history shape must be [rows, 3]")
    if labels.shape != (int(first["row_count"]),) or splits.shape != labels.shape:
        raise ValueError("token cache labels/splits do not match row count")
    return TokenCache(first, tuple(shards), history_indices, labels, splits)


def train(args: TrainArgs) -> Path:
    if min(
        args.epochs,
        args.batch_size,
        args.eval_batch_size,
        args.eval_steps,
        args.hidden_dim,
        args.query_count,
        args.attention_heads,
        args.temporal_layers,
    ) <= 0:
        raise ValueError("training dimensions and intervals must be positive")
    if args.hidden_dim % args.attention_heads:
        raise ValueError("hidden-dim must be divisible by attention-heads")
    cache = load_token_cache(args.token_cache)
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"refusing to overwrite token-head output: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)

    train_indices = cache.indices("train")
    val_indices = cache.indices("val")
    total_steps = math.ceil(len(train_indices) / args.batch_size) * args.epochs
    warmup_steps = round(total_steps * args.warmup_ratio)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.learning_rate / max(warmup_steps + 1, 1),
        peak_value=args.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=args.min_learning_rate,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )
    config = completion_model.CompletionHeadConfig(
        enabled=True,
        variant="token_query_attention",
        temporal_steps=3,
        hidden_dim=args.hidden_dim,
        query_count=args.query_count,
        attention_heads=args.attention_heads,
        temporal_layers=args.temporal_layers,
        dropout_rate=args.dropout_rate,
    )
    head = completion_model.TokenQueryCompletionHead(
        input_dim=int(cache.metadata["feature_dim"]),
        config=config,
        rngs=nnx.Rngs(args.seed),
    )
    graphdef, params = nnx.split(head)
    parameter_count = sum(int(np.prod(value.value.shape)) for value in params.flat_state().values())
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(current_params, current_opt_state, rng, tokens, masks, targets):
        def loss_fn(candidate_params):
            module = nnx.merge(graphdef, candidate_params)
            logits = module(tokens, masks, rng=rng, train=True)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, targets))

        loss, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
        return optax.apply_updates(current_params, updates), next_opt_state, loss

    @jax.jit
    def predict(current_params, tokens, masks):
        return nnx.merge(graphdef, current_params)(tokens, masks, train=False)

    numpy_rng = np.random.default_rng(args.seed)
    train_rng = jax.random.key(args.seed)
    best_loss = math.inf
    best_path = output / "best"
    global_step = 0
    interval_losses: list[float] = []
    for epoch in range(args.epochs):
        shuffled = numpy_rng.permutation(train_indices)
        for batch_indices in _batches(shuffled, args.batch_size):
            tokens, masks = cache.histories(batch_indices)
            targets = np.asarray(cache.labels[batch_indices], dtype=np.float32)
            train_rng, step_rng = jax.random.split(train_rng)
            params, opt_state, loss = train_step(params, opt_state, step_rng, tokens, masks, targets)
            interval_losses.append(float(loss))
            global_step += 1
            if global_step % args.eval_steps != 0 and global_step != total_steps:
                continue

            val_logits: list[np.ndarray] = []
            val_targets: list[np.ndarray] = []
            for val_batch_indices in _batches(val_indices, args.eval_batch_size):
                val_tokens, val_masks = cache.histories(val_batch_indices)
                val_logits.append(np.asarray(predict(params, val_tokens, val_masks)))
                val_targets.append(np.asarray(cache.labels[val_batch_indices], dtype=np.float32))
            logits = np.concatenate(val_logits)
            targets = np.concatenate(val_targets)
            val_loss = float(np.mean(np.logaddexp(0.0, logits) - targets * logits))
            metrics = {
                "epoch": epoch + 1,
                "step": global_step,
                "train_loss": float(np.mean(interval_losses)),
                "val_loss": val_loss,
                **{f"val_{key}": value for key, value in binary_metrics(logits, targets).items()},
            }
            interval_losses.clear()
            print(json.dumps(metrics, sort_keys=True), flush=True)
            metadata = {
                **metrics,
                "cache_metadata": cache.metadata,
                "parameter_count": parameter_count,
                "head": {
                    "variant": "token_query_attention",
                    "temporal_steps": 3,
                    "hidden_dim": args.hidden_dim,
                    "query_count": args.query_count,
                    "attention_heads": args.attention_heads,
                    "temporal_layers": args.temporal_layers,
                    "dropout_rate": args.dropout_rate,
                },
            }
            checkpoint_path = output / "checkpoints" / f"step_{global_step:06d}"
            _save_head(checkpoint_path, params, metadata)
            if val_loss < best_loss:
                if best_path.exists():
                    shutil.rmtree(best_path)
                best_loss = val_loss
                _save_head(best_path, params, metadata)

    serialized_args = dataclasses.asdict(args)
    serialized_args.update(token_cache=str(args.token_cache), output=str(output))
    serialized_args["xla_flags"] = os.environ["XLA_FLAGS"]
    (output / "training_args.json").write_text(json.dumps(serialized_args, indent=2) + "\n", encoding="utf-8")
    return best_path


def _parse_args(argv: Sequence[str] | None = None) -> TrainArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-cache", type=Path, default=DEFAULT_TOKEN_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--min-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--query-count", type=int, default=32)
    parser.add_argument("--attention-heads", type=int, default=12)
    parser.add_argument("--temporal-layers", type=int, default=3)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return TrainArgs(**vars(parser.parse_args(argv)))


def main(argv: Sequence[str] | None = None) -> None:
    best_path = train(_parse_args(argv))
    print(f"Best token-query head-only checkpoint: {best_path / 'params'}")


if __name__ == "__main__":
    main()
