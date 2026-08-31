"""Train a joint progress and done adapter from cached VLA prefix tokens."""

# ruff: noqa: E402, I001 -- importing the done trainer configures XLA before JAX imports.

from __future__ import annotations

import argparse
from collections.abc import Sequence
import dataclasses
import json
import math
import os
from pathlib import Path
import shutil

try:
    from scripts import train_token_done_head as done_training
except ModuleNotFoundError:
    import train_token_done_head as done_training

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import completion as completion_model
from openpi.models import progress_done
from openpi.training import breakfast_done_data


DEFAULT_ANNOTATION_ROOT = Path(
    "/home/geek/share3/breakfest_data/rule_split_330-2_action_state_delay05_cut2dist10_overlap10/split"
)
DEFAULT_OUTPUT = Path("/home/geek/share3/vla_done/v2/progress_done_head_h768_seed42_20260831")


@dataclasses.dataclass(frozen=True)
class TrainArgs:
    token_cache: Path = done_training.DEFAULT_TOKEN_CACHE
    annotation_root: Path = DEFAULT_ANNOTATION_ROOT
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
    progress_loss_weight: float = 1.0
    progress_huber_delta: float = 0.1
    seed: int = 42
    max_steps: int | None = None
    overwrite: bool = False


def build_progress_targets(
    cache_root: Path,
    cache_metadata: dict[str, object],
    annotation_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild progress labels and verify they align with every cached row."""

    dataset_root = Path(str(cache_metadata["dataset_root"]))
    dataset = breakfast_done_data.load_breakfast_done_dataset(dataset_root, annotation_root)
    cached_ids = np.load(cache_root.resolve() / "shard_0" / "sample_ids.npy", allow_pickle=False)
    rebuilt_ids = np.asarray([sample.sample_id for sample in dataset.samples])
    if not np.array_equal(cached_ids, rebuilt_ids):
        raise ValueError("cached rows do not match samples rebuilt from the boundary annotations")

    episode_by_id = {episode.index: episode for episode in dataset.episodes}
    task_indices = {task_id: index for index, task_id in enumerate(breakfast_done_data.SUB_TASK_IDS)}
    targets = np.zeros(len(dataset.samples), dtype=np.float32)
    valid = np.ones(len(dataset.samples), dtype=np.bool_)
    for row, sample in enumerate(dataset.samples):
        episode = episode_by_id[sample.episode_index]
        task_index = task_indices[sample.current_sub_task]
        start = episode.stage_starts[task_index]
        if task_index + 1 < len(episode.stage_starts):
            end = episode.stage_starts[task_index + 1]
        elif episode.terminal_frame is not None:
            end = episode.terminal_frame
        else:
            valid[row] = False
            continue
        targets[row] = np.clip((sample.query_frame - start) / (end - start), 0.0, 1.0)
    return targets, valid


def progress_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    errors = np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def _validate_args(args: TrainArgs) -> None:
    dimensions = (
        args.epochs,
        args.batch_size,
        args.eval_batch_size,
        args.eval_steps,
        args.hidden_dim,
        args.query_count,
        args.attention_heads,
        args.temporal_layers,
    )
    if min(dimensions) <= 0:
        raise ValueError("training dimensions and intervals must be positive")
    if args.progress_loss_weight < 0 or args.progress_huber_delta <= 0:
        raise ValueError("progress loss settings must be non-negative and positive, respectively")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    if args.hidden_dim % args.attention_heads:
        raise ValueError("hidden-dim must be divisible by attention-heads")


def train(args: TrainArgs) -> Path:
    _validate_args(args)
    cache = done_training.load_token_cache(args.token_cache)
    progress_targets, progress_valid = build_progress_targets(
        args.token_cache, cache.metadata, args.annotation_root
    )
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"refusing to overwrite progress-head output: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=False)

    train_indices = cache.indices("train")
    val_indices = cache.indices("val")
    epoch_steps = math.ceil(len(train_indices) / args.batch_size)
    full_steps = epoch_steps * args.epochs
    total_steps = full_steps if args.max_steps is None else min(full_steps, args.max_steps)
    warmup_steps = round(full_steps * args.warmup_ratio)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=args.learning_rate / max(warmup_steps + 1, 1),
        peak_value=args.learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=full_steps,
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
    head = progress_done.TokenQueryProgressDoneHead(
        input_dim=int(cache.metadata["feature_dim"]), config=config, rngs=nnx.Rngs(args.seed)
    )
    graphdef, params = nnx.split(head)
    parameter_count = sum(int(np.prod(value.value.shape)) for value in params.flat_state().values())
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(current_params, current_opt_state, rng, tokens, masks, done_targets, targets, valid):
        def loss_fn(candidate_params):
            module = nnx.merge(graphdef, candidate_params)
            done_logits, progress_logits = module(tokens, masks, rng=rng, train=True)
            done_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(done_logits, done_targets))
            progress = jax.nn.sigmoid(progress_logits)
            row_loss = optax.huber_loss(progress, targets, delta=args.progress_huber_delta)
            progress_loss = jnp.sum(row_loss * valid) / jnp.maximum(jnp.sum(valid), 1.0)
            progress_loss /= args.progress_huber_delta
            return done_loss + args.progress_loss_weight * progress_loss, (done_loss, progress_loss)

        (loss, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(current_params)
        updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
        return optax.apply_updates(current_params, updates), next_opt_state, loss, losses

    @jax.jit
    def predict(current_params, tokens, masks):
        return nnx.merge(graphdef, current_params)(tokens, masks, train=False)

    numpy_rng = np.random.default_rng(args.seed)
    train_rng = jax.random.key(args.seed)
    best_loss = math.inf
    best_path = output / "best"
    global_step = 0
    interval_losses: list[tuple[float, float, float]] = []
    stop = False
    for epoch in range(args.epochs):
        shuffled = numpy_rng.permutation(train_indices)
        for batch_indices in done_training._batches(shuffled, args.batch_size):  # noqa: SLF001
            tokens, masks = cache.histories(batch_indices)
            done_targets = np.asarray(cache.labels[batch_indices], dtype=np.float32)
            targets = np.asarray(progress_targets[batch_indices], dtype=np.float32)
            valid = np.asarray(progress_valid[batch_indices], dtype=np.float32)
            train_rng, step_rng = jax.random.split(train_rng)
            params, opt_state, loss, losses = train_step(
                params, opt_state, step_rng, tokens, masks, done_targets, targets, valid
            )
            interval_losses.append((float(loss), float(losses[0]), float(losses[1])))
            global_step += 1
            if global_step % args.eval_steps != 0 and global_step != total_steps:
                continue

            val_done_logits: list[np.ndarray] = []
            val_progress: list[np.ndarray] = []
            for val_batch_indices in done_training._batches(val_indices, args.eval_batch_size):  # noqa: SLF001
                tokens, masks = cache.histories(val_batch_indices)
                done_logits, progress_logits = predict(params, tokens, masks)
                val_done_logits.append(np.asarray(done_logits))
                val_progress.append(np.asarray(jax.nn.sigmoid(progress_logits)))
            done_logits = np.concatenate(val_done_logits)
            progress = np.concatenate(val_progress)
            done_targets = np.asarray(cache.labels[val_indices], dtype=np.float32)
            targets = progress_targets[val_indices]
            valid = progress_valid[val_indices]
            done_loss = float(np.mean(np.logaddexp(0.0, done_logits) - done_targets * done_logits))
            progress_loss = float(
                np.mean(optax.huber_loss(progress[valid], targets[valid], delta=args.progress_huber_delta))
                / args.progress_huber_delta
            )
            losses_array = np.asarray(interval_losses)
            metrics = {
                "epoch": epoch + 1,
                "step": global_step,
                "train_loss": float(np.mean(losses_array[:, 0])),
                "train_done_loss": float(np.mean(losses_array[:, 1])),
                "train_progress_loss": float(np.mean(losses_array[:, 2])),
                "val_loss": done_loss + args.progress_loss_weight * progress_loss,
                "val_done_loss": done_loss,
                "val_progress_loss": progress_loss,
                **{
                    f"val_done_{key}": value
                    for key, value in done_training.binary_metrics(done_logits, done_targets).items()
                },
                **{
                    f"val_progress_{key}": value
                    for key, value in progress_metrics(progress[valid], targets[valid]).items()
                },
            }
            interval_losses.clear()
            print(json.dumps(metrics, sort_keys=True), flush=True)
            metadata = {
                **metrics,
                "cache_metadata": cache.metadata,
                "parameter_count": parameter_count,
                "progress_valid_count": int(np.sum(progress_valid)),
                "progress_loss_weight": args.progress_loss_weight,
                "progress_huber_delta": args.progress_huber_delta,
                "head": {
                    "variant": "token_query_progress_done",
                    "temporal_steps": 3,
                    "hidden_dim": args.hidden_dim,
                    "query_count": args.query_count,
                    "attention_heads": args.attention_heads,
                    "temporal_layers": args.temporal_layers,
                    "dropout_rate": args.dropout_rate,
                },
            }
            checkpoint_path = output / "checkpoints" / f"step_{global_step:06d}"
            done_training._save_head(checkpoint_path, params, metadata)  # noqa: SLF001
            if metrics["val_loss"] < best_loss:
                if best_path.exists():
                    shutil.rmtree(best_path)
                best_loss = metrics["val_loss"]
                done_training._save_head(best_path, params, metadata)  # noqa: SLF001
            if global_step == total_steps:
                stop = True
                break
        if stop:
            break

    serialized_args = dataclasses.asdict(args)
    serialized_args.update(
        token_cache=str(args.token_cache),
        annotation_root=str(args.annotation_root),
        output=str(output),
        xla_flags=os.environ["XLA_FLAGS"],
    )
    (output / "training_args.json").write_text(json.dumps(serialized_args, indent=2) + "\n", encoding="utf-8")
    return best_path


def _parse_args(argv: Sequence[str] | None = None) -> TrainArgs:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-cache", type=Path, default=done_training.DEFAULT_TOKEN_CACHE)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
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
    parser.add_argument("--progress-loss-weight", type=float, default=1.0)
    parser.add_argument("--progress-huber-delta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return TrainArgs(**vars(parser.parse_args(argv)))


def main(argv: Sequence[str] | None = None) -> None:
    best_path = train(_parse_args(argv))
    print(f"Best progress+done head checkpoint: {best_path / 'params'}")


if __name__ == "__main__":
    main()
