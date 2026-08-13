import dataclasses
import functools
import json
import logging
import platform
import shutil
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.models.pi0_config as _pi0_config
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.completion as _completion
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _frozen_param_dtype_filter(config: _config.TrainConfig) -> nnx.filterlib.Filter:
    """Returns the frozen params that should be stored in bfloat16."""

    if config.completion.stage == "head":
        if not isinstance(config.model, _pi0_config.Pi0Config):
            raise ValueError("completion head training is only supported for Pi0Config models")
        return config.model.get_vlm_freeze_filter()
    return config.freeze_filter


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    if config.completion.stage == "head":
        tx = _optimizer.create_completion_head_optimizer(
            config.completion,
            decay_steps=config.num_train_steps,
        )
    else:
        tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    dtype_filter = _frozen_param_dtype_filter(config)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, dtype_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        trainable_params = params.filter(config.trainable_filter)

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(trainable_params),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def _jax_correlation(left: jax.Array, right: jax.Array) -> jax.Array:
    """Finite Pearson correlation for a fixed-size JAX batch."""

    left = jnp.asarray(left, dtype=jnp.float32).reshape(-1)
    right = jnp.asarray(right, dtype=jnp.float32).reshape(-1)
    left_centered = left - jnp.mean(left)
    right_centered = right - jnp.mean(right)
    denominator = jnp.sqrt(jnp.sum(jnp.square(left_centered)) * jnp.sum(jnp.square(right_centered)))
    return jnp.where(
        denominator > jnp.finfo(jnp.float32).eps, jnp.sum(left_centered * right_centered) / denominator, 0.0
    )


def _jax_ordinal_ranks(values: jax.Array) -> jax.Array:
    """Ranks used for per-batch Spearman logging (ties are deterministically ordered)."""

    values = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
    return jnp.argsort(jnp.argsort(values)).astype(jnp.float32)


def progress_batch_metrics(predictions: jax.Array, targets: jax.Array) -> dict[str, jax.Array]:
    """Regression metrics emitted from the JITted progress training step."""

    predictions = jnp.asarray(predictions, dtype=jnp.float32).reshape(-1)
    targets = jnp.asarray(targets, dtype=jnp.float32).reshape(-1)
    absolute_error = jnp.abs(predictions - targets)
    early = targets <= 0.1
    late = targets >= 0.9

    def endpoint_mae(mask: jax.Array) -> jax.Array:
        count = jnp.sum(mask.astype(jnp.float32))
        return jnp.sum(jnp.where(mask, absolute_error, 0.0)) / jnp.maximum(count, 1.0)

    return {
        "mae": jnp.mean(absolute_error),
        "rmse": jnp.sqrt(jnp.mean(jnp.square(predictions - targets))),
        "pearson": _jax_correlation(predictions, targets),
        "spearman": _jax_correlation(_jax_ordinal_ranks(predictions), _jax_ordinal_ranks(targets)),
        "prediction_mean": jnp.mean(predictions),
        "prediction_std": jnp.std(predictions),
        "prediction_min": jnp.min(predictions),
        "prediction_max": jnp.max(predictions),
        "target_mean": jnp.mean(targets),
        "target_std": jnp.std(targets),
        "target_min": jnp.min(targets),
        "target_max": jnp.max(targets),
        "early_mae": endpoint_mae(early),
        "late_mae": endpoint_mae(late),
    }


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions] | tuple[_model.Observation, _model.Actions, at.Array],
    *,
    pos_weight: float | None = None,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    train_rng = jax.random.fold_in(rng, state.step)
    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    completion_stage = config.completion.stage

    if completion_stage == "head":
        observation, _actions, completion_targets = batch
        completion_targets = jnp.asarray(completion_targets, dtype=jnp.float32)
        if completion_targets.ndim == 2 and completion_targets.shape[-1] == 1:
            completion_targets = completion_targets[..., 0]

        focal_gamma = config.completion.focal_gamma
        focal_alpha = config.completion.focal_alpha
        uses_focal = config.completion.uses_focal_loss
        uses_progress = config.completion.uses_progress_objective
        if not uses_progress and not uses_focal and pos_weight is None:
            raise ValueError("pos_weight is required for completion head training")

        if uses_progress:

            def progress_loss_fn(model, rng, observation, targets):
                logits = model.compute_completion_logits(rng, observation, train=True)
                if logits.shape != targets.shape:
                    raise ValueError(
                        f"progress target shape {targets.shape} does not match model logits shape {logits.shape}"
                    )
                predictions = _completion.progress_predictions_from_logits(logits)
                loss = jnp.mean(_completion.progress_huber_loss(logits, targets, delta=config.completion.huber_delta))
                return loss, progress_batch_metrics(predictions, targets)

            (loss, progress_metrics), grads = nnx.value_and_grad(
                progress_loss_fn,
                argnums=diff_state,
                has_aux=True,
            )(model, train_rng, observation, completion_targets)
            completion_loss = loss
        else:

            def completion_loss_fn(model, rng, observation, targets):
                logits = model.compute_completion_logits(rng, observation, train=True)
                if logits.shape != targets.shape:
                    raise ValueError(
                        f"completion target shape {targets.shape} does not match model logits shape {logits.shape}"
                    )
                if uses_focal:
                    return jnp.mean(
                        _completion.focal_loss_with_logits(logits, targets, gamma=focal_gamma, alpha=focal_alpha)
                    )
                return jnp.mean(_completion.weighted_bce_with_logits(logits, targets, pos_weight))

            loss, grads = nnx.value_and_grad(
                completion_loss_fn,
                argnums=diff_state,
            )(model, train_rng, observation, completion_targets)
            completion_loss = loss
    else:
        observation, actions = batch

        @at.typecheck
        def action_loss_fn(
            model: _model.BaseModel,
            rng: at.KeyArrayLike,
            observation: _model.Observation,
            actions: _model.Actions,
        ):
            if config.training_time_rtc.enabled:
                chunked_loss = model.compute_loss(
                    rng, observation, actions, train=True, training_time_rtc=config.training_time_rtc
                )
            else:
                chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            return jnp.mean(chunked_loss)

        loss, grads = nnx.value_and_grad(action_loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    if completion_stage == "action":
        info.update(
            {
                "action_loss": loss,
                "total_loss": loss,
                "action_grad_norm": optax.global_norm(grads),
            }
        )
    elif completion_stage == "head":
        if uses_progress:
            head_info = {
                "progress_loss": completion_loss,
                "total_loss": loss,
                "completion_grad_norm": optax.global_norm(grads),
                **{f"progress_{name}": value for name, value in progress_metrics.items()},
            }
        else:
            head_info = {
                "completion_loss": completion_loss,
                "total_loss": loss,
                "completion_grad_norm": optax.global_norm(grads),
                "completion_positive_count": jnp.sum(completion_targets),
                "completion_positive_fraction": jnp.mean(completion_targets),
            }
            if uses_focal:
                head_info["focal_gamma"] = jnp.asarray(focal_gamma, dtype=jnp.float32)
                head_info["focal_alpha"] = jnp.asarray(focal_alpha, dtype=jnp.float32)
            else:
                head_info["pos_weight"] = jnp.asarray(pos_weight, dtype=jnp.float32)
        info.update(head_info)
    return new_state, info


def completion_eval_step(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions, at.Array],
) -> tuple[jax.Array, jax.Array]:
    """Computes validation logits without evaluating the action objective."""

    eval_params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, eval_params)
    model.eval()
    observation, _, targets = batch
    targets = jnp.asarray(targets, dtype=jnp.float32)
    if targets.ndim == 2 and targets.shape[-1] == 1:
        targets = targets[..., 0]
    logits = model.compute_completion_logits(rng, observation, train=False)
    if logits.shape != targets.shape:
        raise ValueError(f"completion target shape {targets.shape} does not match logits shape {logits.shape}")
    return logits, targets


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank assignment for ties, matching scipy.stats.rankdata (method='average')."""

    values = np.asarray(values, dtype=np.float64)
    sorter = np.argsort(values, kind="mergesort")
    # inv[i] = position of values[i] in the sorted array (0-indexed).
    inv = np.empty(sorter.size, dtype=np.int64)
    inv[sorter] = np.arange(sorter.size, dtype=np.int64)
    sorted_values = values[sorter]
    # Identify tie groups in sorted order.
    is_tie_start = np.concatenate(([True], sorted_values[1:] != sorted_values[:-1]))
    sorted_group_ids = np.cumsum(is_tie_start) - 1
    # Map group ids back to original positions.
    group_ids = sorted_group_ids[inv]
    # Average rank within each tie group: mean of (1-indexed) sorted positions.
    counts = np.bincount(sorted_group_ids)
    rank_sums = np.zeros(counts.size, dtype=np.float64)
    np.add.at(rank_sums, sorted_group_ids, np.arange(1, sorter.size + 1, dtype=np.float64))
    avg_ranks = rank_sums / np.maximum(counts, 1)
    return avg_ranks[group_ids]


def completion_validation_metrics(logits: np.ndarray, targets: np.ndarray, *, prefix: str = "val") -> dict[str, float]:
    """Aggregates the fixed-threshold completion metrics requested for validation.

    In addition to the fixed 0.5-threshold precision/recall/f1, this also reports
    the best-F1 threshold and the ROC-AUC. The fixed-threshold metrics can be
    misleading when the head's logit distribution is shifted away from 0 (so that
    all sigmoid scores fall on one side of 0.5); the best-threshold and AUC
    metrics disentangle "the head learned nothing" from "the threshold is wrong".
    """

    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    if logits.shape != targets.shape or logits.size == 0:
        raise ValueError(f"invalid validation arrays: logits={logits.shape}, targets={targets.shape}")
    if not np.all(np.logical_or(targets == 0, targets == 1)):
        raise ValueError("validation completion targets must contain only 0/1")
    scores = np.empty_like(logits)
    nonnegative = logits >= 0
    scores[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exp_logits = np.exp(logits[~nonnegative])
    scores[~nonnegative] = exp_logits / (1.0 + exp_logits)
    predictions = scores >= 0.5
    positives = targets == 1
    negatives = ~positives
    positive_count = int(np.sum(positives))
    negative_count = int(np.sum(negatives))
    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            f"validation split must contain both classes, got positive={positive_count}, negative={negative_count}"
        )
    true_positives = int(np.sum(np.logical_and(predictions, positives)))
    false_positives = int(np.sum(np.logical_and(predictions, negatives)))
    false_negatives = int(np.sum(np.logical_and(~predictions, positives)))
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, np.finfo(np.float64).eps)
    bce = np.mean(np.logaddexp(0.0, logits) - targets * logits)

    # Best-threshold F1: sweep over unique scores plus 0.5 to find the threshold
    # that maximises F1. This reveals whether the head has learned a useful
    # ordering even when the fixed 0.5 threshold yields zero precision/recall.
    candidate_thresholds = np.unique(scores)
    best_f1 = 0.0
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    for threshold in candidate_thresholds:
        preds = scores >= threshold
        tp = int(np.sum(np.logical_and(preds, positives)))
        fp = int(np.sum(np.logical_and(preds, negatives)))
        fn = int(np.sum(np.logical_and(~preds, positives)))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2.0 * p * r / max(p + r, np.finfo(np.float64).eps)
        if f > best_f1:
            best_f1 = f
            best_threshold = float(threshold)
            best_precision = p
            best_recall = r

    # ROC-AUC computed by the Mann-Whitney U statistic (rank-based, threshold-free).
    ranks = _rankdata(scores)
    auc = (np.sum(ranks[positives]) - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)

    return {
        f"{prefix}/bce": float(bce),
        f"{prefix}/positive_count": float(positive_count),
        f"{prefix}/negative_count": float(negative_count),
        f"{prefix}/positive_score_mean": float(np.mean(scores[positives])),
        f"{prefix}/negative_score_mean": float(np.mean(scores[negatives])),
        f"{prefix}/precision_at_0.5": float(precision),
        f"{prefix}/recall_at_0.5": float(recall),
        f"{prefix}/f1_at_0.5": float(f1),
        f"{prefix}/best_f1": float(best_f1),
        f"{prefix}/best_threshold": float(best_threshold),
        f"{prefix}/best_precision": float(best_precision),
        f"{prefix}/best_recall": float(best_recall),
        f"{prefix}/auc": float(auc),
    }


def _sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    scores = np.empty_like(logits)
    nonnegative = logits >= 0
    scores[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exp_logits = np.exp(logits[~nonnegative])
    scores[~nonnegative] = exp_logits / (1.0 + exp_logits)
    return scores


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(left) < 2:
        return 0.0
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(np.sqrt(np.sum(np.square(left_centered)) * np.sum(np.square(right_centered))))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.sum(left_centered * right_centered) / denominator)


def progress_validation_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    *,
    huber_delta: float,
    prefix: str = "val",
) -> dict[str, float]:
    """Aggregates continuous progress-regression metrics without binary thresholds."""

    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if logits.shape != targets.shape or logits.size == 0:
        raise ValueError(f"invalid progress validation arrays: logits={logits.shape}, targets={targets.shape}")
    if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(targets)):
        raise ValueError("progress validation logits and targets must be finite")
    if np.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("progress validation targets must lie in [0, 1]")
    if huber_delta <= 0:
        raise ValueError("progress Huber delta must be positive")

    predictions = _sigmoid_numpy(logits)
    absolute_error = np.abs(predictions - targets)
    huber = np.where(
        absolute_error <= huber_delta,
        0.5 * np.square(absolute_error),
        huber_delta * (absolute_error - 0.5 * huber_delta),
    )
    early = targets <= 0.1
    late = targets >= 0.9
    return {
        f"{prefix}/loss": float(np.mean(huber)),
        f"{prefix}/mae": float(np.mean(absolute_error)),
        f"{prefix}/rmse": float(np.sqrt(np.mean(np.square(predictions - targets)))),
        f"{prefix}/pearson": _safe_correlation(predictions, targets),
        f"{prefix}/spearman": _safe_correlation(_rankdata(predictions), _rankdata(targets)),
        f"{prefix}/prediction_mean": float(np.mean(predictions)),
        f"{prefix}/prediction_std": float(np.std(predictions)),
        f"{prefix}/prediction_min": float(np.min(predictions)),
        f"{prefix}/prediction_max": float(np.max(predictions)),
        f"{prefix}/target_mean": float(np.mean(targets)),
        f"{prefix}/target_std": float(np.std(targets)),
        f"{prefix}/target_min": float(np.min(targets)),
        f"{prefix}/target_max": float(np.max(targets)),
        f"{prefix}/early_mae": float(np.mean(absolute_error[early])) if np.any(early) else 0.0,
        f"{prefix}/late_mae": float(np.mean(absolute_error[late])) if np.any(late) else 0.0,
        f"{prefix}/frame_count": float(targets.size),
    }


def evaluate_completion(
    eval_step,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    *,
    expected_count: int,
    metric_prefix: str = "val",
    objective: _completion.CompletionObjective = "binary",
    huber_delta: float = 0.1,
) -> dict[str, float]:
    all_logits = []
    all_targets = []
    for batch_index, batch in enumerate(data_loader):
        logits, targets = eval_step(jax.random.fold_in(rng, batch_index), state, batch)
        all_logits.append(np.asarray(jax.device_get(logits)))
        all_targets.append(np.asarray(jax.device_get(targets)))
    if not all_logits:
        raise ValueError("validation loader produced no completion examples")
    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    if logits.shape[0] < expected_count:
        raise ValueError(f"validation loader evaluated {logits.shape[0]} frames, expected at least {expected_count}")
    # The loader may repeat a few rows so the final batch can be sharded across
    # all devices. Only the original validation examples contribute metrics.
    if objective == "progress":
        return progress_validation_metrics(
            logits[:expected_count],
            targets[:expected_count],
            huber_delta=huber_delta,
            prefix=metric_prefix,
        )
    if objective == "binary":
        return completion_validation_metrics(logits[:expected_count], targets[:expected_count], prefix=metric_prefix)
    raise ValueError(f"unsupported completion objective: {objective!r}")


def compute_epoch_total_steps(steps_per_epoch: int, epochs: int) -> int:
    """Total optimizer steps for epoch-based boundary training.

    ``epochs`` is already validated to 1 or 2 by ``CompletionTrainingConfig``;
    this function re-validates so it can be unit-tested independently.
    """

    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
    if epochs not in (1, 2):
        raise ValueError(f"epochs must be 1 or 2, got {epochs}")
    return steps_per_epoch * epochs


def compute_boundary_total_steps(
    steps_per_epoch: int,
    *,
    epochs: int | None,
    train_steps: int | None,
) -> int:
    """Resolves an integer-epoch or explicit-step boundary training budget."""

    if (epochs is None) == (train_steps is None):
        raise ValueError("exactly one of epochs or train_steps must be set")
    if train_steps is not None:
        if train_steps <= 0:
            raise ValueError("train_steps must be positive")
        return train_steps
    assert epochs is not None
    return compute_epoch_total_steps(steps_per_epoch, epochs)


def should_save_epoch_checkpoint(
    completed_steps: int,
    *,
    steps_per_epoch: int,
    total_steps: int,
    save_interval: int = 200,
) -> bool:
    """Whether to checkpoint after ``completed_steps`` optimizer steps.

    Saves at every ``save_interval`` completed steps, at each epoch boundary, and
    at the final step. ``completed_steps`` is the *completed* step count (==
    ``train_state.step`` after the update), so the checkpoint directory name
    matches the state it holds -- no off-by-one between the dir name and the
    restored ``step`` on resume.
    """

    if completed_steps <= 0:
        return False
    if completed_steps >= total_steps:
        return True
    if completed_steps % steps_per_epoch == 0:
        return True
    return save_interval > 0 and completed_steps % save_interval == 0


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.training_time_rtc.enabled and config.model.model_type not in (
        _model.ModelType.PI0,
        _model.ModelType.PI05,
    ):
        raise ValueError("training_time_rtc is only supported by Pi0/Pi0.5 JAX models.")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    completion_stage = config.completion.stage
    uses_completion_data = config.completion.uses_completion_data
    trains_completion_head = config.completion.trains_completion_head
    completion_data_info = None
    val_data_loader = None
    train_eval_data_loader = None
    pos_weight = None
    if uses_completion_data:
        completion_data_info = _data_loader.prepare_completion_data(config)
    if trains_completion_head:
        assert completion_data_info is not None
        if not config.completion.uses_progress_objective:
            pos_weight = (
                config.completion.bce_pos_weight_override
                if config.completion.bce_pos_weight_override is not None
                else completion_data_info.pos_weight
            )
            if not config.completion.uses_focal_loss and pos_weight is None:
                raise ValueError("completion head training requires audited train labels and pos_weight")

    data_loader = _data_loader.create_data_loader(
        config,
        split="train",
        completion_data_info=completion_data_info,
        sharding=data_sharding,
        shuffle=True,
    )

    # Budgeted boundary training: derive steps_per_epoch from the real sampler,
    # then resolve either an integer-epoch or explicit-step budget. Override
    # num_train_steps so the LR schedule, loop, and checkpoints agree.
    is_epoch_based = config.completion.is_epoch_based
    steps_per_epoch: int | None = None
    total_steps = config.num_train_steps
    eval_checkpoint_step: int | None = None
    if is_epoch_based:
        boundary_sampler = data_loader.boundary_sampler
        if boundary_sampler is None:
            raise ValueError(
                "budgeted boundary training (completion.epochs or train_steps set) requires the "
                "boundary sampler; ensure boundary_sampling=True and a boundary-labeled dataset"
            )
        steps_per_epoch = boundary_sampler.steps_per_epoch
        total_steps = compute_boundary_total_steps(
            steps_per_epoch,
            epochs=config.completion.epochs,
            train_steps=config.completion.train_steps,
        )
        config = dataclasses.replace(config, num_train_steps=total_steps)
        eval_checkpoint_step = config.completion.eval_checkpoint_step or (
            total_steps if config.completion.train_steps is not None else steps_per_epoch
        )
        logging.info(
            "Boundary training: steps_per_epoch=%d epochs=%s configured_train_steps=%s "
            "total_steps=%d eval_checkpoint_step=%d train_samples=%d",
            steps_per_epoch,
            config.completion.epochs,
            config.completion.train_steps,
            total_steps,
            eval_checkpoint_step,
            boundary_sampler.num_samples,
        )

    # The val split is disabled for boundary training (val_groups=0); the merged
    # test split is evaluated only offline, never during training.
    if trains_completion_head and config.completion.val_groups > 0:
        val_data_loader = _data_loader.create_data_loader(
            config,
            split="val",
            completion_data_info=completion_data_info,
            sharding=data_sharding,
            shuffle=False,
        )
        if config.completion.train_episode_limit is not None:
            train_eval_data_loader = _data_loader.create_data_loader(
                config,
                split="train",
                completion_data_info=completion_data_info,
                sharding=data_sharding,
                shuffle=False,
                natural_train_eval=True,
            )

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    start_step = int(train_state.step)

    # For epoch-based training, restore the sampler to the correct epoch and
    # fast-forward past already-consumed batches *before* the first batch is
    # drawn, so a resumed run continues the exact deterministic sequence.
    if is_epoch_based:
        assert steps_per_epoch is not None
        boundary_sampler = data_loader.boundary_sampler
        assert boundary_sampler is not None
        resume_epoch = start_step // steps_per_epoch
        skip_batches = start_step % steps_per_epoch
        boundary_sampler.set_epoch(resume_epoch)
        boundary_sampler.set_skip_batches(skip_batches)
        logging.info(
            "Boundary sampler resume: start_step=%d epoch=%d skip_batches=%d",
            start_step,
            resume_epoch,
            skip_batches,
        )

    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    if uses_completion_data:
        audit_model = nnx.merge(train_state.model_def, train_state.params)
        expected_trainable_groups = ("action",) if completion_stage == "action" else ("completion",)
        audit = _pi0_config.audit_frozen_vlm_parameters(
            audit_model,
            config.freeze_filter,
            trainable_groups=expected_trainable_groups,
        )
        logging.info(
            "Parameter audit: frozen_vlm=%d frozen_action=%d trainable_action=%d trainable_completion=%d",
            len(audit.frozen_vlm),
            len(audit.frozen_action),
            len(audit.trainable_action),
            len(audit.trainable_completion),
        )
        assert completion_data_info is not None
        dataset_metrics = {
            "dataset/train_episode_count": len(completion_data_info.manifest.episode_ids("train")),
            "dataset/val_episode_count": len(completion_data_info.manifest.episode_ids("val")),
            "dataset/test_episode_count": len(completion_data_info.manifest.episode_ids("test")),
        }
        if config.completion.uses_progress_objective:
            dataset_metrics.update(
                {
                    "dataset/progress_objective": 1.0,
                    "dataset/progress_bin_count": _completion.PROGRESS_BIN_COUNT,
                    "dataset/progress_huber_delta": config.completion.huber_delta,
                }
            )
        elif completion_data_info.pos_weight is not None:
            dataset_metrics.update(
                {
                    "dataset/train_positive_count": completion_data_info.train_positive_count,
                    "dataset/train_negative_count": completion_data_info.train_negative_count,
                    "dataset/pos_weight": completion_data_info.pos_weight,
                    "dataset/effective_pos_weight": pos_weight,
                }
            )
        if is_epoch_based:
            assert steps_per_epoch is not None
            boundary_audits = completion_data_info.boundary_episode_audits or {}
            train_ids = completion_data_info.manifest.episode_ids("train")
            sampled_positive = sum(boundary_audits[eid].positive_count for eid in train_ids if eid in boundary_audits)
            boundary_copy_positive = sum(
                boundary_audits[eid].boundary_copy_count for eid in train_ids if eid in boundary_audits
            )
            assert data_loader.boundary_sampler is not None
            sampled_total = data_loader.boundary_sampler.num_samples
            sampled_negative = sampled_total - sampled_positive
            dataset_metrics.update(
                {
                    "dataset/steps_per_epoch": steps_per_epoch,
                    "dataset/total_steps": total_steps,
                    "dataset/effective_epochs": total_steps / steps_per_epoch,
                    "dataset/eval_checkpoint_step": eval_checkpoint_step,
                    "dataset/train_sampled_positive_count": sampled_positive,
                    "dataset/train_sampled_negative_count": sampled_negative,
                    "dataset/train_sampled_pos_neg_ratio": (
                        sampled_negative / sampled_positive if sampled_positive > 0 else 0.0
                    ),
                    "dataset/boundary_copy_positive_count": boundary_copy_positive,
                }
            )
            if config.completion.epochs is not None:
                dataset_metrics["dataset/epochs"] = config.completion.epochs
            if config.completion.train_steps is not None:
                dataset_metrics["dataset/configured_train_steps"] = config.completion.train_steps
        logging.info("Completion dataset metrics: %s", dataset_metrics)
        wandb.log(dataset_metrics, step=0)

    # Predetermine the single test checkpoint before any training metric exists,
    # so test data can never influence checkpoint choice.
    if is_epoch_based:
        assert eval_checkpoint_step is not None
        eval_ckpt_path = epath.Path(config.checkpoint_dir) / "eval_checkpoint.json"
        eval_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        eval_ckpt_path.write_text(
            json.dumps(
                {
                    "eval_checkpoint_step": eval_checkpoint_step,
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total_steps,
                    "epochs": config.completion.epochs,
                    "train_steps": config.completion.train_steps,
                    "effective_epochs": total_steps / steps_per_epoch,
                }
            ),
            encoding="utf-8",
        )
        wandb.log({"checkpoint/eval_step": eval_checkpoint_step}, step=0)
        logging.info("Predetermined eval checkpoint step: %d", eval_checkpoint_step)

    ptrain_step = jax.jit(
        functools.partial(train_step, config, pos_weight=pos_weight),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    pcompletion_eval_step = None
    if trains_completion_head and config.completion.val_groups > 0:
        pcompletion_eval_step = jax.jit(
            completion_eval_step,
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        )

    pbar = tqdm.tqdm(
        range(start_step, total_steps),
        initial=start_step,
        total=total_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            if is_epoch_based:
                log_payload = dict(reduced_info)
                log_payload["train/epoch"] = step // steps_per_epoch if steps_per_epoch else 0
                log_payload["train/step"] = step
                wandb.log(log_payload, step=step)
            else:
                wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (
            not is_epoch_based
            and trains_completion_head
            and config.completion.val_groups > 0
            and ((step + 1) % config.completion.val_interval == 0 or step == total_steps - 1)
        ):
            assert pcompletion_eval_step is not None
            assert val_data_loader is not None
            assert completion_data_info is not None
            expected_val_count = sum(
                completion_data_info.episode_audits[episode_id].frame_count
                for episode_id in completion_data_info.manifest.episode_ids("val")
            )
            with sharding.set_mesh(mesh):
                val_metrics = evaluate_completion(
                    pcompletion_eval_step,
                    jax.random.fold_in(train_rng, step + 1),
                    train_state,
                    val_data_loader,
                    expected_count=expected_val_count,
                    objective=config.completion.objective,
                    huber_delta=config.completion.huber_delta,
                )
            if config.completion.uses_progress_objective:
                actual_val_count = int(val_metrics["val/frame_count"])
            else:
                actual_val_count = int(val_metrics["val/positive_count"] + val_metrics["val/negative_count"])
            if actual_val_count != expected_val_count:
                raise ValueError(
                    f"validation loader evaluated {actual_val_count} frames, expected {expected_val_count}"
                )
            pbar.write(
                f"Step {step} validation: " + ", ".join(f"{key}={value:.4f}" for key, value in val_metrics.items())
            )
            wandb.log(val_metrics, step=step)

            if train_eval_data_loader is not None:
                train_episode_ids = completion_data_info.manifest.episode_ids("train")[
                    : config.completion.train_episode_limit
                ]
                expected_train_eval_count = sum(
                    completion_data_info.episode_audits[episode_id].frame_count for episode_id in train_episode_ids
                )
                with sharding.set_mesh(mesh):
                    train_eval_metrics = evaluate_completion(
                        pcompletion_eval_step,
                        jax.random.fold_in(train_rng, step + 0xC0A4),
                        train_state,
                        train_eval_data_loader,
                        expected_count=expected_train_eval_count,
                        metric_prefix="train_overfit",
                        objective=config.completion.objective,
                        huber_delta=config.completion.huber_delta,
                    )
                pbar.write(
                    f"Step {step} train-overfit: "
                    + ", ".join(f"{key}={value:.4f}" for key, value in train_eval_metrics.items())
                )
                wandb.log(train_eval_metrics, step=step)

        if is_epoch_based:
            completed = int(train_state.step)
            if should_save_epoch_checkpoint(
                completed,
                steps_per_epoch=steps_per_epoch if steps_per_epoch is not None else 1,
                total_steps=total_steps,
                save_interval=config.save_interval,
            ):
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, completed)
                wandb.log({"checkpoint/saved_step": completed}, step=step)
                # Protect a predetermined non-final eval checkpoint from
                # max_to_keep cleanup by copying it outside the manager's
                # numeric checkpoint directories.
                if eval_checkpoint_step is not None and completed == eval_checkpoint_step and completed < total_steps:
                    checkpoint_manager.wait_until_finished()
                    src = epath.Path(config.checkpoint_dir) / str(completed)
                    dst = epath.Path(config.checkpoint_dir) / "eval_checkpoint" / str(completed)
                    if src.is_dir():
                        if dst.exists():
                            dst.rmtree()
                        shutil.copytree(str(src), str(dst))
                        # Write a step marker inside the protected copy so the
                        # evaluator can verify it matches the requested step
                        # (P1-A: prevent the protected copy from masquerading
                        # as a different step).
                        (dst / "_protected_step.json").write_text(json.dumps({"step": completed}), encoding="utf-8")
                        logging.info("Protected eval checkpoint copy: %s -> %s", src, dst)
        elif (step % config.save_interval == 0 and step > start_step) or step == total_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()

    # P1: Assert the predetermined eval checkpoint survived training.
    if is_epoch_based and eval_checkpoint_step is not None:
        eval_ckpt_managed = epath.Path(config.checkpoint_dir) / str(eval_checkpoint_step)
        eval_ckpt_protected = epath.Path(config.checkpoint_dir) / "eval_checkpoint" / str(eval_checkpoint_step)
        if not eval_ckpt_managed.is_dir() and not eval_ckpt_protected.is_dir():
            raise FileNotFoundError(
                f"Predetermined eval checkpoint (step {eval_checkpoint_step}) was deleted by the "
                f"checkpoint manager (max_to_keep=1) and no protected copy exists. "
                f"This should not happen — please report this bug."
            )


if __name__ == "__main__":
    main(_config.cli())
