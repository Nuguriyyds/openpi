import dataclasses
import functools
import logging
import platform
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
        if not uses_focal and pos_weight is None:
            raise ValueError("pos_weight is required for completion head training")

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


def completion_validation_metrics(
    logits: np.ndarray, targets: np.ndarray, *, prefix: str = "val"
) -> dict[str, float]:
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
    auc = (np.sum(ranks[positives]) - positive_count * (positive_count + 1) / 2) / (
        positive_count * negative_count
    )

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


def evaluate_completion(
    eval_step,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    *,
    expected_count: int,
    metric_prefix: str = "val",
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
    return completion_validation_metrics(logits[:expected_count], targets[:expected_count], prefix=metric_prefix)


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
    if trains_completion_head:
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
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    if uses_completion_data:
        audit_model = nnx.merge(train_state.model_def, train_state.params)
        expected_trainable_groups = ("action",) if completion_stage == "action" else ("completion",)
        audit = _pi0_config.audit_frozen_vlm_parameters(
            audit_model,
            config.freeze_filter,
            trainable_groups=expected_trainable_groups,
        )
        logging.info(
            "Parameter audit: frozen_vlm=%d frozen_action=%d "
            "trainable_action=%d trainable_completion=%d",
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
        if completion_data_info.pos_weight is not None:
            dataset_metrics.update(
                {
                    "dataset/train_positive_count": completion_data_info.train_positive_count,
                    "dataset/train_negative_count": completion_data_info.train_negative_count,
                    "dataset/pos_weight": completion_data_info.pos_weight,
                    "dataset/effective_pos_weight": pos_weight,
                }
            )
        logging.info("Completion dataset metrics: %s", dataset_metrics)
        wandb.log(dataset_metrics, step=0)

    ptrain_step = jax.jit(
        functools.partial(train_step, config, pos_weight=pos_weight),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    pcompletion_eval_step = None
    if trains_completion_head:
        pcompletion_eval_step = jax.jit(
            completion_eval_step,
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
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
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if trains_completion_head and (
            (step + 1) % config.completion.val_interval == 0 or step == config.num_train_steps - 1
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
                )
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
                    completion_data_info.episode_audits[episode_id].frame_count
                    for episode_id in train_episode_ids
                )
                with sharding.set_mesh(mesh):
                    train_eval_metrics = evaluate_completion(
                        pcompletion_eval_step,
                        jax.random.fold_in(train_rng, step + 0xC0A4),
                        train_state,
                        train_eval_data_loader,
                        expected_count=expected_train_eval_count,
                        metric_prefix="train_overfit",
                    )
                pbar.write(
                    f"Step {step} train-overfit: "
                    + ", ".join(f"{key}={value:.4f}" for key, value in train_eval_metrics.items())
                )
                wandb.log(train_eval_metrics, step=step)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
