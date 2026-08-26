"""Evaluate a three-frame raw-prefix completion checkpoint on natural rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from openpi.training import raw_prefix_completion_features as raw_features
from openpi.training import temporal_completion_data as temporal_data
from openpi.training import temporal_completion_metrics as temporal_metrics
from openpi.training import temporal_raw_prefix_completion_features as temporal_raw_features

DEFAULT_CONFIG_NAME = "pi05_agilex_breakfast_temporal_raw_prefix_completion_head"
DEFAULT_BASE_CACHE = Path("/mnt/data/models/wyt/evaluations/current_raw_prefix_tokens_v1")
DEFAULT_HISTORY_CACHE = Path("/mnt/data/models/wyt/evaluations/temporal_raw_prefix_history_v1")


def _bce(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def _metrics(rows: tuple[Any, ...], logits: np.ndarray) -> dict[str, float]:
    labels = np.asarray([int(row.label) for row in rows], dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if labels.shape != logits.shape or logits.size == 0:
        raise ValueError(f"metrics rows/logits mismatch: {len(rows)}/{logits.shape}")
    scores = temporal_metrics.stable_sigmoid(logits)
    ranking = temporal_metrics.binary_ranking_metrics(labels, scores)
    positive = labels == 1
    hard = np.asarray([row.sample_kind == "hard_negative" for row in rows], dtype=np.bool_)
    ordinary = np.asarray([row.sample_kind == "ordinary_negative" for row in rows], dtype=np.bool_)
    paired: list[tuple[float, float]] = []
    events: dict[tuple[str, int, int], dict[str, float]] = {}
    for row, score in zip(rows, scores, strict=True):
        key = (str(row.trajectory_id), int(row.task_index), int(row.boundary_tick))
        values = events.setdefault(key, {})
        if row.sample_kind == "positive":
            values["positive"] = float(score)
        elif row.sample_kind == "hard_negative":
            values["hard"] = float(score)
    for values in events.values():
        if "positive" in values and "hard" in values:
            paired.append((values["positive"], values["hard"]))
    margins = np.asarray([positive_score - hard_score for positive_score, hard_score in paired], dtype=np.float64)
    return {
        "sample_count": float(ranking["sample_count"]),
        "positive_count": float(ranking["positive_count"]),
        "negative_count": float(ranking["negative_count"]),
        "auprc": float(ranking["auprc"]),
        "auroc": float(ranking["roc_auc"]),
        "bce": _bce(logits, labels),
        "positive_score_mean": float(np.mean(scores[positive])) if np.any(positive) else math.nan,
        "hard_negative_score_mean": float(np.mean(scores[hard])) if np.any(hard) else math.nan,
        "ordinary_negative_score_mean": float(np.mean(scores[ordinary])) if np.any(ordinary) else math.nan,
        "positive_hard_paired_ordering_accuracy": (
            float(np.mean([positive_score > hard_score for positive_score, hard_score in paired])) if paired else math.nan
        ),
        "positive_hard_margin_mean": float(np.mean(margins)) if margins.size else math.nan,
        "positive_hard_margin_median": float(np.median(margins)) if margins.size else math.nan,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _predict_logits(
    config: Any,
    checkpoint: Path,
    cache: temporal_raw_features.TemporalRawPrefixHistoryCache,
    indices: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    import flax.nnx as nnx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    import openpi.models.model as model_api  # noqa: PLC0415

    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if getattr(config.model.completion_head, "variant", None) != "temporal_raw_prefix_decoder":
        raise ValueError(
            "evaluation config must use completion_head.variant='temporal_raw_prefix_decoder'"
        )
    params = model_api.restore_params(checkpoint / "params")
    model = config.model.load(params)
    model.eval()
    if int(getattr(model, "prefix_feature_dim", -1)) != int(cache.metadata.input_dim):
        raise ValueError(
            f"checkpoint prefix dim {getattr(model, 'prefix_feature_dim', None)} "
            f"does not match raw-prefix cache dim {cache.metadata.input_dim}"
        )
    graphdef, state = nnx.split(model)

    def score_batch(
        state_value: Any,
        prefix_history: Any,
        prefix_mask_history: Any,
        segment_ids: Any,
        position_ids: Any,
    ) -> Any:
        module = nnx.merge(graphdef, state_value)
        return module.compute_temporal_raw_prefix_completion_logits(
            jax.random.key(0),
            prefix_history,
            prefix_mask_history,
            segment_ids,
            position_ids,
            train=False,
        )

    score_fn = jax.jit(score_batch)
    chunks: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        selected_history, selected_mask_history = cache.features_for_rows(selected)
        logits = np.asarray(
            jax.block_until_ready(
                score_fn(
                    state,
                    jnp.asarray(selected_history),
                    jnp.asarray(selected_mask_history),
                    jnp.asarray(cache.prefix_segment_ids),
                    jnp.asarray(cache.prefix_position_ids),
                )
            ),
            dtype=np.float64,
        ).reshape(-1)
        if logits.shape != (len(selected),) or not np.isfinite(logits).all():
            raise ValueError(f"temporal raw-prefix head returned invalid logits for rows {start}:{start + len(selected)}")
        chunks.append(logits)
    if not chunks:
        raise ValueError("selected split contains no rows")
    return np.concatenate(chunks)


def evaluate(args: argparse.Namespace) -> Path:
    from openpi.training import config as training_config  # noqa: PLC0415

    config = training_config.get_config(args.config_name)
    if not config.completion.uses_temporal_raw_prefix_completion:
        raise ValueError(f"config {args.config_name!r} does not enable temporal raw-prefix completion")
    if args.split not in ("val", "test"):
        raise ValueError(f"unsupported split {args.split!r}")
    manifest_path = Path(config.completion.split_manifest_path).resolve()
    manifest = temporal_data.load_temporal_manifest(manifest_path)
    base_cache_path = args.base_cache.resolve()
    history_cache_path = args.history_cache.resolve()
    base_cache = raw_features.load_raw_prefix_cache(
        base_cache_path,
        manifest=manifest,
        expected_checkpoint_path=config.completion.raw_prefix_source_checkpoint_path,
        expected_model_config_name=config.completion.raw_prefix_source_model_config_name,
    )
    cache = temporal_raw_features.load_temporal_raw_prefix_history(
        history_cache_path,
        manifest=manifest,
        base_cache=base_cache,
        expected_base_cache_path=base_cache_path,
        expected_checkpoint_path=config.completion.raw_prefix_source_checkpoint_path,
        expected_model_config_name=config.completion.raw_prefix_source_model_config_name,
    )
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"completion checkpoint params not found: {checkpoint / 'params'}")
    output = args.output.resolve()
    predictions_path = (
        args.predictions_output.resolve()
        if args.predictions_output is not None
        else output.with_name(output.stem + "_predictions.json")
    )
    protected = {base_cache_path, history_cache_path, manifest_path, checkpoint}
    if output in protected or predictions_path in protected or output == predictions_path:
        raise ValueError("evaluation outputs must not overwrite the cache, manifest, or checkpoint")

    cache_indices = cache.indices_for_split(args.split)
    rows = tuple(cache.rows[int(index)] for index in cache_indices)
    logits = _predict_logits(config, checkpoint, cache, cache_indices, batch_size=args.batch_size)
    scores = temporal_metrics.stable_sigmoid(logits)
    overall = _metrics(rows, logits)
    per_task: dict[str, dict[str, float]] = {}
    for task_index in range(temporal_data.TASKS_PER_TRAJECTORY):
        task_indices = np.asarray(
            [index for index, row in enumerate(rows) if int(row.task_index) == task_index],
            dtype=np.int64,
        )
        task_rows = tuple(rows[int(index)] for index in task_indices)
        per_task[str(task_index)] = _metrics(task_rows, logits[task_indices])
    finite_task_auprc = [value["auprc"] for value in per_task.values() if math.isfinite(value["auprc"])]
    predictions = [
        {
            "trajectory_id": row.trajectory_id,
            "full_episode_id": row.full_episode_id,
            "task_index": row.task_index,
            "logical_tick": row.logical_tick,
            "boundary_tick": row.boundary_tick,
            "source_episode_id": row.source_episode_ids[-1],
            "source_frame_index": row.source_frame_indices[-1],
            "sample_kind": row.sample_kind,
            "target": row.label,
            "logit": float(logit),
            "sigmoid_score": float(score),
        }
        for row, logit, score in zip(rows, logits, scores, strict=True)
    ]
    report = {
        "schema_version": 1,
        "config_name": config.name,
        "completion_checkpoint": str(checkpoint),
        "raw_prefix_cache": str(base_cache_path),
        "raw_prefix_base_cache": str(base_cache_path),
        "temporal_raw_prefix_history_cache": str(history_cache_path),
        "split": args.split,
        "metrics": {
            "overall": overall,
            "macro_task_auprc": float(np.mean(finite_task_auprc)) if finite_task_auprc else math.nan,
            "per_task": per_task,
        },
        "predictions_path": str(predictions_path),
    }
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(_json_safe(predictions), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote temporal raw-prefix completion evaluation report: {output}")
    print(f"Wrote predictions: {predictions_path}")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument("--history-cache", type=Path, default=DEFAULT_HISTORY_CACHE)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


if __name__ == "__main__":
    evaluate(_parser().parse_args())
