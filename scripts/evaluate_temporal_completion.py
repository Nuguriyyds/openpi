"""Evaluate the validation-selected temporal completion checkpoint.

This command evaluates the immutable, natural 2 Hz rows from a sealed prefix
feature cache.  It deliberately has no checkpoint-search or test-threshold
search mode:

* ``best_temporal_validation.json`` chooses the one retained checkpoint;
* the stored threshold must declare ``selected_on_split='val'`` and must be
  reproduced from that checkpoint's validation predictions; and
* the test split is passed exactly once to
  ``evaluate_test_with_validation_threshold`` with the frozen validation
  threshold.

All rows use oracle prompts from the manifest.  The resulting report therefore
does not claim closed-loop prompt-switching performance.  An optional paired
current-only baseline is accepted only as a separately fitted, train-only
linear-probe artifact plus a sealed cache with exactly the same rows.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from openpi.training import temporal_completion_data as temporal_data
from openpi.training import temporal_completion_features as temporal_features
from openpi.training import temporal_completion_metrics as temporal_metrics

DEFAULT_CONFIG_NAME = "pi05_agilex_breakfast_temporal_completion_head"
SELECTION_FILE_NAME = "best_temporal_validation.json"
REPORT_SCHEMA_VERSION = 1
CURRENT_PROBE_SCHEMA_VERSION = 1
SELECTION_FLOAT_ATOL = 1.0e-6


@dataclasses.dataclass(frozen=True)
class ValidationArtifact:
    """Validation-only model/threshold selection persisted by training."""

    checkpoint_step: int
    last_validated_checkpoint_step: int
    validation_rank: tuple[float, float, float]
    threshold_selection: temporal_metrics.ThresholdSelection
    feature_cache_schema_version: int
    feature_cache_model_config_name: str
    feature_cache_checkpoint_path: str
    feature_cache_row_count: int
    feature_cache_feature_dim: int
    temporal_input_mode: str
    path: Path


@dataclasses.dataclass(frozen=True)
class CurrentOnlyLinearProbe:
    """Externally fitted deterministic current-prefix linear probe."""

    weight: np.ndarray
    bias: float
    metadata: Mapping[str, Any]

    def logits(self, current_features: np.ndarray) -> np.ndarray:
        features = np.asarray(current_features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.weight.shape[0]:
            raise ValueError(f"current-only features must have shape [N, {self.weight.shape[0]}], got {features.shape}")
        if not np.isfinite(features).all():
            raise ValueError("current-only features contain non-finite values")
        return features @ self.weight + self.bias


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required JSON file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _parse_threshold_selection(value: Any) -> temporal_metrics.ThresholdSelection:
    if not isinstance(value, dict):
        raise ValueError("threshold_selection must be a JSON object")
    expected_fields = {field.name for field in dataclasses.fields(temporal_metrics.ThresholdSelection)}
    if set(value) != expected_fields:
        raise ValueError(
            "threshold_selection fields do not match schema; "
            f"missing={sorted(expected_fields - set(value))}, unexpected={sorted(set(value) - expected_fields)}"
        )
    if value["selected_on_split"] != "val":
        raise ValueError("best_temporal_validation threshold must originate from validation")
    if value["rule"] != temporal_metrics.THRESHOLD_SELECTION_RULE:
        raise ValueError("best_temporal_validation threshold selection rule is stale or unrecognised")

    integer_fields = (
        "validation_event_count",
        "validation_early_trigger_events",
        "candidate_count",
    )
    for field in integer_fields:
        if isinstance(value[field], bool) or not isinstance(value[field], int):
            raise ValueError(f"threshold_selection.{field} must be an integer")
    if value["validation_event_count"] <= 0 or value["candidate_count"] <= 0:
        raise ValueError("threshold selection requires positive validation event and candidate counts")
    if not 0 <= value["validation_early_trigger_events"] <= value["validation_event_count"]:
        raise ValueError("threshold selection has an invalid early-trigger event count")

    numeric_fields = ("threshold", "validation_event_recall", "validation_event_f1")
    for field in numeric_fields:
        if isinstance(value[field], bool) or not isinstance(value[field], (int, float)):
            raise ValueError(f"threshold_selection.{field} must be numeric")
        if not math.isfinite(float(value[field])):
            raise ValueError(f"threshold_selection.{field} must be finite")
    if not 0.0 <= float(value["validation_event_recall"]) <= 1.0:
        raise ValueError("threshold selection validation_event_recall must be in [0, 1]")
    if not 0.0 <= float(value["validation_event_f1"]) <= 1.0:
        raise ValueError("threshold selection validation_event_f1 must be in [0, 1]")

    selection = temporal_metrics.ThresholdSelection(**value)
    # Match the metric implementation's explicit abstention sentinel domain.
    if not 0.0 <= selection.threshold <= float(np.nextafter(1.0, np.inf)):
        raise ValueError("threshold_selection.threshold is outside the supported probability range")
    return selection


def load_validation_artifact(
    checkpoint_root: Path,
    *,
    feature_cache_schema_version: int,
    feature_cache_model_config_name: str,
    feature_cache_checkpoint_path: str,
    feature_cache_row_count: int,
    feature_cache_feature_dim: int,
    temporal_input_mode: str,
) -> ValidationArtifact:
    """Loads the mandatory validation artifact and audits its data binding."""

    root = checkpoint_root.resolve()
    path = root / SELECTION_FILE_NAME
    value = _read_json_object(path)
    required_fields = {
        "checkpoint_step",
        "last_validated_checkpoint_step",
        "validation_rank",
        "threshold_selection",
        "feature_cache_schema_version",
        "feature_cache_model_config_name",
        "feature_cache_checkpoint_path",
        "feature_cache_row_count",
        "feature_cache_feature_dim",
        "temporal_input_mode",
    }
    if not required_fields.issubset(value):
        raise ValueError(f"{path} is missing required fields: {sorted(required_fields - set(value))}")

    step = value["checkpoint_step"]
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("best_temporal_validation checkpoint_step must be a positive integer")
    last_validated_step = value["last_validated_checkpoint_step"]
    if isinstance(last_validated_step, bool) or not isinstance(last_validated_step, int) or last_validated_step <= 0:
        raise ValueError("best_temporal_validation last_validated_checkpoint_step must be a positive integer")
    if last_validated_step < step:
        raise ValueError("last_validated_checkpoint_step cannot precede the selected checkpoint_step")
    rank_value = value["validation_rank"]
    if not isinstance(rank_value, list) or len(rank_value) != 3:
        raise ValueError("best_temporal_validation validation_rank must contain exactly three numbers")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in rank_value):
        raise ValueError("best_temporal_validation validation_rank must contain only numbers")
    if not all(math.isfinite(float(item)) for item in rank_value):
        raise ValueError("best_temporal_validation validation_rank must contain only finite numbers")

    expected_cache_binding = {
        "feature_cache_schema_version": feature_cache_schema_version,
        "feature_cache_model_config_name": feature_cache_model_config_name,
        "feature_cache_checkpoint_path": feature_cache_checkpoint_path,
        "feature_cache_row_count": feature_cache_row_count,
        "feature_cache_feature_dim": feature_cache_feature_dim,
    }
    for field, expected in expected_cache_binding.items():
        if value[field] != expected:
            raise ValueError(f"best_temporal_validation {field} does not match the requested feature cache")
    if temporal_input_mode not in ("history", "current_only"):
        raise ValueError(f"unsupported evaluator temporal_input_mode: {temporal_input_mode!r}")
    stored_input_mode = value["temporal_input_mode"]
    if stored_input_mode not in ("history", "current_only"):
        raise ValueError("best_temporal_validation has an unsupported temporal_input_mode")
    if stored_input_mode != temporal_input_mode:
        raise ValueError("best_temporal_validation temporal_input_mode does not match the requested evaluation config")

    return ValidationArtifact(
        checkpoint_step=step,
        last_validated_checkpoint_step=last_validated_step,
        validation_rank=tuple(float(item) for item in rank_value),  # type: ignore[arg-type]
        threshold_selection=_parse_threshold_selection(value["threshold_selection"]),
        feature_cache_schema_version=int(value["feature_cache_schema_version"]),
        feature_cache_model_config_name=str(value["feature_cache_model_config_name"]),
        feature_cache_checkpoint_path=str(value["feature_cache_checkpoint_path"]),
        feature_cache_row_count=int(value["feature_cache_row_count"]),
        feature_cache_feature_dim=int(value["feature_cache_feature_dim"]),
        temporal_input_mode=stored_input_mode,
        path=path,
    )


def resolve_validation_checkpoint(checkpoint_root: Path, artifact: ValidationArtifact) -> Path:
    """Resolves only the exact retained step chosen by validation.

    There is intentionally no ``latest`` fallback, numeric-directory scan, or
    protected-copy fallback.  If training did not retain the validation winner,
    test evaluation stops rather than choosing a different model.
    """

    root = checkpoint_root.resolve()
    if artifact.path.resolve() != root / SELECTION_FILE_NAME:
        raise ValueError("validation artifact does not belong to the requested checkpoint root")
    checkpoint = root / str(artifact.checkpoint_step)
    if not checkpoint.is_dir() or not (checkpoint / "params").is_dir():
        raise FileNotFoundError(
            "validation-selected retained checkpoint is unavailable: "
            f"step={artifact.checkpoint_step}, expected={checkpoint / 'params'}"
        )
    return checkpoint


def require_completed_temporal_run(
    checkpoint_root: Path,
    artifact: ValidationArtifact,
    *,
    expected_final_step: int,
) -> Path:
    """Rejects test evaluation of a partial or crashed training run."""

    if isinstance(expected_final_step, bool) or not isinstance(expected_final_step, int) or expected_final_step <= 0:
        raise ValueError("temporal evaluation expected_final_step must be a positive integer")
    if artifact.last_validated_checkpoint_step != expected_final_step:
        raise ValueError(
            "temporal validation artifact does not prove a completed training run: "
            f"last_validated={artifact.last_validated_checkpoint_step}, expected_final={expected_final_step}"
        )
    final_checkpoint = checkpoint_root.resolve() / str(expected_final_step)
    if not final_checkpoint.is_dir() or not (final_checkpoint / "params").is_dir():
        raise FileNotFoundError(
            f"completed temporal run is missing its final checkpoint params: expected={final_checkpoint / 'params'}"
        )
    return final_checkpoint


def _load_sealed_manifest(path: Path) -> temporal_data.TemporalCompletionManifest:
    """Loads and validates the temporal manifest schema and contents."""

    return temporal_data.TemporalCompletionManifest.from_dict(_read_json_object(path))


def _verify_recomputed_selection(
    stored: temporal_metrics.ThresholdSelection,
    recomputed: temporal_metrics.ThresholdSelection,
) -> None:
    """Requires the retained checkpoint to reproduce its val selection."""

    if stored.selected_on_split != "val" or recomputed.selected_on_split != "val":
        raise ValueError("test threshold must originate from validation")
    exact_fields = (
        "validation_event_count",
        "validation_early_trigger_events",
        "candidate_count",
        "rule",
    )
    for field in exact_fields:
        if getattr(stored, field) != getattr(recomputed, field):
            raise ValueError(f"validation-selected threshold verification failed for {field}")
    float_fields = ("threshold", "validation_event_recall", "validation_event_f1")
    for field in float_fields:
        if not math.isclose(
            float(getattr(stored, field)),
            float(getattr(recomputed, field)),
            rel_tol=0.0,
            abs_tol=SELECTION_FLOAT_ATOL,
        ):
            raise ValueError(f"validation-selected threshold verification failed for {field}")


def _metrics_by_task(report: temporal_metrics.TemporalCompletionReport) -> dict[str, Any]:
    overall: dict[str, float] = {}
    per_task: dict[str, dict[str, float]] = {
        str(task_index): {} for task_index in range(temporal_data.TASKS_PER_TRAJECTORY)
    }
    for key, value in report.metrics.items():
        head, separator, tail = key.partition("/")
        if separator and head.startswith("task_") and head[5:].isdigit():
            task_index = int(head[5:])
            if task_index in range(temporal_data.TASKS_PER_TRAJECTORY):
                per_task[str(task_index)][tail] = float(value)
                continue
        overall[key] = float(value)
    return {"overall": overall, "per_task": per_task}


def evaluate_oracle_prompt_scores(
    *,
    validation_rows: Sequence[temporal_data.TemporalSampleRow],
    validation_scores: np.ndarray,
    test_rows: Sequence[temporal_data.TemporalSampleRow],
    test_scores: np.ndarray,
    stored_selection: temporal_metrics.ThresholdSelection,
) -> dict[str, Any]:
    """Verifies val selection, then performs one fail-closed test evaluation."""

    validation = verify_oracle_validation_scores(
        validation_rows=validation_rows,
        validation_scores=validation_scores,
        stored_selection=stored_selection,
    )
    test = evaluate_oracle_test_scores(
        test_rows=test_rows,
        test_scores=test_scores,
        stored_selection=stored_selection,
    )
    return {"validation": validation, "test": test}


def verify_oracle_validation_scores(
    *,
    validation_rows: Sequence[temporal_data.TemporalSampleRow],
    validation_scores: np.ndarray,
    stored_selection: temporal_metrics.ThresholdSelection,
) -> dict[str, Any]:
    """Reproduces the sealed validation selection before test is touched."""

    if {row.split for row in validation_rows} != {"val"}:
        raise ValueError("oracle validation evaluation requires only val rows")
    recomputed = temporal_metrics.select_validation_threshold(validation_rows, validation_scores)
    _verify_recomputed_selection(stored_selection, recomputed)
    validation_report = temporal_metrics.evaluate_temporal_completion(
        validation_rows,
        scores=validation_scores,
        threshold=stored_selection,
    )
    return _metrics_by_task(validation_report)


def evaluate_oracle_test_scores(
    *,
    test_rows: Sequence[temporal_data.TemporalSampleRow],
    test_scores: np.ndarray,
    stored_selection: temporal_metrics.ThresholdSelection,
) -> dict[str, Any]:
    """Performs the sole test metric call after validation is verified."""

    if {row.split for row in test_rows} != {"test"}:
        raise ValueError("oracle test evaluation requires only test rows")
    # This is the sole test-set metric call in this evaluation path.  The API
    # itself rejects raw/test-selected thresholds and never searches test.
    test_report = temporal_metrics.evaluate_test_with_validation_threshold(
        test_rows,
        stored_selection,
        scores=test_scores,
    )
    return _metrics_by_task(test_report)


def _split_cache(
    cache: temporal_features.TemporalFeatureCache,
    split: temporal_data.SplitName,
) -> tuple[tuple[temporal_data.TemporalSampleRow, ...], np.ndarray]:
    indices = cache.indices_for_split(split)
    if indices.size == 0:
        raise ValueError(f"sealed temporal cache has no {split!r} rows")
    rows = tuple(cache.rows[int(index)] for index in indices)
    history = np.asarray(cache.prefix_history[indices], dtype=np.float32)
    return rows, history


def _predict_temporal_logits(
    model: Any,
    histories: np.ndarray,
    *,
    batch_size: int,
    input_mode: str,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if histories.ndim != 3 or histories.shape[1] != temporal_data.TEMPORAL_HISTORY_STEPS:
        raise ValueError(f"temporal histories must have shape [N, 3, D], got {histories.shape}")

    # Heavy JAX/NNX imports remain inside the actual inference path so schema
    # and checkpoint-selection tests stay CPU-only and lightweight.
    import flax.nnx as nnx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from openpi.training import completion as completion_training  # noqa: PLC0415

    graphdef, state = nnx.split(model)

    def score_batch(state: Any, history: Any) -> Any:
        module = nnx.merge(graphdef, state)
        return module.compute_temporal_completion_logits(jax.random.key(0), history, train=False)

    score_fn = jax.jit(score_batch)
    chunks: list[np.ndarray] = []
    for start in range(0, histories.shape[0], batch_size):
        batch = completion_training.apply_temporal_input_mode(
            jnp.asarray(histories[start : start + batch_size], dtype=jnp.float32),
            input_mode,
        )
        logits = np.asarray(jax.block_until_ready(score_fn(state, batch)), dtype=np.float64).reshape(-1)
        if logits.shape != (len(batch),) or not np.isfinite(logits).all():
            raise ValueError(f"temporal head returned invalid logits for batch starting at row {start}")
        chunks.append(logits)
    return np.concatenate(chunks)


def _load_temporal_model(config: Any, checkpoint: Path, *, feature_dim: int) -> Any:
    import openpi.models.model as model_api  # noqa: PLC0415

    if not bool(getattr(config.completion, "uses_temporal_completion", False)):
        raise ValueError(f"config {config.name!r} is not a temporal completion-head config")
    completion_head = getattr(config.model, "completion_head", None)
    if getattr(completion_head, "variant", None) != "temporal_mlp":
        raise ValueError("evaluation config must use completion_head.variant='temporal_mlp'")
    params = model_api.restore_params(checkpoint / "params")
    model = config.model.load(params)
    model.eval()
    if int(getattr(model, "prefix_feature_dim", -1)) != feature_dim:
        raise ValueError(
            f"checkpoint temporal feature dim {getattr(model, 'prefix_feature_dim', None)} "
            f"does not match cache dim {feature_dim}"
        )
    if not hasattr(model, "compute_temporal_completion_logits"):
        raise ValueError("loaded checkpoint lacks compute_temporal_completion_logits")
    return model


def _load_current_only_probe(
    path: Path,
    *,
    cache: temporal_features.TemporalFeatureCache,
) -> CurrentOnlyLinearProbe:
    if not path.is_file():
        raise FileNotFoundError(f"current-only probe weights not found: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        expected_arrays = {"metadata_json", "weight", "bias"}
        if set(arrays.files) != expected_arrays:
            raise ValueError(
                "current-only probe artifact fields do not match schema; "
                f"missing={sorted(expected_arrays - set(arrays.files))}, "
                f"unexpected={sorted(set(arrays.files) - expected_arrays)}"
            )
        metadata_value = arrays["metadata_json"]
        if metadata_value.ndim != 0:
            raise ValueError("current-only probe metadata_json must be a scalar string")
        metadata = json.loads(str(metadata_value.item()))
        weight = np.asarray(arrays["weight"], dtype=np.float64)
        bias_value = np.asarray(arrays["bias"], dtype=np.float64)
    if not isinstance(metadata, dict):
        raise ValueError("current-only probe metadata_json must encode an object")
    expected_metadata = {
        "schema_version",
        "probe_type",
        "fit_split",
        "feature_cache_schema_version",
        "feature_cache_model_config_name",
        "feature_cache_checkpoint_path",
        "feature_cache_row_count",
        "feature_dim",
    }
    if set(metadata) != expected_metadata:
        raise ValueError(
            "current-only probe metadata fields do not match schema; "
            f"missing={sorted(expected_metadata - set(metadata))}, unexpected={sorted(set(metadata) - expected_metadata)}"
        )
    if metadata["schema_version"] != CURRENT_PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported current-only probe schema_version")
    if metadata["probe_type"] != "deterministic_linear_logit" or metadata["fit_split"] != "train":
        raise ValueError("current-only ablation requires a deterministic linear-logit probe fitted only on train")
    expected_bindings = {
        "feature_cache_schema_version": cache.metadata.schema_version,
        "feature_cache_model_config_name": cache.metadata.model_config_name,
        "feature_cache_checkpoint_path": cache.metadata.checkpoint_path,
        "feature_cache_row_count": cache.metadata.row_count,
        "feature_dim": cache.metadata.feature_dim,
    }
    for field, expected in expected_bindings.items():
        if metadata[field] != expected:
            raise ValueError(f"current-only probe {field} does not match its sealed feature cache")
    if weight.shape != (cache.metadata.feature_dim,) or bias_value.ndim != 0:
        raise ValueError(
            f"current-only probe requires weight [{cache.metadata.feature_dim}] and scalar bias, "
            f"got {weight.shape}/{bias_value.shape}"
        )
    if not np.isfinite(weight).all() or not np.isfinite(bias_value):
        raise ValueError("current-only probe weights must be finite")
    return CurrentOnlyLinearProbe(weight=weight, bias=float(bias_value), metadata=metadata)


def _evaluate_current_only(
    *,
    reference_cache: temporal_features.TemporalFeatureCache,
    current_cache: temporal_features.TemporalFeatureCache,
    probe: CurrentOnlyLinearProbe,
) -> dict[str, Any]:
    if current_cache.rows != reference_cache.rows:
        raise ValueError("current-only ablation cache must contain exactly the same canonical rows in the same order")
    validation_rows, validation_history = _split_cache(current_cache, "val")
    test_rows, test_history = _split_cache(current_cache, "test")
    validation_scores = temporal_metrics.stable_sigmoid(probe.logits(validation_history[:, -1, :]))
    test_scores = temporal_metrics.stable_sigmoid(probe.logits(test_history[:, -1, :]))
    selection = temporal_metrics.select_validation_threshold(validation_rows, validation_scores)
    result = evaluate_oracle_prompt_scores(
        validation_rows=validation_rows,
        validation_scores=validation_scores,
        test_rows=test_rows,
        test_scores=test_scores,
        stored_selection=selection,
    )
    return {
        "enabled": True,
        "probe_type": "deterministic_linear_logit",
        "fit_split": "train",
        "paired_row_count": len(current_cache.rows),
        "source_cache": {
            "schema_version": current_cache.metadata.schema_version,
            "model_config_name": current_cache.metadata.model_config_name,
            "checkpoint_path": current_cache.metadata.checkpoint_path,
            "row_count": current_cache.metadata.row_count,
        },
        "threshold_selection": dataclasses.asdict(selection),
        **result,
    }


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    output = path.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing evaluation report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_safe(report), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_evaluation(args: argparse.Namespace) -> Path:
    if (args.current_only_cache is None) != (args.current_only_probe_weights is None):
        raise ValueError("current-only ablation requires both --current-only-cache and --current-only-probe-weights")

    # Importing the full training config initializes the model/JAX stack, so do
    # it only after cheap CLI consistency checks.
    from openpi.training import config as training_config  # noqa: PLC0415

    config = training_config.get_config(args.config_name)
    if not config.completion.uses_temporal_completion:
        raise ValueError(f"config {args.config_name!r} does not enable temporal completion")
    manifest_path = Path(args.manifest or config.completion.split_manifest_path).resolve()
    feature_cache_path = Path(args.feature_cache or config.completion.temporal_feature_cache_path).resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    output = args.output.resolve()
    flattened_inputs: set[Path] = {manifest_path, feature_cache_path, checkpoint_root / SELECTION_FILE_NAME}
    for optional in (args.current_only_cache, args.current_only_probe_weights):
        if optional is not None:
            flattened_inputs.add(optional.resolve())
    if output in flattened_inputs:
        raise ValueError("output report path must not overwrite any evaluation input")

    manifest = _load_sealed_manifest(manifest_path)
    cache = temporal_features.load_temporal_feature_cache(
        feature_cache_path,
        manifest=manifest,
        expected_checkpoint_path=config.completion.temporal_source_checkpoint_path,
        expected_model_config_name=config.completion.temporal_source_model_config_name,
    )
    artifact = load_validation_artifact(
        checkpoint_root,
        feature_cache_schema_version=cache.metadata.schema_version,
        feature_cache_model_config_name=cache.metadata.model_config_name,
        feature_cache_checkpoint_path=cache.metadata.checkpoint_path,
        feature_cache_row_count=cache.metadata.row_count,
        feature_cache_feature_dim=cache.metadata.feature_dim,
        temporal_input_mode=config.completion.temporal_input_mode,
    )
    final_checkpoint = require_completed_temporal_run(
        checkpoint_root,
        artifact,
        expected_final_step=config.num_train_steps,
    )
    checkpoint = resolve_validation_checkpoint(checkpoint_root, artifact)

    model = _load_temporal_model(config, checkpoint, feature_dim=cache.metadata.feature_dim)
    batch_size = config.batch_size if args.batch_size is None else args.batch_size
    validation_rows, validation_history = _split_cache(cache, "val")
    test_rows, test_history = _split_cache(cache, "test")
    validation_logits = _predict_temporal_logits(
        model,
        validation_history,
        batch_size=batch_size,
        input_mode=config.completion.temporal_input_mode,
    )
    validation_metrics = verify_oracle_validation_scores(
        validation_rows=validation_rows,
        validation_scores=temporal_metrics.stable_sigmoid(validation_logits),
        stored_selection=artifact.threshold_selection,
    )
    # No test feature is scored until the persisted validation threshold has
    # been reproduced exactly for this checkpoint.
    test_logits = _predict_temporal_logits(
        model,
        test_history,
        batch_size=batch_size,
        input_mode=config.completion.temporal_input_mode,
    )
    test_metrics = evaluate_oracle_test_scores(
        test_rows=test_rows,
        test_scores=temporal_metrics.stable_sigmoid(test_logits),
        stored_selection=artifact.threshold_selection,
    )
    oracle_prompt = {"validation": validation_metrics, "test": test_metrics}

    current_only: dict[str, Any] = {
        "enabled": False,
        "reason": "supply both a sealed cache and separately fitted train-only probe weights to enable",
    }
    if args.current_only_cache is not None and args.current_only_probe_weights is not None:
        current_cache = temporal_features.load_temporal_feature_cache(
            args.current_only_cache.resolve(),
            manifest=manifest,
            expected_checkpoint_path=cache.metadata.checkpoint_path,
            expected_model_config_name=cache.metadata.model_config_name,
        )
        probe = _load_current_only_probe(args.current_only_probe_weights.resolve(), cache=current_cache)
        current_only = _evaluate_current_only(
            reference_cache=cache,
            current_cache=current_cache,
            probe=probe,
        )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol": {
            "evaluation_mode": "oracle_prompt_natural_2hz",
            "threshold_source": "validation",
            "checkpoint_source": SELECTION_FILE_NAME,
            "test_threshold_search": False,
            "retained_checkpoint_test_evaluation_passes": 1,
            "temporal_input_mode": artifact.temporal_input_mode,
            "closed_loop_evaluated": False,
            "closed_loop_note": (
                "Rows use manifest oracle prompts; run a separate controller rollout before claiming closed-loop results."
            ),
        },
        "bindings": {
            "config_name": config.name,
            "manifest_path": str(manifest_path),
            "feature_cache_path": str(feature_cache_path),
            "feature_cache_schema_version": cache.metadata.schema_version,
            "feature_cache_model_config_name": cache.metadata.model_config_name,
            "feature_cache_checkpoint_path": cache.metadata.checkpoint_path,
            "feature_cache_row_count": cache.metadata.row_count,
            "feature_cache_feature_dim": cache.metadata.feature_dim,
            "temporal_input_mode": artifact.temporal_input_mode,
            "validation_selection_path": str(artifact.path),
            "checkpoint_step": artifact.checkpoint_step,
            "last_validated_checkpoint_step": artifact.last_validated_checkpoint_step,
            "final_checkpoint_path": str(final_checkpoint),
            "checkpoint_path": str(checkpoint),
        },
        "threshold_selection": dataclasses.asdict(artifact.threshold_selection),
        "validation_rank": artifact.validation_rank,
        "oracle_prompt": oracle_prompt,
        "current_only_paired_ablation": current_only,
    }
    _write_report(output, report)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        required=True,
        help="Experiment directory containing best_temporal_validation.json and retained numeric checkpoints.",
    )
    parser.add_argument("--manifest", type=Path, help="Defaults to completion.split_manifest_path from the config.")
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="Defaults to completion.temporal_feature_cache_path from the config.",
    )
    parser.add_argument("--output", type=Path, required=True, help="New JSON report path; existing files are refused.")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Inference batch size; defaults to the training config batch size for closest numerical reproduction.",
    )
    parser.add_argument(
        "--current-only-cache",
        type=Path,
        help="Optional sealed temporal cache; its current (last) prefix is evaluated on exactly the reference rows.",
    )
    parser.add_argument(
        "--current-only-probe-weights",
        type=Path,
        help="Optional .npz deterministic_linear_logit probe fitted only on the train split.",
    )
    return parser


def main() -> None:
    output = run_evaluation(_parser().parse_args())
    print(f"Wrote temporal completion evaluation report: {output}")


if __name__ == "__main__":
    main()
