"""Fit the strict current-prefix linear-probe ablation for temporal completion.

Protocol (version 1)
--------------------

* Read the same subtask-reverse temporal prefix cache and manifest as the temporal MLP.
* Use only ``prefix_history[:, -1, :]``.  Every natural train row is used once
  in each deterministic full-batch objective; no temporal history is exposed.
* Fit L2-regularised logistic probes for the fixed ``L2_CANDIDATES`` below.
  ``FIT_SEED`` fixes the (tiny) optimiser initialisation.  Select L2 solely by
  natural-validation binary log loss, with stronger L2 winning exact ties.
* Test rows are never indexed.  They remain reserved for the one paired test
  evaluation in ``evaluate_temporal_completion.py``.
* Emit exactly ``CURRENT_PROBE_SCHEMA_VERSION == 1``.  The evaluator deliberately
  rejects extra arrays or metadata, so the immutable protocol constants in this
  source file are the authoritative selection recipe.

The output is a raw-feature-coordinate linear logit: ``x @ weight + bias``.
Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any, Final

import numpy as np

from openpi.training import temporal_completion_data as temporal_data
from openpi.training import temporal_completion_features as temporal_features

DEFAULT_CONFIG_NAME: Final = "pi05_agilex_breakfast_temporal_completion_head"
CURRENT_PROBE_SCHEMA_VERSION: Final = 1

# These values are protocol, not CLI knobs.  Keeping model selection fixed makes
# independently fitted artifacts directly reproducible and prevents post-hoc
# tuning after seeing test results.
FIT_SEED: Final = 42
L2_CANDIDATES: Final[tuple[float, ...]] = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
MAX_OPTIMIZER_ITERATIONS: Final = 500
OPTIMIZER_GTOL: Final = 1.0e-6
OPTIMIZER_FTOL: Final = 1.0e-10
CONSTANT_FEATURE_EPSILON: Final = 1.0e-12


@dataclasses.dataclass(frozen=True)
class ProbeFitResult:
    """Deterministic fit result; selection diagnostics are intentionally not serialized."""

    weight: np.ndarray
    bias: float
    selected_l2: float
    validation_log_loss: float
    optimizer_iterations: int


def _require_binary_split(labels: np.ndarray, *, split: str) -> None:
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError(f"current-only probe requires non-empty one-dimensional {split} labels")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError(f"current-only probe {split} labels must be exactly binary")
    if np.unique(labels).size != 2:
        raise ValueError(f"current-only probe {split} split must contain both binary classes")


def _current_split(
    cache: temporal_features.TemporalFeatureCache,
    split: temporal_data.SplitName,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns one natural split using only its current (last) prefix."""

    if split == "test":
        raise ValueError("probe fitting is forbidden from indexing test rows")
    indices = cache.indices_for_split(split)
    if indices.size == 0:
        raise ValueError(f"sealed temporal cache has no {split!r} rows")
    current = np.asarray(cache.prefix_history[indices, -1, :], dtype=np.float64)
    labels = np.asarray([cache.rows[int(index)].label for index in indices], dtype=np.float64)
    if current.ndim != 2 or current.shape[0] != labels.size:
        raise ValueError(f"current-only {split} features have an invalid shape: {current.shape}")
    if not np.isfinite(current).all():
        raise ValueError(f"current-only {split} features contain non-finite values")
    _require_binary_split(labels, split=split)
    return current, labels


def _stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    probabilities = np.empty_like(values)
    nonnegative = values >= 0.0
    probabilities[nonnegative] = 1.0 / (1.0 + np.exp(-values[nonnegative]))
    exponent = np.exp(values[~nonnegative])
    probabilities[~nonnegative] = exponent / (1.0 + exponent)
    return probabilities


def _binary_log_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if values.shape != targets.shape:
        raise ValueError(f"logits/labels shape mismatch: {values.shape} != {targets.shape}")
    loss = float(np.mean(np.logaddexp(0.0, values) - targets * values))
    if not math.isfinite(loss):
        raise ValueError("current-only probe produced non-finite validation log loss")
    return loss


def _fit_l2_candidate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    l2: float,
    candidate_index: int,
) -> tuple[np.ndarray, float, int]:
    """Fits one standardized full-batch convex logistic objective with L-BFGS."""

    # SciPy is an existing JAX runtime dependency.  Keeping the import inside
    # the numerical path leaves --help and schema tests lightweight.
    from scipy import optimize  # noqa: PLC0415

    if not math.isfinite(l2) or l2 <= 0.0:
        raise ValueError("L2 candidate must be finite and positive")
    feature_mean = np.mean(train_features, axis=0, dtype=np.float64)
    feature_scale = np.std(train_features, axis=0, dtype=np.float64)
    feature_scale = np.where(feature_scale > CONSTANT_FEATURE_EPSILON, feature_scale, 1.0)
    standardized = np.asarray((train_features - feature_mean) / feature_scale, dtype=np.float64, order="C")
    sample_count, feature_dim = standardized.shape

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weight = parameters[:-1]
        bias = parameters[-1]
        logits = standardized @ weight + bias
        residual = _stable_sigmoid(logits) - train_labels
        value = np.mean(np.logaddexp(0.0, logits) - train_labels * logits) + 0.5 * l2 * np.dot(weight, weight)
        gradient = np.empty_like(parameters)
        gradient[:-1] = standardized.T @ residual / sample_count + l2 * weight
        gradient[-1] = np.mean(residual)
        return float(value), gradient

    rng = np.random.default_rng(np.random.SeedSequence([FIT_SEED, candidate_index]))
    initial = rng.normal(loc=0.0, scale=1.0e-8, size=feature_dim + 1)
    optimum = optimize.minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": MAX_OPTIMIZER_ITERATIONS,
            "gtol": OPTIMIZER_GTOL,
            "ftol": OPTIMIZER_FTOL,
            "maxls": 50,
        },
    )
    if not optimum.success:
        raise RuntimeError(f"L-BFGS failed for L2={l2:g}: {optimum.message}")
    standardized_parameters = np.asarray(optimum.x, dtype=np.float64)
    if standardized_parameters.shape != (feature_dim + 1,) or not np.isfinite(standardized_parameters).all():
        raise ValueError(f"L-BFGS returned invalid parameters for L2={l2:g}")

    # Convert back to the raw cache feature coordinate system expected by the
    # evaluator: ((x - mean) / scale) @ w_s + b_s == x @ w + b.
    weight = standardized_parameters[:-1] / feature_scale
    bias = float(standardized_parameters[-1] - np.dot(feature_mean, weight))
    if not np.isfinite(weight).all() or not math.isfinite(bias):
        raise ValueError(f"raw current-only parameters are non-finite for L2={l2:g}")
    return np.asarray(weight, dtype=np.float64), bias, int(optimum.nit)


def fit_current_only_probe(cache: temporal_features.TemporalFeatureCache) -> ProbeFitResult:
    """Fits on train and selects L2 on val without reading the test split."""

    train_features, train_labels = _current_split(cache, "train")
    validation_features, validation_labels = _current_split(cache, "val")
    if train_features.shape[1] != validation_features.shape[1]:
        raise ValueError("current-only train/validation feature dimensions differ")
    if train_features.shape[1] != cache.metadata.feature_dim:
        raise ValueError("current-only features do not match sealed cache feature_dim")

    candidates: list[ProbeFitResult] = []
    for candidate_index, l2 in enumerate(L2_CANDIDATES):
        weight, bias, iterations = _fit_l2_candidate(
            train_features,
            train_labels,
            l2=l2,
            candidate_index=candidate_index,
        )
        validation_logits = validation_features @ weight + bias
        candidates.append(
            ProbeFitResult(
                weight=weight,
                bias=bias,
                selected_l2=l2,
                validation_log_loss=_binary_log_loss(validation_logits, validation_labels),
                optimizer_iterations=iterations,
            )
        )
    # Stronger regularization wins an exact validation-loss tie.  Test labels,
    # scores, and metrics are absent from this selection key and call graph.
    return min(candidates, key=lambda candidate: (candidate.validation_log_loss, -candidate.selected_l2))


def probe_metadata(cache: temporal_features.TemporalFeatureCache) -> dict[str, Any]:
    """Builds the exact schema-v1 binding accepted by the evaluator."""

    return {
        "schema_version": CURRENT_PROBE_SCHEMA_VERSION,
        "probe_type": "deterministic_linear_logit",
        "fit_split": "train",
        "feature_cache_schema_version": cache.metadata.schema_version,
        "feature_cache_model_config_name": cache.metadata.model_config_name,
        "feature_cache_checkpoint_path": cache.metadata.checkpoint_path,
        "feature_cache_row_count": cache.metadata.row_count,
        "feature_dim": cache.metadata.feature_dim,
    }


def save_probe_artifact(
    path: str | os.PathLike[str],
    *,
    cache: temporal_features.TemporalFeatureCache,
    fit: ProbeFitResult,
) -> Path:
    """Atomically writes exactly the evaluator's schema-v1 NPZ artifact."""

    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing current-only probe artifact: {output}")
    weight = np.asarray(fit.weight, dtype=np.float64)
    bias = np.asarray(fit.bias, dtype=np.float64)
    if weight.shape != (cache.metadata.feature_dim,) or bias.ndim != 0:
        raise ValueError(
            f"current-only artifact requires weight [{cache.metadata.feature_dim}] and scalar bias, "
            f"got {weight.shape}/{bias.shape}"
        )
    if not np.isfinite(weight).all() or not np.isfinite(bias):
        raise ValueError("refusing to save non-finite current-only probe parameters")
    metadata_json = json.dumps(probe_metadata(cache), sort_keys=True, separators=(",", ":"), allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as file:
            np.savez(file, metadata_json=np.asarray(metadata_json), weight=weight, bias=bias)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _load_sealed_manifest(path: Path) -> temporal_data.TemporalCompletionManifest:
    if not path.is_file():
        raise FileNotFoundError(f"required manifest not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object in {path}")
    return temporal_data.TemporalCompletionManifest.from_dict(value)


def load_fitting_cache(
    *,
    temporal_config: Any,
    manifest: temporal_data.TemporalCompletionManifest,
    feature_cache_path: Path,
) -> temporal_features.TemporalFeatureCache:
    """Loads the cache selected by the temporal training configuration."""

    checkpoint_path = temporal_config.completion.temporal_source_checkpoint_path
    return temporal_features.load_temporal_feature_cache(
        feature_cache_path,
        manifest=manifest,
        expected_checkpoint_path=checkpoint_path,
        expected_model_config_name=temporal_config.completion.temporal_source_model_config_name,
        sampling_protocol=temporal_config.completion.temporal_sampling_protocol,
    )


def run_fitting(args: argparse.Namespace) -> tuple[Path, ProbeFitResult]:
    # Importing the full config initializes model/JAX modules, so delay it until
    # after cheap output/path checks.
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing current-only probe artifact: {output}")
    from openpi.training import config as training_config  # noqa: PLC0415

    temporal_config = training_config.get_config(args.config_name)
    if not temporal_config.completion.uses_temporal_completion:
        raise ValueError(f"config {args.config_name!r} does not enable temporal completion")
    manifest_path = Path(args.manifest or temporal_config.completion.split_manifest_path).resolve()
    feature_cache_path = Path(args.feature_cache or temporal_config.completion.temporal_feature_cache_path).resolve()
    if output in {manifest_path, feature_cache_path}:
        raise ValueError("output artifact path must not overwrite a fitting input")

    manifest = _load_sealed_manifest(manifest_path)
    cache = load_fitting_cache(
        temporal_config=temporal_config,
        manifest=manifest,
        feature_cache_path=feature_cache_path,
    )
    fit = fit_current_only_probe(cache)
    return save_probe_artifact(output, cache=cache, fit=fit), fit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--manifest", type=Path, help="Defaults to completion.split_manifest_path from the config.")
    parser.add_argument(
        "--feature-cache",
        type=Path,
        help="Defaults to completion.temporal_feature_cache_path from the config.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New schema-v1 .npz path; existing files are refused."
    )
    return parser


def main() -> None:
    output, fit = run_fitting(_parser().parse_args())
    print(
        "Wrote deterministic current-only probe: "
        f"{output} (seed={FIT_SEED}, L2={fit.selected_l2:g}, "
        f"val_log_loss={fit.validation_log_loss:.9g}, iterations={fit.optimizer_iterations})"
    )


if __name__ == "__main__":
    main()
