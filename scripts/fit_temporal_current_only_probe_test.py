"""Protocol tests for fit_temporal_current_only_probe.py."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_temporal_completion as evaluator
from scripts import fit_temporal_current_only_probe as fitter


@dataclasses.dataclass(frozen=True)
class _Row:
    split: str
    label: int


@dataclasses.dataclass(frozen=True)
class _Metadata:
    schema_version: int = 3
    model_config_name: str = "clean_pi05"
    checkpoint_path: str = "/checkpoint/49999"
    row_count: int = 14
    feature_dim: int = 2


class _Cache:
    def __init__(self, *, test_labels: tuple[int, ...] = (0, 1)) -> None:
        train_features = np.asarray(
            [
                [-2.0, -0.5],
                [-1.6, -1.0],
                [-1.2, -0.3],
                [-0.8, -1.4],
                [0.8, 1.1],
                [1.2, 0.5],
                [1.6, 1.3],
                [2.0, 0.7],
            ],
            dtype=np.float32,
        )
        validation_features = np.asarray([[-1.8, -0.7], [-0.9, -1.0], [0.9, 0.8], [1.8, 1.0]], dtype=np.float32)
        test_features = np.asarray([[-100.0, 100.0], [100.0, -100.0]], dtype=np.float32)
        current = np.concatenate((train_features, validation_features, test_features), axis=0)
        self.prefix_history = np.zeros((len(current), 3, current.shape[1]), dtype=np.float32)
        self.prefix_history[:, -1, :] = current
        self.rows = (
            *(_Row("train", label) for label in (0, 0, 0, 0, 1, 1, 1, 1)),
            *(_Row("val", label) for label in (0, 0, 1, 1)),
            *(_Row("test", label) for label in test_labels),
        )
        self.metadata = _Metadata()
        self.requested_splits: list[str] = []

    def indices_for_split(self, split: str) -> np.ndarray:
        self.requested_splits.append(split)
        if split == "test":
            raise AssertionError("probe fitting must never request test indices")
        return np.asarray([index for index, row in enumerate(self.rows) if row.split == split], dtype=np.int64)


def test_fit_is_deterministic_and_test_labels_cannot_affect_weights() -> None:
    first_cache = _Cache(test_labels=(0, 1))
    changed_test_cache = _Cache(test_labels=(1, 0))

    first = fitter.fit_current_only_probe(first_cache)  # type: ignore[arg-type]
    repeated = fitter.fit_current_only_probe(first_cache)  # type: ignore[arg-type]
    changed_test = fitter.fit_current_only_probe(changed_test_cache)  # type: ignore[arg-type]

    assert fitter.FIT_SEED == 42
    assert fitter.L2_CANDIDATES == (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
    assert first_cache.requested_splits == ["train", "val", "train", "val"]
    assert changed_test_cache.requested_splits == ["train", "val"]
    assert first.selected_l2 in fitter.L2_CANDIDATES
    np.testing.assert_array_equal(first.weight, repeated.weight)
    np.testing.assert_array_equal(first.weight, changed_test.weight)
    assert first.bias == repeated.bias == changed_test.bias
    assert first.selected_l2 == repeated.selected_l2 == changed_test.selected_l2
    assert first.validation_log_loss == repeated.validation_log_loss == changed_test.validation_log_loss


def test_artifact_exactly_matches_evaluator_schema_and_binds_cache(tmp_path: Path) -> None:
    cache = _Cache()
    fit = fitter.fit_current_only_probe(cache)  # type: ignore[arg-type]
    output = fitter.save_probe_artifact(tmp_path / "current_probe.npz", cache=cache, fit=fit)  # type: ignore[arg-type]

    with np.load(output, allow_pickle=False) as arrays:
        assert set(arrays.files) == {"metadata_json", "weight", "bias"}
        metadata = json.loads(str(arrays["metadata_json"].item()))
        assert set(metadata) == {
            "schema_version",
            "probe_type",
            "fit_split",
            "feature_cache_schema_version",
            "feature_cache_model_config_name",
            "feature_cache_checkpoint_path",
            "feature_cache_row_count",
            "feature_dim",
        }
        assert metadata["schema_version"] == evaluator.CURRENT_PROBE_SCHEMA_VERSION
        assert arrays["weight"].dtype == np.float64
        assert arrays["bias"].dtype == np.float64
        assert np.isfinite(arrays["weight"]).all()
        assert np.isfinite(arrays["bias"])

    loaded = evaluator._load_current_only_probe(output, cache=cache)  # type: ignore[arg-type]  # noqa: SLF001
    np.testing.assert_array_equal(loaded.weight, fit.weight)
    assert loaded.bias == fit.bias

    changed_binding = _Cache()
    changed_binding.metadata = dataclasses.replace(changed_binding.metadata, checkpoint_path="/other/checkpoint")
    with pytest.raises(ValueError, match="feature_cache_checkpoint_path"):
        evaluator._load_current_only_probe(  # type: ignore[arg-type]  # noqa: SLF001
            output,
            cache=changed_binding,
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        fitter.save_probe_artifact(output, cache=cache, fit=fit)  # type: ignore[arg-type]


def test_clean_checkpoint_path_and_model_config_bindings_are_mandatory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "clean" / "49999"
    temporal_config = SimpleNamespace(
        completion=SimpleNamespace(
            temporal_source_checkpoint_path=str(checkpoint),
            temporal_source_model_config_name="clean_pi05",
        )
    )
    manifest = object()
    cache = object()
    calls: dict[str, object] = {}

    def load_temporal_feature_cache(path, **kwargs):
        calls["cache_path"] = Path(path)
        calls["cache_kwargs"] = kwargs
        return cache

    monkeypatch.setattr(fitter.temporal_features, "load_temporal_feature_cache", load_temporal_feature_cache)
    cache_path = tmp_path / "features.npz"

    assert (
        fitter.load_fitting_cache(
            temporal_config=temporal_config,
            manifest=manifest,  # type: ignore[arg-type]
            feature_cache_path=cache_path,
        )
        is cache
    )
    assert calls["cache_path"] == cache_path
    assert calls["cache_kwargs"] == {
        "manifest": manifest,
        "expected_checkpoint_path": str(checkpoint),
        "expected_model_config_name": "clean_pi05",
    }


def test_fit_rejects_single_class_train_or_validation_split() -> None:
    cache = _Cache()
    cache.rows = tuple(dataclasses.replace(row, label=0) if row.split == "val" else row for row in cache.rows)

    with pytest.raises(ValueError, match="val split must contain both"):
        fitter.fit_current_only_probe(cache)  # type: ignore[arg-type]
