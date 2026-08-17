"""Canonical identity for clean temporal-prefix preprocessing.

The extractor, trainer, and evaluator all call the same helper.  A cache is
therefore rejected when it was produced with a different model/data config,
prompt set, checkpoint assets, implementation, runtime, or pooling contract.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import enum
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np

import openpi.training.temporal_completion_data as _temporal_data
import openpi.training.temporal_completion_features as _temporal_features

_IMPLEMENTATION_FILES = (
    "pyproject.toml",
    "uv.lock",
    "packages/openpi-client/src/openpi_client/image_tools.py",
    "src/openpi/transforms.py",
    "src/openpi/policies/agilex_policy.py",
    "src/openpi/policies/policy.py",
    "src/openpi/policies/policy_config.py",
    "src/openpi/models/completion.py",
    "src/openpi/models/gemma.py",
    "src/openpi/models/lora.py",
    "src/openpi/models/model.py",
    "src/openpi/models/pi0.py",
    "src/openpi/models/pi0_config.py",
    "src/openpi/models/siglip.py",
    "src/openpi/models/tokenizer.py",
    "src/openpi/shared/image_tools.py",
    "src/openpi/shared/normalize.py",
    "src/openpi/training/config.py",
    "src/openpi/training/temporal_completion_preprocess.py",
    "scripts/extract_temporal_completion_features.py",
)


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_runtime_value(value: Any) -> Any:
    """Converts preprocessing state into strict JSON-fingerprintable data."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return canonical_runtime_value(value.value)
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "type": "numpy.ndarray",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "sha256": _sha256_bytes(contiguous.tobytes(order="C")),
        }
    if isinstance(value, Mapping):
        return {
            str(key): canonical_runtime_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_runtime_value(item) for item in value]
    if dataclasses.is_dataclass(value):
        return {
            "type": _qualified_type(value),
            "fields": {
                field.name: canonical_runtime_value(getattr(value, field.name)) for field in dataclasses.fields(value)
            },
        }

    sentencepiece = getattr(value, "_tokenizer", None)
    serialize = getattr(sentencepiece, "serialized_model_proto", None)
    if callable(serialize) and hasattr(value, "_max_len"):
        model_bytes = bytes(serialize())
        return {
            "type": _qualified_type(value),
            "max_len": int(vars(value)["_max_len"]),
            "sentencepiece_sha256": _sha256_bytes(model_bytes),
        }
    raise TypeError(f"cannot fingerprint preprocessing value of type {_qualified_type(value)}")


def implementation_fingerprint(repo_root: str | os.PathLike[str] | None = None) -> str:
    """Fingerprints every source file that can change an extracted prefix."""

    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[3]
    records: list[dict[str, str]] = []
    for relative in _IMPLEMENTATION_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"preprocessing implementation file not found: {path}")
        records.append({"path": relative, "sha256": _sha256_bytes(path.read_bytes())})
    return _temporal_data.stable_fingerprint(records)


def runtime_versions() -> dict[str, str]:
    """Returns stable versions for libraries involved in prefix inference."""

    versions = {"python": platform.python_version()}
    for distribution in (
        "av",
        "einops",
        "flax",
        "jax",
        "jaxlib",
        "lerobot",
        "numpy",
        "pillow",
        "sentencepiece",
        "torch",
        "torchcodec",
        "torchvision",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def make_preprocess_fingerprint(
    *,
    model_config: Any,
    data_config: Any,
    prompts: Mapping[int, str],
    checkpoint_assets_fingerprint: str,
    code_fingerprint: str,
) -> str:
    """Seals all runtime inputs that can change clean prefix preprocessing."""

    payload = {
        "protocol_version": _temporal_features.PREPROCESS_PROTOCOL_VERSION,
        "pooling_method": _temporal_features.POOLING_METHOD,
        "model_config": canonical_runtime_value(model_config),
        "data_config": canonical_runtime_value(data_config),
        "logical_prompts": canonical_runtime_value(dict(prompts)),
        "checkpoint_assets_fingerprint": checkpoint_assets_fingerprint,
        "implementation_fingerprint": code_fingerprint,
        "runtime_versions": runtime_versions(),
        "eval": True,
        "augmentation": False,
        "prompt_override_before_transforms": True,
    }
    return _temporal_data.stable_fingerprint(payload)


def expected_preprocess_fingerprint(
    *,
    source_train_config: Any,
    manifest: _temporal_data.TemporalCompletionManifest,
    checkpoint_path: str | os.PathLike[str],
    repo_root: str | os.PathLike[str] | None = None,
) -> str:
    """Rebuilds the unique expected cache preprocessing identity."""

    data_config = source_train_config.data.create(source_train_config.assets_dirs, source_train_config.model)
    prompts = dict(enumerate(manifest.task_prompts))
    checkpoint = Path(checkpoint_path).resolve()
    return make_preprocess_fingerprint(
        model_config=source_train_config.model,
        data_config=data_config,
        prompts=prompts,
        checkpoint_assets_fingerprint=_temporal_features.directory_fingerprint(checkpoint / "assets"),
        code_fingerprint=implementation_fingerprint(repo_root),
    )


def describe_preprocess_contract() -> str:
    """Returns a compact, stable human-readable contract for artifact logs."""

    return json.dumps(
        {
            "protocol_version": _temporal_features.PREPROCESS_PROTOCOL_VERSION,
            "pooling_method": _temporal_features.POOLING_METHOD,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
