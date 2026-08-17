from __future__ import annotations

import dataclasses
import logging
import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

if TYPE_CHECKING:
    import openpi.shared.array_typing as at

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    # Missing keys are initialized from the reference model only when they
    # fully match this expression. The default preserves historical LoRA-only
    # behavior; completion configs explicitly add ``completion_head/.*``.
    missing_regex: str = ".*lora.*"
    # Frozen-head configs can opt into a fail-closed key audit. Legacy
    # checkpoint loaders keep the historical permissive subset behavior.
    reject_unexpected: bool = False

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        import openpi.models.model as _model  # noqa: PLC0415
        import openpi.shared.download as download  # noqa: PLC0415

        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(
            loaded_params,
            params,
            missing_regex=self.missing_regex,
            reject_unexpected=self.reject_unexpected,
        )


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        import openpi.shared.download as download  # noqa: PLC0415

        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(
    loaded_params: at.Params,
    params: at.Params,
    *,
    missing_regex: str,
    reject_unexpected: bool = False,
) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.
        reject_unexpected: Whether to reject checkpoint keys absent from the reference model.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    if reject_unexpected:
        unexpected = sorted(set(flat_loaded) - set(flat_ref))
        if unexpected:
            preview = ", ".join(unexpected[:10])
            suffix = "" if len(unexpected) <= 10 else f", ... ({len(unexpected)} total)"
            raise ValueError(f"checkpoint contains unexpected parameter keys: {preview}{suffix}")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
