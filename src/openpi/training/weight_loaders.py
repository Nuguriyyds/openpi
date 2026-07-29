import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import jax
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

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

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class FurnitureVLAWeightLoader(WeightLoader):
    """Loads pi0.5 base weights into the 15-dimensional FurnitureVLA model."""

    params_path: str
    seed: int = 0

    source_action_dim: int = 32
    robot_action_dim: int = 14

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path),
            restore_type=np.ndarray,
        )

        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        missing_keys = set(flat_ref) - set(flat_loaded)
        extra_keys = set(flat_loaded) - set(flat_ref)

        if missing_keys:
            raise ValueError(
                f"Checkpoint is missing parameters: {sorted(missing_keys)}"
            )

        if extra_keys:
            raise ValueError(
                f"Checkpoint contains unexpected parameters: {sorted(extra_keys)}"
            )

        input_kernel_key = "action_in_proj/kernel"
        output_kernel_key = "action_out_proj/kernel"
        output_bias_key = "action_out_proj/bias"

        resized_keys = {
            input_kernel_key,
            output_kernel_key,
            output_bias_key,
        }

        target_action_dim = self.robot_action_dim + 1

        rng = jax.random.PRNGKey(self.seed)
        input_rng, output_rng = jax.random.split(rng)

        kernel_init = jax.nn.initializers.lecun_normal()

        result = {}

        for key, target_spec in flat_ref.items():
            source_value = flat_loaded[key]

            # Parameters whose shapes did not change are loaded normally.
            if source_value.shape == target_spec.shape:
                result[key] = source_value.astype(target_spec.dtype)
                continue

            # Only the three action projection parameters may change shape.
            if key not in resized_keys:
                raise ValueError(
                    f"Unexpected shape mismatch for {key}: "
                    f"checkpoint={source_value.shape}, "
                    f"target={target_spec.shape}"
                )

            if key == input_kernel_key:
                expected_source_shape = (
                    self.source_action_dim,
                    target_spec.shape[1],
                )
                expected_target_shape = (
                    target_action_dim,
                    target_spec.shape[1],
                )

                if source_value.shape != expected_source_shape:
                    raise ValueError(
                        f"Unexpected checkpoint shape for {key}: "
                        f"expected={expected_source_shape}, "
                        f"got={source_value.shape}"
                    )

                if target_spec.shape != expected_target_shape:
                    raise ValueError(
                        f"Unexpected target shape for {key}: "
                        f"expected={expected_target_shape}, "
                        f"got={target_spec.shape}"
                    )

                adapted_value = np.array(
                    kernel_init(
                        input_rng,
                        target_spec.shape,
                        target_spec.dtype,
                    )
                )

                adapted_value[: self.robot_action_dim, :] = source_value[
                    : self.robot_action_dim, :
                ]

            elif key == output_kernel_key:
                expected_source_shape = (
                    target_spec.shape[0],
                    self.source_action_dim,
                )
                expected_target_shape = (
                    target_spec.shape[0],
                    target_action_dim,
                )

                if source_value.shape != expected_source_shape:
                    raise ValueError(
                        f"Unexpected checkpoint shape for {key}: "
                        f"expected={expected_source_shape}, "
                        f"got={source_value.shape}"
                    )

                if target_spec.shape != expected_target_shape:
                    raise ValueError(
                        f"Unexpected target shape for {key}: "
                        f"expected={expected_target_shape}, "
                        f"got={target_spec.shape}"
                    )

                adapted_value = np.array(
                    kernel_init(
                        output_rng,
                        target_spec.shape,
                        target_spec.dtype,
                    )
                )

                adapted_value[:, : self.robot_action_dim] = source_value[
                    :, : self.robot_action_dim
                ]

            else:
                expected_source_shape = (self.source_action_dim,)
                expected_target_shape = (target_action_dim,)

                if source_value.shape != expected_source_shape:
                    raise ValueError(
                        f"Unexpected checkpoint shape for {key}: "
                        f"expected={expected_source_shape}, "
                        f"got={source_value.shape}"
                    )

                if target_spec.shape != expected_target_shape:
                    raise ValueError(
                        f"Unexpected target shape for {key}: "
                        f"expected={expected_target_shape}, "
                        f"got={target_spec.shape}"
                    )

                adapted_value = np.zeros(
                    target_spec.shape,
                    dtype=target_spec.dtype,
                )

                adapted_value[: self.robot_action_dim] = source_value[
                    : self.robot_action_dim
                ]

            adapted_value = adapted_value.astype(target_spec.dtype)
            result[key] = adapted_value

            logger.info(
                "Adapted FurnitureVLA parameter %s from %s to %s",
                key,
                source_value.shape,
                adapted_value.shape,
            )

        return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

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
