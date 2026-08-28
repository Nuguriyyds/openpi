from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            # Keep JAX-only policy import/testing independent of platform-
            # specific torch shared libraries.
            import torch

            self._torch = torch
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            sample_with_prefix = getattr(model, "sample_actions_with_prefix_feature", None)
            temporal_logits = getattr(model, "compute_temporal_completion_logits", None)
            self._sample_actions_with_prefix_feature = (
                None if sample_with_prefix is None else nnx_utils.module_jit(sample_with_prefix)
            )
            self._temporal_completion_logits = (
                None if temporal_logits is None else nnx_utils.module_jit(temporal_logits)
            )
            self._rng = rng if rng is not None else jax.random.key(0)
            # Completion inference is deterministic (dropout is disabled).  A
            # separate fixed key keeps scoring from advancing the action-noise
            # stream and changing subsequent actions.
            self._temporal_completion_rng = jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        action_prefix: np.ndarray | None = None,
        delay: int | np.ndarray | None = None,
        num_steps: int | None = None,
        return_model_actions: bool = False,
        return_prefix_feature: bool = False,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        obs = dict(obs)
        action_prefix = action_prefix if action_prefix is not None else obs.pop("action_prefix", None)
        action_prefix = action_prefix if action_prefix is not None else obs.pop("action_prefix_model", None)
        delay = delay if delay is not None else obs.pop("delay", None)
        delay = delay if delay is not None else obs.pop("delay_steps", None)
        num_steps = num_steps if num_steps is not None else obs.pop("num_steps", None)
        num_steps = num_steps if num_steps is not None else obs.pop("num_denoising_steps", None)
        return_model_actions = bool(obs.pop("return_model_actions", return_model_actions))
        return_prefix_feature = bool(obs.pop("return_prefix_feature", return_prefix_feature))
        if return_prefix_feature and self._is_pytorch_model:
            raise ValueError("return_prefix_feature is only implemented for JAX Pi0/Pi0.5 policies")
        if return_prefix_feature and self._sample_actions_with_prefix_feature is None:
            raise ValueError("the loaded JAX model cannot return a reusable prefix feature")

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(
                lambda x: self._torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if num_steps is not None:
            sample_kwargs["num_steps"] = num_steps
        if noise is not None:
            noise = (
                self._torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            )

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise
        if action_prefix is not None:
            if self._is_pytorch_model:
                raise ValueError("training-time RTC action_prefix sampling is only implemented for JAX models")
            action_prefix = jnp.asarray(action_prefix)
            if action_prefix.ndim == 2:
                action_prefix = action_prefix[None, ...]
            sample_kwargs["action_prefix"] = action_prefix
            sample_kwargs["delay"] = 0 if delay is None else delay

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        prefix_feature = None
        if return_prefix_feature:
            assert self._sample_actions_with_prefix_feature is not None
            actions_model, prefix_feature = self._sample_actions_with_prefix_feature(
                sample_rng_or_pytorch_device, observation, **sample_kwargs
            )
        else:
            actions_model = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {
            "state": inputs["state"],
            "actions": actions_model,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        actions_model_np = np.asarray(outputs["actions"]).copy()
        outputs = self._output_transform(outputs)
        if return_model_actions:
            outputs["actions_model"] = actions_model_np
        if prefix_feature is not None:
            outputs["prefix_feature"] = np.asarray(prefix_feature[0, ...], dtype=np.float32)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def score_temporal_completion(self, prefix_history: np.ndarray, *, return_logit: bool = False) -> float:
        """Scores one oldest-to-newest three-prefix history with the loaded head."""

        if self._is_pytorch_model:
            raise ValueError("temporal completion scoring is only implemented for JAX Pi0/Pi0.5 policies")
        if self._temporal_completion_logits is None:
            raise ValueError("the loaded JAX model does not provide a temporal completion head")
        history = np.asarray(prefix_history, dtype=np.float32)
        token_mode = getattr(self._model, "completion_head_variant", None) == "token_query_attention"
        if token_mode:
            input_dim = int(self._model.prefix_feature_dim)
            if history.ndim != 3 or history.shape[0] != 3 or history.shape[-1] != input_dim + 1:
                raise ValueError(f"token prefix_history must have shape [3, N, {input_dim + 1}], got {history.shape}")
            tokens = history[..., :input_dim]
            masks = history[..., input_dim] > 0.5
            logits = self._temporal_completion_logits(
                self._temporal_completion_rng,
                jnp.asarray(tokens)[None, ...],
                jnp.asarray(masks)[None, ...],
            )
        else:
            if history.ndim != 2 or history.shape[0] != 3:
                raise ValueError(f"prefix_history must have shape [3, D], got {history.shape}")
            # Do not pass ``train=False`` through module_jit: a non-static Python
            # boolean would become a tracer inside the head's dropout branch.  The
            # method default is already the required deterministic eval mode.
            logits = self._temporal_completion_logits(
                self._temporal_completion_rng,
                jnp.asarray(history)[None, ...],
            )
        logit = float(np.asarray(logits)[0])
        if return_logit:
            return logit
        return float(jax.nn.sigmoid(jnp.asarray(logit, dtype=jnp.float32)))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
