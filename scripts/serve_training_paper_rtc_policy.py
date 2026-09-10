"""Serve the AgileX 10470 training-paper RTC checkpoint for paper_rtc control.

This entry point keeps the WebSocket/chunk protocol from
``serve_training_paper_rtc_base.py`` but uses the model's native
``sample_actions(action_prefix=..., delay=...)`` path. That matches the
training-paper RTC paper more closely than the older compatibility sampler:

* only the first ``d`` server-encoded prefix actions are hard-conditioned;
* prefix tokens are kept clean at every denoising step;
* no PiGDM/Jacobian guidance is used at inference time.
"""

from __future__ import annotations

from typing import Any

import einops
import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import serve_training_paper_rtc_base as _base
from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models.pi0 import Pi0


DEFAULT_CHECKPOINT_DIR = "/home/geekplus/develop/openpi/checkpoints/rtc_10470_50k/30000"


def _posemb_sincos_any_shape(
    pos: jax.Array,
    embedding_dim: int,
    *,
    min_period: float,
    max_period: float,
) -> jax.Array:
    """Sine-cosine timestep embedding that accepts token-wise timesteps.

    The stock Pi0 helper is jaxtyping-annotated for a batch-level ``(B,)``
    timestep. Training-paper RTC needs ``(B, H)`` timesteps for action tokens, so
    this local copy keeps the same math without changing existing OpenPI files.
    """

    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")
    pos_shape = pos.shape
    flat_pos = jnp.reshape(pos, (-1,))
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        flat_pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    emb = jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)
    return jnp.reshape(emb, (*pos_shape, embedding_dim))


def _install_runtime_model_patches() -> None:
    """Patch only this server process; no existing source files are modified."""

    if getattr(Pi0.sample_actions, "_training_paper_rtc_runtime_patch", False):
        return

    def _runtime_embed_suffix(self, obs, noisy_actions, timestep):
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = _posemb_sincos_any_shape(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        )
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            if time_emb.ndim == 2:
                time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            else:
                time_tokens = time_emb
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _runtime_sample_actions_impl(
        self,
        rng,
        observation,
        *,
        num_steps=10,
        noise=None,
        action_prefix=None,
        delay=None,
    ):
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        action_prefix_mask = None
        if action_prefix is not None:
            if delay is None:
                raise ValueError("delay must be provided when action_prefix is provided")
            action_prefix = jnp.asarray(action_prefix, dtype=noise.dtype)
            if action_prefix.ndim == 2:
                action_prefix = action_prefix[None, ...]
            action_prefix = jnp.broadcast_to(action_prefix, noise.shape)
            delay = jnp.asarray(delay)
            if delay.ndim == 0:
                delay = jnp.broadcast_to(delay, (batch_size,))
            delay = jnp.clip(delay, 0, self.action_horizon)
            action_prefix_mask = jnp.arange(self.action_horizon)[None, :] < delay[:, None]

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (raw_prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )

        def step(carry):
            x_t, time = carry
            time_for_actions = jnp.broadcast_to(time, (batch_size,))
            if action_prefix is not None:
                # Training-paper RTC keeps known prefix tokens clean. OpenPI
                # samples from t=1 noise to t=0 action, so clean prefix tokens
                # use timestep 0.0.
                x_t = jnp.where(action_prefix_mask[:, :, None], action_prefix, x_t)
                time_for_actions = jnp.where(action_prefix_mask, 0.0, time)

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                time_for_actions,
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_step = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask_step, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            x_next = x_t + dt * v_t
            if action_prefix is not None:
                x_next = jnp.where(action_prefix_mask[:, :, None], action_prefix, x_next)
            return x_next, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        if action_prefix is not None:
            x_0 = jnp.where(action_prefix_mask[:, :, None], action_prefix, x_0)
        return x_0, jax.lax.stop_gradient(raw_prefix_out), prefix_mask

    def _runtime_sample_actions(
        self,
        rng,
        observation,
        *,
        num_steps=10,
        noise=None,
        action_prefix=None,
        delay=None,
    ):
        actions, _, _ = _runtime_sample_actions_impl(
            self,
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
            action_prefix=action_prefix,
            delay=delay,
        )
        return actions

    def _runtime_sample_actions_with_raw_prefix(
        self,
        rng,
        observation,
        *,
        num_steps=10,
        noise=None,
        action_prefix=None,
        delay=None,
    ):
        return _runtime_sample_actions_impl(
            self,
            rng,
            observation,
            num_steps=num_steps,
            noise=noise,
            action_prefix=action_prefix,
            delay=delay,
        )

    class RuntimeRMSNorm(_gemma.RMSNorm):
        @nn.compact
        def __call__(self, x, cond):
            dtype = x.dtype
            var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
            normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
            if cond is None:
                scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
                normed_inputs = normed_inputs * (1 + scale)
                return normed_inputs.astype(dtype), None

            modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
            if modulation.ndim == x.ndim - 1:
                modulation = modulation[:, None, :]
            elif modulation.ndim != x.ndim:
                raise ValueError(f"adarms_cond rank {modulation.ndim} is incompatible with token rank {x.ndim}")
            scale, shift, gate = jnp.split(modulation, 3, axis=-1)
            normed_inputs = normed_inputs * (1 + scale) + shift
            return normed_inputs.astype(dtype), gate

    _runtime_sample_actions._training_paper_rtc_runtime_patch = True
    _runtime_sample_actions_with_raw_prefix._training_paper_rtc_runtime_patch = True
    RuntimeRMSNorm._training_paper_rtc_runtime_patch = True
    Pi0.embed_suffix = _runtime_embed_suffix
    Pi0.sample_actions = _runtime_sample_actions
    Pi0.sample_actions_with_raw_prefix = _runtime_sample_actions_with_raw_prefix
    _gemma.RMSNorm = RuntimeRMSNorm


class TrainingPaperRtcSampler:
    """Native training-paper RTC sampler using Pi0 action_prefix/delay support."""

    def __init__(
        self,
        policy: _base._policy.Policy,
        *,
        num_steps: int,
    ) -> None:
        if not isinstance(policy._model, Pi0):  # noqa: SLF001
            raise ValueError(
                "training_paper_rtc requires a standard JAX Pi0/Pi0.5 policy. "
                f"Loaded model type: {type(policy._model).__name__}."
            )
        self._policy = policy
        self._model = policy._model  # noqa: SLF001
        self._action_horizon = self._model.action_horizon
        self._action_dim = self._model.action_dim
        self._num_steps = int(num_steps)

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _prepare_prefix(
        self,
        action_prefix_model: np.ndarray | None,
        prefix_steps: int,
    ) -> tuple[np.ndarray, int, int]:
        # Keep the JIT call signature stable: the first chunk uses a zero
        # prefix with delay=0 instead of omitting action_prefix entirely.
        prefix = np.zeros((self._action_horizon, self._action_dim), dtype=np.float32)
        if action_prefix_model is None:
            return prefix, 0, 0

        arr = np.asarray(action_prefix_model, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected model action prefix rank 2, got {arr.shape}")

        prefix_len = min(arr.shape[0], self._action_horizon)
        copy_dim = min(arr.shape[1], self._action_dim)
        valid_prefix = min(max(0, int(prefix_steps)), prefix_len)
        if valid_prefix > 0 and copy_dim > 0:
            prefix[:valid_prefix, :copy_dim] = arr[:valid_prefix, :copy_dim]
        return prefix, prefix_len, valid_prefix

    def model_prefix_from_robot_actions(self, obs: dict, robot_actions: np.ndarray) -> np.ndarray:
        """Convert absolute robot-space leftover actions into current model space.

        RA checkpoints train joint actions as chunk-wise deltas relative to the
        current observation state. Reusing the previous chunk's model-space
        deltas directly would keep them relative to an old state. Passing the
        absolute leftover robot targets back through the policy input transform
        recomputes the correct current-state deltas and normalization.
        """

        arr = np.asarray(robot_actions, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected robot leftover action chunk rank 2, got {arr.shape}")
        obs_with_actions = dict(obs)
        obs_with_actions["actions"] = arr.copy()
        transformed = self._policy._input_transform(obs_with_actions)  # noqa: SLF001
        actions = np.asarray(transformed.get("actions"), dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Expected transformed model prefix rank 2, got {actions.shape}")
        if actions.shape[-1] != self._action_dim:
            raise ValueError(
                f"Expected transformed model prefix dim {self._action_dim}, got {actions.shape}"
            )
        return actions

    def infer_chunk(
        self,
        obs: dict,
        *,
        action_prefix_model: np.ndarray | None,
        prefix_steps: int,
    ) -> dict[str, np.ndarray | int]:
        action_prefix, prefix_len, valid_prefix = self._prepare_prefix(
            action_prefix_model,
            prefix_steps,
        )
        outputs = self._policy.infer(
            obs,
            action_prefix=action_prefix,
            delay=np.asarray(valid_prefix, dtype=np.int32),
            num_steps=self._num_steps,
        )
        robot_actions = np.asarray(outputs["actions"], dtype=np.float32)

        return {
            "actions": robot_actions,
            "prefix_input_len": np.asarray(prefix_len, dtype=np.int32),
            "conditioned_prefix_steps": np.asarray(valid_prefix, dtype=np.int32),
        }

    def infer_chunk_with_raw_prefix(
        self,
        obs: dict,
        *,
        action_prefix_model: np.ndarray | None,
        prefix_steps: int,
    ) -> dict[str, Any]:
        """Generate an action chunk and expose its same-forward raw prefix."""

        action_prefix, prefix_len, valid_prefix = self._prepare_prefix(
            action_prefix_model,
            prefix_steps,
        )
        outputs = self._policy.infer_with_raw_prefix(
            obs,
            action_prefix=action_prefix,
            delay=np.asarray(valid_prefix, dtype=np.int32),
            num_steps=self._num_steps,
        )
        robot_actions = np.asarray(outputs["actions"], dtype=np.float32)
        return {
            "actions": robot_actions,
            "prefix_input_len": np.asarray(prefix_len, dtype=np.int32),
            "conditioned_prefix_steps": np.asarray(valid_prefix, dtype=np.int32),
            "raw_prefix_out": outputs["raw_prefix_out"],
            "raw_prefix_mask": outputs["raw_prefix_mask"],
        }


def _install_training_paper_rtc_overrides() -> None:
    _install_runtime_model_patches()
    _base.TrainingPaperRtcSampler = TrainingPaperRtcSampler
    _base.AGILEX_10470_CHECKPOINT = DEFAULT_CHECKPOINT_DIR
    _base.DEFAULT_CHECKPOINT[_base.EnvMode.AGILEX] = _base.Checkpoint(
        config=_base.AGILEX_10470_CONFIG_NAME,
        dir=DEFAULT_CHECKPOINT_DIR,
    )


def main(args: _base.Args) -> None:
    _install_training_paper_rtc_overrides()
    jax.config.update("jax_default_matmul_precision", "float32")
    _base.main(args)


if __name__ == "__main__":
    _install_training_paper_rtc_overrides()
    _base.logging.basicConfig(level=_base.logging.INFO, force=True)
    main(_base.tyro.cli(_base.Args))
