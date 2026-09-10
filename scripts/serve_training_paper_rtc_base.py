"""Serve a Pi0/Pi0.5 policy with training-paper RTC action conditioning.

This is a new entrypoint. It intentionally keeps the existing
``serve_paper_rtc_policy.py`` and robot control scripts untouched.

The transport and chunk-execution contract follow the existing
``paper_rtc_online_inference_execution.py`` client:

* the client executes chunks locally at the robot control rate;
* a background client planner sends the unconsumed robot-space absolute-action
  tail of the previous chunk back to the server;
* the server re-encodes that robot-space prefix with the current observation
  before applying training-time RTC action conditioning;
* one WebSocket response returns only a full robot-space chunk.

The sampler differs from paper/PiGDM inference-time RTC. It implements the
training-time action-conditioning idea from Algorithm 1 using the current
OpenPI Pi0/Pi0.5 flow direction. OpenPI denoises from ``t=1`` noise to ``t=0``
actions, so each ODE step injects the known delayed action prefix into ``x_t``
and denoises only the remaining suffix. No PiGDM VJP guidance is used.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import enum
import http
import logging
import math
import os
import pathlib
import socket
import threading
import time
import traceback
from typing import Any

import cv2
import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import msgpack_numpy
import tyro
import websockets
import websockets.asyncio.server as _ws_server
import websockets.frames

from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


logger = logging.getLogger(__name__)

_RA_IMAGE_TRANSPORT_KEY = "_ra_image_transport"
_RA_IMAGE_TRANSPORT_VERSION = 1
_RA_IMAGE_SIZE = 224
_RA_CAMERAS = ("cam_top", "cam_left_wrist", "cam_right_wrist")
_RA_IMAGE_TRANSPORT_METADATA = {
    "version": _RA_IMAGE_TRANSPORT_VERSION,
    "encoding": "jpeg",
    "image_size": _RA_IMAGE_SIZE,
    "cameras": list(_RA_CAMERAS),
}


def _decode_ra_jpeg_observation(obs: dict[str, Any]) -> dict[str, Any]:
    """Validate the RA wire format and restore JPEG images as CHW RGB arrays."""
    transport_version = obs.pop(_RA_IMAGE_TRANSPORT_KEY, None)
    if transport_version != _RA_IMAGE_TRANSPORT_VERSION:
        raise ValueError(
            "RA JPEG observation transport mismatch: "
            f"expected version {_RA_IMAGE_TRANSPORT_VERSION}, got {transport_version!r}"
        )

    images = obs.get("images")
    if not isinstance(images, dict) or set(images) != set(_RA_CAMERAS):
        received = sorted(images) if isinstance(images, dict) else type(images).__name__
        raise ValueError(
            f"RA JPEG observation must contain exactly {_RA_CAMERAS}, got {received!r}"
        )

    decoded_images: dict[str, np.ndarray] = {}
    for camera in _RA_CAMERAS:
        payload = images[camera]
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"RA JPEG payload for {camera} must be bytes, got {type(payload).__name__}"
            )
        bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"failed to decode RA JPEG payload for {camera}")
        if bgr.shape != (_RA_IMAGE_SIZE, _RA_IMAGE_SIZE, 3):
            raise ValueError(
                f"decoded RA image for {camera} must be "
                f"{_RA_IMAGE_SIZE}x{_RA_IMAGE_SIZE} RGB, got {bgr.shape}"
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        decoded_images[camera] = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))

    obs["images"] = decoded_images
    return obs

AGILEX_10470_CONFIG_NAME = "ttrtc"
AGILEX_10470_CHECKPOINT = "/home/geekplus/develop/openpi/checkpoints/rtc_10470_50k/30000"


class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    AGILEX = "agilex"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    """Arguments for the training-paper RTC policy server."""

    env: EnvMode = EnvMode.AGILEX
    default_prompt: str | None = None
    port: int = 8000

    # Core chunk parameters. They are printed at startup and exported through
    # server metadata so a client log clearly shows which RTC setting is active.
    action_horizon: int = 50
    control_hz: float = 20.0
    num_denoising_steps: int = 5
    # Minimum number of actions the existing client executes from the current
    # chunk before requesting the next one.
    execution_horizon: int = 10
    # Metadata only: this is the simulated delay used by the current fine-tuned
    # training-paper RTC checkpoint.
    trained_simulated_delay: int = 5
    # Optional hard cap on the conditioned prefix. If unset, the client-provided
    # fixed delay is used directly.
    max_conditioned_prefix_steps: int | None = None

    # Delay estimation. ``max`` is the most conservative and closest to the
    # real robot safety posture; p95 is useful if one-off spikes are too costly.
    delay_buffer_size: int = 20
    delay_percentile: float = 1.0
    initial_delay_s: float = 0.10
    min_delay_steps: int = 1
    max_delay_steps: int | None = None
    # Legacy compatibility knob. Replanning now starts when the executed
    # in-chunk index reaches ``execution_horizon`` exactly.
    replan_margin_steps: int = 1

    # Optional final output smoothing for real hardware. 1.0 disables it.
    filter_alpha: float = 1.0
    hold_last_on_underrun: bool = True

    # Saves chunk records under ``policy_records/training_paper_rtc`` when enabled.
    record: bool = False

    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.AGILEX: Checkpoint(
        config=AGILEX_10470_CONFIG_NAME,
        dir=AGILEX_10470_CHECKPOINT,
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config),
                args.policy.dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            if checkpoint := DEFAULT_CHECKPOINT.get(args.env):
                return _policy_config.create_trained_policy(
                    _config.get_config(checkpoint.config),
                    checkpoint.dir,
                    default_prompt=args.default_prompt,
                )
            raise ValueError(f"Unsupported environment mode: {args.env}")


def execution_action_space(args: Args) -> str:
    """Derive the robot-side action space from the loaded training config."""
    if isinstance(args.policy, Checkpoint):
        config_name = args.policy.config
    elif checkpoint := DEFAULT_CHECKPOINT.get(args.env):
        config_name = checkpoint.config
    else:
        raise ValueError(f"Unsupported environment mode: {args.env}")

    data = _config.get_config(config_name).data
    if bool(getattr(data, "absolute_end_effector_actions", False)) or bool(
        getattr(data, "relative_end_effector_actions", False)
    ):
        # Relative EE checkpoints are converted back to absolute EE targets by
        # AbsoluteEndEffectorActions in the policy output transform.
        return "absolute_end_effector"
    return "absolute_joint"


def _validate_args(args: Args) -> None:
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive")
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be positive")
    if args.num_denoising_steps <= 0:
        raise ValueError("--num-denoising-steps must be positive")
    if args.execution_horizon <= 0:
        raise ValueError("--execution-horizon must be positive")
    if args.trained_simulated_delay < 0:
        raise ValueError("--trained-simulated-delay must be non-negative")
    if args.max_conditioned_prefix_steps is not None and args.max_conditioned_prefix_steps < 0:
        raise ValueError("--max-conditioned-prefix-steps must be non-negative")
    if args.delay_buffer_size <= 0:
        raise ValueError("--delay-buffer-size must be positive")
    if not 0.0 <= args.delay_percentile <= 1.0:
        raise ValueError("--delay-percentile must be in [0, 1]")
    if not 0.0 < args.filter_alpha <= 1.0:
        raise ValueError("--filter-alpha must be in (0, 1]")


def _build_training_paper_rtc_sampler(model: Pi0, *, num_steps: int):
    graphdef, state = nnx.split(model)
    action_horizon = model.action_horizon
    action_dim = model.action_dim

    def core(
        frozen_state: nnx.State,
        rng: jax.Array,
        observation: _model.Observation,
        prev_chunk: jax.Array,
        prev_valid_mask: jax.Array,
        prefix_steps: jax.Array,
    ) -> jax.Array:
        pi0 = nnx.merge(graphdef, frozen_state)
        obs = _model.preprocess_observation(None, observation, train=False)
        batch_size = obs.state.shape[0]

        noise = jax.random.normal(
            rng,
            (batch_size, action_horizon, action_dim),
            dtype=jnp.float32,
        )

        prefix_tokens, prefix_mask, prefix_ar_mask = pi0.embed_prefix(obs)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = pi0.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )

        valid_len = jnp.sum(prev_valid_mask.astype(jnp.int32))
        prefix_len = jnp.minimum(jnp.asarray(prefix_steps, jnp.int32), valid_len)
        prefix_len = jnp.minimum(prefix_len, jnp.asarray(action_horizon, jnp.int32))
        action_prefix_mask = jnp.arange(action_horizon) < prefix_len
        action_prefix_mask = jnp.logical_and(action_prefix_mask, prev_valid_mask)

        def inject_action_prefix(x_t: jax.Array) -> jax.Array:
            # The current training run used a Pi0-compatible approximation of
            # Algorithm 1: clean known actions are injected into x_t and their
            # loss is masked, while Pi0 keeps a batch-level timestep.
            return jnp.where(action_prefix_mask[None, :, None], prev_chunk, x_t)

        def velocity(x_t: jax.Array, time_scalar: jax.Array) -> jax.Array:
            time_batch = jnp.broadcast_to(time_scalar, (batch_size,))
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = pi0.embed_suffix(
                obs,
                x_t,
                time_batch,
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_step = einops.repeat(
                prefix_mask,
                "b p -> b s p",
                s=suffix_tokens.shape[1],
            )
            full_attn_mask = jnp.concatenate(
                [prefix_attn_mask_step, suffix_attn_mask],
                axis=-1,
            )
            positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (_, suffix_out), _ = pi0.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return pi0.action_out_proj(suffix_out[:, -action_horizon:]).astype(x_t.dtype)

        def step(carry: tuple[jax.Array, jax.Array], _) -> tuple[tuple[jax.Array, jax.Array], None]:
            x_t, time_scalar = carry
            conditioned_x_t = inject_action_prefix(x_t)
            v_t = velocity(conditioned_x_t, time_scalar)
            dt = jnp.asarray(-1.0 / num_steps, dtype=jnp.float32)
            return (conditioned_x_t + dt * v_t, time_scalar + dt), None

        (actions, _), _ = jax.lax.scan(
            step,
            (noise, jnp.asarray(1.0, dtype=jnp.float32)),
            xs=None,
            length=num_steps,
        )
        return inject_action_prefix(actions)

    jitted_core = jax.jit(core)

    def sample(
        rng: jax.Array,
        observation: _model.Observation,
        prev_chunk: jax.Array,
        prev_valid_mask: jax.Array,
        prefix_steps: int,
    ) -> jax.Array:
        return jitted_core(
            state,
            rng,
            observation,
            prev_chunk,
            prev_valid_mask,
            jnp.asarray(prefix_steps, dtype=jnp.int32),
        )

    return sample


class TrainingPaperRtcSampler:
    """Runs training-paper RTC sampling with server-side robot-prefix encoding."""

    def __init__(
        self,
        policy: _policy.Policy,
        *,
        num_steps: int,
    ) -> None:
        if not isinstance(policy._model, Pi0):  # noqa: SLF001
            raise ValueError(
                "Training-paper RTC requires a standard JAX Pi0/Pi0.5 policy. "
                f"Loaded model type: {type(policy._model).__name__}."
            )
        self._policy = policy
        self._model = policy._model  # noqa: SLF001
        self._action_horizon = self._model.action_horizon
        self._action_dim = self._model.action_dim
        self._sample = _build_training_paper_rtc_sampler(
            self._model,
            num_steps=num_steps,
        )

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _prepare_prefix(
        self,
        action_prefix_model: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        prefix_chunk = np.zeros(
            (1, self._action_horizon, self._action_dim),
            dtype=np.float32,
        )
        prefix_valid = np.zeros((self._action_horizon,), dtype=bool)
        if action_prefix_model is None:
            return prefix_chunk, prefix_valid, 0

        arr = np.asarray(action_prefix_model, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected action prefix rank 2, got {arr.shape}")
        copy_len = min(arr.shape[0], self._action_horizon)
        copy_dim = min(arr.shape[1], self._action_dim)
        if copy_len > 0 and copy_dim > 0:
            prefix_chunk[0, :copy_len, :copy_dim] = arr[:copy_len, :copy_dim]
            prefix_valid[:copy_len] = True
        return prefix_chunk, prefix_valid, copy_len

    def infer_chunk(
        self,
        obs: dict,
        *,
        action_prefix_model: np.ndarray | None,
        prefix_steps: int,
    ) -> dict[str, np.ndarray | int]:
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._policy._input_transform(inputs)  # noqa: SLF001
        batched_inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        observation = _model.Observation.from_dict(batched_inputs)

        prefix_chunk, prefix_valid, prefix_len = self._prepare_prefix(action_prefix_model)

        self._policy._rng, sample_rng = jax.random.split(self._policy._rng)  # noqa: SLF001
        sampled_actions = self._sample(
            sample_rng,
            observation,
            jnp.asarray(prefix_chunk),
            jnp.asarray(prefix_valid),
            prefix_steps,
        )
        sampled_actions_np = np.asarray(sampled_actions[0], dtype=np.float32)

        outputs: dict[str, Any] = {
            "state": np.asarray(batched_inputs["state"][0]),
            "actions": sampled_actions_np,
        }
        outputs = self._policy._output_transform(outputs)  # noqa: SLF001
        robot_actions = np.asarray(outputs["actions"], dtype=np.float32)
        return {
            "actions": robot_actions,
            "prefix_input_len": np.asarray(prefix_len, dtype=np.int32),
            "conditioned_prefix_steps": np.asarray(
                min(max(0, int(prefix_steps)), int(prefix_len), self._action_horizon),
                dtype=np.int32,
            ),
        }


class LatencyTracker:
    def __init__(self, maxlen: int) -> None:
        self._values: collections.deque[float] = collections.deque(maxlen=maxlen)

    def add(self, seconds: float) -> None:
        if seconds >= 0.0:
            self._values.append(float(seconds))

    def estimate(self, percentile: float, default_s: float) -> float:
        if not self._values:
            return default_s
        values = np.asarray(self._values, dtype=np.float32)
        if percentile >= 1.0:
            return float(np.max(values))
        if percentile <= 0.0:
            return float(np.min(values))
        return float(np.quantile(values, percentile))

    def __len__(self) -> int:
        return len(self._values)


class ActionQueue:
    """Thread-safe robot action queue."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._robot_queue: np.ndarray | None = None
        self._index = 0
        self._served_total = 0

    def clear(self) -> None:
        with self._lock:
            self._robot_queue = None
            self._index = 0
            self._served_total = 0

    def get(self) -> np.ndarray | None:
        with self._lock:
            if self._robot_queue is None or self._index >= len(self._robot_queue):
                return None
            action = self._robot_queue[self._index].copy()
            self._index += 1
            self._served_total += 1
            return action

    def note_control_step(self) -> None:
        with self._lock:
            self._served_total += 1

    def remaining(self) -> int:
        with self._lock:
            if self._robot_queue is None:
                return 0
            return max(0, len(self._robot_queue) - self._index)

    def status(self) -> tuple[int, int]:
        with self._lock:
            if self._robot_queue is None:
                return self._index, 0
            return self._index, max(0, len(self._robot_queue) - self._index)

    def snapshot_leftover_robot(self) -> tuple[np.ndarray | None, int, int, int]:
        with self._lock:
            leftover = None
            if self._robot_queue is not None:
                leftover = self._robot_queue[self._index :].copy()
            remaining = 0
            if self._robot_queue is not None:
                remaining = max(0, len(self._robot_queue) - self._index)
            return leftover, self._index, self._served_total, remaining

    def total_served(self) -> int:
        with self._lock:
            return self._served_total

    def replace(
        self,
        robot_actions: np.ndarray,
        real_delay: int,
        *,
        last_action: np.ndarray | None = None,
    ) -> tuple[int, dict[str, float | int]]:
        with self._lock:
            delay = max(0, min(int(real_delay), len(robot_actions)))
            new_robot = robot_actions[delay:].copy()

            diagnostics: dict[str, float | int] = {
                "boundary_jump_pre": 0.0,
                "boundary_jump_post": 0.0,
            }
            if last_action is not None and len(new_robot) > 0:
                last = np.asarray(last_action, dtype=np.float32)
                if last.shape == new_robot[0].shape:
                    diagnostics["boundary_jump_pre"] = float(np.max(np.abs(new_robot[0] - last)))

            if last_action is not None and len(new_robot) > 0:
                last = np.asarray(last_action, dtype=np.float32)
                if last.shape == new_robot[0].shape:
                    diagnostics["boundary_jump_post"] = float(np.max(np.abs(new_robot[0] - last)))

            self._robot_queue = new_robot
            self._index = 0
            return delay, diagnostics


class TrainingPaperRtcPolicy:
    """Training-paper RTC chunk controller around a transformed OpenPI policy."""

    def __init__(
        self,
        policy: _policy.Policy,
        args: Args,
    ) -> None:
        self._args = args
        self._sampler = TrainingPaperRtcSampler(
            policy,
            num_steps=args.num_denoising_steps,
        )
        if args.action_horizon != self._sampler.action_horizon:
            logger.warning(
                "--action-horizon=%d differs from model action_horizon=%d; using model value.",
                args.action_horizon,
                self._sampler.action_horizon,
            )
        self._action_horizon = self._sampler.action_horizon
        self._queue = ActionQueue()
        self._latencies = LatencyTracker(args.delay_buffer_size)

        self._cv = threading.Condition()
        self._latest_obs: dict | None = None
        self._trigger_requested = False
        self._inference_active = False
        self._running = False
        self._thread: threading.Thread | None = None

        self._last_action: np.ndarray | None = None
        self._prev_filtered_action: np.ndarray | None = None
        self._diagnostics: dict[str, Any] = {}
        self._last_chunk_ready_s: float | None = None
        self._last_trigger_s: float | None = None
        self._trigger_count = 0
        self._skip_first_latency_sample = True

        self._record_dir: pathlib.Path | None = None
        self._record_step = 0
        if args.record:
            self._record_dir = pathlib.Path("policy_records") / "training_paper_rtc"
            self._record_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Recording training-paper RTC chunks to %s", self._record_dir)

    def start(self) -> None:
        with self._cv:
            self._running = True
            self._thread = None
        logger.info(
            "Training-paper RTC running in client chunk-only mode; "
            "single-step queue is disabled."
        )

    def stop(self) -> None:
        with self._cv:
            self._running = False
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=30.0)

    def reset(self) -> None:
        with self._cv:
            self._queue.clear()
            self._latest_obs = None
            self._trigger_requested = False
            self._inference_active = False
            self._last_action = None
            self._prev_filtered_action = None
            self._diagnostics = {}
            self._last_chunk_ready_s = None
            self._last_trigger_s = None
            self._trigger_count = 0
            self._cv.notify_all()

    def metadata(self) -> dict[str, Any]:
        policy_config = None
        checkpoint_dir = None
        if isinstance(self._args.policy, Checkpoint):
            policy_config = self._args.policy.config
            checkpoint_dir = self._args.policy.dir
        elif checkpoint := DEFAULT_CHECKPOINT.get(self._args.env):
            policy_config = checkpoint.config
            checkpoint_dir = checkpoint.dir

        return {
            "rtc_type": "training_paper_rtc_action_conditioning",
            "config_name": policy_config,
            "checkpoint_dir": checkpoint_dir,
            "action_horizon": self._action_horizon,
            "robot_action_dim": 14,
            "model_action_dim": self._sampler.action_dim,
            "execution_action_space": execution_action_space(self._args),
            "execution_action_schema": "agilex_bimanual_v1",
            "control_hz": self._args.control_hz,
            "num_denoising_steps": self._args.num_denoising_steps,
            "execution_horizon": self._args.execution_horizon,
            "trained_simulated_delay": self._args.trained_simulated_delay,
            "max_conditioned_prefix_steps": self._args.max_conditioned_prefix_steps,
            "conditioned_prefix_source": "client_fixed_delay_steps",
            "uses_pigdm_guidance": False,
            "delay_percentile": self._args.delay_percentile,
            "filter_alpha": self._args.filter_alpha,
            "supports_client_chunk_execution": True,
            "supports_single_step_queue": False,
            "prefix_attention_horizon": max(
                0,
                self._action_horizon - min(self._args.execution_horizon, self._action_horizon),
            ),
        }

    def diagnostics(self) -> dict[str, Any]:
        with self._cv:
            diagnostics = dict(self._diagnostics)
            if self._last_chunk_ready_s is not None:
                diagnostics["chunk_age_ms"] = (
                    time.monotonic() - self._last_chunk_ready_s
                ) * 1000
            return diagnostics

    def _estimate_delay_steps(self) -> int:
        seconds = self._latencies.estimate(
            self._args.delay_percentile,
            self._args.initial_delay_s,
        )
        estimate = int(math.ceil(seconds * self._args.control_hz))
        max_delay = self._args.max_delay_steps
        if max_delay is None:
            max_delay = max(1, self._action_horizon - 1)
        estimate = max(self._args.min_delay_steps, estimate)
        estimate = min(max_delay, estimate, self._action_horizon - 1)
        return estimate

    def _replan_min_steps(self) -> int:
        return min(self._action_horizon - 1, max(1, self._args.execution_horizon))

    def _replan_threshold(self) -> int:
        return self._replan_min_steps()

    def _maybe_request_inference_locked(self) -> None:
        if self._inference_active:
            return
        action_idx, remaining = self._queue.status()
        if self._latest_obs is None:
            return
        s_min = self._replan_min_steps()
        if remaining == 0 or action_idx >= s_min:
            if not self._trigger_requested:
                self._trigger_count += 1
                self._last_trigger_s = time.monotonic()
                self._diagnostics.update(
                    {
                        "trigger_count": self._trigger_count,
                        "trigger_action_idx": action_idx,
                        "trigger_remaining": remaining,
                        "trigger_threshold": s_min,
                        "trigger_prefix_attention_horizon": remaining,
                    }
                )
            self._trigger_requested = True
            self._cv.notify_all()

    def _filter_action(self, action: np.ndarray) -> np.ndarray:
        if self._args.filter_alpha >= 1.0:
            return action
        if self._prev_filtered_action is None:
            self._prev_filtered_action = action.copy()
            return action
        filtered = (
            self._args.filter_alpha * action
            + (1.0 - self._args.filter_alpha) * self._prev_filtered_action
        )
        self._prev_filtered_action = filtered.copy()
        return filtered

    @staticmethod
    def _is_client_chunk_request(obs: dict | None) -> bool:
        return isinstance(obs, dict) and bool(
            obs.get("_paper_rtc_client_chunk_request")
            or obs.get("_paper_rtc_chunk_request")
            or obs.get("_training_paper_rtc_client_chunk_request")
        )

    @staticmethod
    def _strip_private_request_keys(obs: dict) -> dict:
        return {
            key: value
            for key, value in obs.items()
            if not str(key).startswith(("_paper_rtc_", "_training_paper_rtc_"))
        }

    def _infer_client_chunk(
        self,
        clean_obs: dict,
        *,
        action_prefix_model: np.ndarray | None,
        prefix_steps: int,
        request_obs: dict,
    ) -> dict[str, Any]:
        """Extension point for joint action-side outputs.

        The stock server ignores ``request_obs`` and preserves the original
        action-only sampler path.  The integrated completion entrypoint
        overrides this method in its isolated repository.
        """

        del request_obs
        return self._sampler.infer_chunk(
            clean_obs,
            action_prefix_model=action_prefix_model,
            prefix_steps=prefix_steps,
        )

    def generate_client_chunk(self, obs: dict) -> dict[str, Any]:
        """Generate a full RTC chunk for client-side chunk execution."""
        if not isinstance(obs, dict):
            raise ValueError("Training-paper RTC client chunk request must be a dict observation.")

        request_start = time.monotonic()
        clean_obs = self._strip_private_request_keys(obs)
        action_prefix_model = None
        prefix_source = "none"
        prev_leftover_robot = obs.get("_paper_rtc_prev_leftover_robot")
        if prev_leftover_robot is not None:
            prev_leftover_robot = np.asarray(prev_leftover_robot, dtype=np.float32)
            convert_leftover = getattr(self._sampler, "model_prefix_from_robot_actions", None)
            if convert_leftover is None:
                raise ValueError("Received robot-space leftover actions, but this sampler cannot convert them.")
            action_prefix_model = convert_leftover(clean_obs, prev_leftover_robot)
            prefix_source = "robot_absolute_reencoded"

        fixed_delay = int(obs.get("_paper_rtc_inference_delay_steps", self._args.min_delay_steps))
        prefix_attention_horizon = int(
            obs.get(
                "_paper_rtc_prefix_attention_horizon",
                0 if action_prefix_model is None else len(action_prefix_model),
            )
        )
        prefix_attention_horizon = max(0, min(prefix_attention_horizon, self._action_horizon))
        fixed_delay = max(0, min(fixed_delay, prefix_attention_horizon))
        prefix_steps = int(obs.get("_training_paper_rtc_prefix_steps", fixed_delay))
        prefix_steps = max(0, min(prefix_steps, prefix_attention_horizon, self._action_horizon))
        if self._args.max_conditioned_prefix_steps is not None:
            prefix_steps = min(prefix_steps, self._args.max_conditioned_prefix_steps)

        logger.info(
            "Training-paper RTC client chunk request step=%s idx=%s "
            "d_fixed=%d conditioned_prefix=%d prefix_h=%d action_prefix=%s",
            obs.get("_paper_rtc_client_step", "?"),
            obs.get("_paper_rtc_local_chunk_index", "?"),
            fixed_delay,
            prefix_steps,
            prefix_attention_horizon,
            None if action_prefix_model is None else action_prefix_model.shape,
        )

        infer_start = time.monotonic()
        result = self._infer_client_chunk(
            clean_obs,
            action_prefix_model=action_prefix_model,
            prefix_steps=prefix_steps,
            request_obs=obs,
        )
        infer_s = time.monotonic() - infer_start

        latency_sample_skipped = False
        if self._skip_first_latency_sample:
            self._skip_first_latency_sample = False
            latency_sample_skipped = True
        else:
            self._latencies.add(infer_s)

        robot_chunk = np.asarray(result["actions"], dtype=np.float32)
        if robot_chunk.ndim != 2 or robot_chunk.shape[-1] != 14:
            raise ValueError(
                "Training-paper RTC expected dual-arm AgileX robot chunk shape (H, 14), "
                f"got {robot_chunk.shape}"
            )

        diagnostics = {
            "chunk_mode": "client_execution",
            "last_infer_ms": infer_s * 1000,
            "delay_est_steps": fixed_delay,
            "fixed_delay_steps": fixed_delay,
            "conditioned_prefix_steps": int(result["conditioned_prefix_steps"]),
            "execution_horizon": self._replan_min_steps(),
            "prefix_attention_horizon": prefix_attention_horizon,
            "action_horizon": int(robot_chunk.shape[0]),
            "prefix_input_len": int(result["prefix_input_len"]),
            "prefix_source": prefix_source,
            "latency_samples": len(self._latencies),
            "latency_sample_skipped": latency_sample_skipped,
            "client_step": obs.get("_paper_rtc_client_step", -1),
            "client_local_chunk_index": obs.get("_paper_rtc_local_chunk_index", -1),
            "client_local_remaining": obs.get("_paper_rtc_local_remaining", -1),
            "server_prepare_ms": (time.monotonic() - request_start) * 1000,
        }
        self._record_chunk(clean_obs, result, diagnostics)
        with self._cv:
            self._last_chunk_ready_s = time.monotonic()
            self._diagnostics.update(diagnostics)
            self._diagnostics["inference_active"] = False

        response = {
            "actions": robot_chunk,
            "server_timing": diagnostics,
        }
        if "completion" in result:
            response["completion"] = result["completion"]
        return response

    def get_action(self, obs: dict) -> np.ndarray:
        raise RuntimeError(
            "Training-paper RTC single-step obs/tick requests are disabled; "
            "send _paper_rtc_client_chunk_request=True and execute chunks on the client."
        )

    def _record_chunk(
        self,
        obs: dict,
        result: dict[str, np.ndarray | int],
        diagnostics: dict[str, Any],
    ) -> None:
        if self._record_dir is None:
            return
        payload = {
            "obs": obs,
            "result": result,
            "diagnostics": diagnostics,
        }
        path = self._record_dir / f"chunk_{self._record_step:06d}.npy"
        self._record_step += 1
        np.save(path, np.asarray(payload, dtype=object), allow_pickle=True)

    def _inference_loop(self) -> None:
        logger.info("Training-paper RTC inference thread started.")
        while True:
            with self._cv:
                self._cv.wait_for(
                    lambda: not self._running or self._trigger_requested
                )
                if not self._running:
                    return
                obs = self._latest_obs
                self._trigger_requested = False
                self._inference_active = True
                self._diagnostics["inference_active"] = True
                trigger_monotonic_s = self._last_trigger_s

            if obs is None:
                with self._cv:
                    self._inference_active = False
                continue

            prev_leftover_robot, action_idx, served_before, remaining = (
                self._queue.snapshot_leftover_robot()
            )
            delay_est = self._estimate_delay_steps()
            s_min = self._replan_min_steps()
            prefix_attention_horizon = 0 if prev_leftover_robot is None else remaining
            prefix_steps = min(delay_est, prefix_attention_horizon)
            if self._args.max_conditioned_prefix_steps is not None:
                prefix_steps = min(prefix_steps, self._args.max_conditioned_prefix_steps)

            logger.info(
                "Training-paper RTC: trigger s=%d s_min=%d remaining=%d prefix_h=%d d_est=%d",
                action_idx,
                s_min,
                remaining,
                prefix_attention_horizon,
                delay_est,
            )

            start = time.monotonic()
            action_prefix_model = None
            if prev_leftover_robot is not None:
                convert_leftover = getattr(self._sampler, "model_prefix_from_robot_actions", None)
                if convert_leftover is None:
                    raise ValueError("Received robot-space leftover actions, but this sampler cannot convert them.")
                action_prefix_model = convert_leftover(obs, prev_leftover_robot)
            result = self._sampler.infer_chunk(
                obs,
                action_prefix_model=action_prefix_model,
                prefix_steps=prefix_steps,
            )
            infer_s = time.monotonic() - start
            latency_sample_skipped = False
            if self._skip_first_latency_sample:
                # The first real inference usually includes JIT compilation. Keep it
                # out of the RTC delay estimator, then use normal samples afterward.
                self._skip_first_latency_sample = False
                latency_sample_skipped = True
            else:
                self._latencies.add(infer_s)

            robot_chunk = np.asarray(result["actions"], dtype=np.float32)
            if robot_chunk.ndim != 2 or robot_chunk.shape[-1] != 14:
                raise ValueError(
                    "Training-paper RTC expected dual-arm AgileX robot chunk shape (H, 14), "
                    f"got {robot_chunk.shape}"
                )

            served_after = self._queue.total_served()
            real_delay = max(0, served_after - served_before)
            last_action = None if self._last_action is None else self._last_action.copy()
            skipped, boundary_diag = self._queue.replace(
                robot_chunk,
                real_delay,
                last_action=last_action,
            )

            diagnostics = {
                "last_infer_ms": infer_s * 1000,
                "delay_est_steps": delay_est,
                "conditioned_prefix_steps": int(result["conditioned_prefix_steps"]),
                "real_delay_steps": real_delay,
                "skipped_steps": skipped,
                "prefix_input_len": int(result["prefix_input_len"]),
                "queue_remaining": self._queue.remaining(),
                "execution_horizon": s_min,
                "prefix_attention_horizon": prefix_attention_horizon,
                "chunk_action_idx_before": action_idx,
                "chunk_remaining_before": remaining,
                "served_before": served_before,
                "served_after": served_after,
                "latency_samples": len(self._latencies),
                "latency_sample_skipped": latency_sample_skipped,
                **boundary_diag,
            }
            if isinstance(trigger_monotonic_s, float):
                diagnostics["trigger_to_infer_start_ms"] = (
                    start - trigger_monotonic_s
                ) * 1000
            self._record_chunk(obs, result, diagnostics)

            with self._cv:
                self._last_chunk_ready_s = time.monotonic()
                self._diagnostics.update(diagnostics)
                self._inference_active = False
                self._diagnostics["inference_active"] = False
                self._cv.notify_all()

            logger.info(
                "Training-paper RTC: chunk ready infer=%.1fms d_est=%d real_delay=%d "
                "skipped=%d queue=%d prefix_input=%d prefix_h=%d jump %.4f->%.4f "
                "latency_skipped=%s",
                infer_s * 1000,
                delay_est,
                real_delay,
                skipped,
                self._queue.remaining(),
                int(result["prefix_input_len"]),
                prefix_attention_horizon,
                float(boundary_diag.get("boundary_jump_pre", 0.0)),
                float(boundary_diag.get("boundary_jump_post", 0.0)),
                latency_sample_skipped,
            )


class TrainingPaperRtcWebsocketPolicyServer:
    def __init__(
        self,
        rtc_policy: TrainingPaperRtcPolicy,
        *,
        host: str,
        port: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._rtc_policy = rtc_policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        self._rtc_policy.start()
        try:
            asyncio.run(self.run())
        finally:
            self._rtc_policy.stop()

    async def run(self) -> None:
        async with _ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _ws_server.ServerConnection) -> None:
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self._metadata))
        self._rtc_policy.reset()

        loop = asyncio.get_running_loop()
        prev_total_time = None
        prev_pack_ms = None
        prev_send_ms = None
        while True:
            try:
                recv_wait_start = time.monotonic()
                message = await websocket.recv()
                recv_done = time.monotonic()
                request_start = recv_done
                if isinstance(message, (bytes, bytearray, memoryview)):
                    request_rx_bytes = len(message)
                else:
                    request_rx_bytes = len(str(message).encode("utf-8"))

                unpack_start = time.monotonic()
                obs = msgpack_numpy.unpackb(message)
                if not isinstance(obs, dict):
                    raise TypeError(
                        f"RA JPEG request must unpack to a dict, got {type(obs).__name__}"
                    )
                obs = _decode_ra_jpeg_observation(obs)
                unpack_ms = (time.monotonic() - unpack_start) * 1000
                is_client_chunk_request = TrainingPaperRtcPolicy._is_client_chunk_request(obs)
                if not is_client_chunk_request:
                    raise ValueError(
                        "Training-paper RTC single-step obs/tick requests are disabled. "
                        "Client requests must include _paper_rtc_client_chunk_request=True."
                    )
                request_kind = "client_chunk"

                t0 = time.monotonic()
                policy_result = await loop.run_in_executor(
                    None,
                    self._rtc_policy.generate_client_chunk,
                    obs,
                )
                action = policy_result["actions"]
                policy_timing = dict(policy_result.get("server_timing", {}))
                t_get = time.monotonic() - t0

                response = {
                    "actions": action,
                    "server_timing": {
                        "get_action_ms": t_get * 1000,
                        "server_recv_wait_ms": (recv_done - recv_wait_start) * 1000,
                        "server_unpack_ms": unpack_ms,
                        "request_rx_kib": request_rx_bytes / 1024,
                        "request_kind": request_kind,
                        **policy_timing,
                    },
                }
                if "completion" in policy_result:
                    response["completion"] = policy_result["completion"]
                if prev_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                if prev_pack_ms is not None:
                    response["server_timing"]["prev_pack_ms"] = prev_pack_ms
                if prev_send_ms is not None:
                    response["server_timing"]["prev_send_ms"] = prev_send_ms

                pack_start = time.monotonic()
                payload = packer.pack(response)
                pack_ms = (time.monotonic() - pack_start) * 1000
                response["server_timing"]["server_pack_ms"] = pack_ms
                response["server_timing"]["response_tx_kib"] = len(payload) / 1024
                response["server_timing"]["server_prepare_ms"] = (
                    time.monotonic() - request_start
                ) * 1000
                payload = packer.pack(response)

                send_start = time.monotonic()
                await websocket.send(payload)
                send_ms = (time.monotonic() - send_start) * 1000
                prev_total_time = time.monotonic() - request_start
                prev_pack_ms = pack_ms
                prev_send_ms = send_ms

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: _ws_server.ServerConnection,
    request: _ws_server.Request,
) -> _ws_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def _log_args(args: Args, policy_metadata: dict[str, Any] | None, rtc_metadata: dict[str, Any]) -> None:
    logger.info("Training-paper RTC server parameters:")
    for key, value in rtc_metadata.items():
        logger.info("  %-28s %s", key + ":", value)
    if policy_metadata:
        logger.info("Policy metadata: %s", policy_metadata)


def main(args: Args) -> None:
    _validate_args(args)

    # Keep the same numerical guard used by the existing RTC scripts.
    jax.config.update("jax_default_matmul_precision", "float32")
    os.environ["OPENPI_FLOAT32_INFERENCE"] = "1"

    policy = create_policy(args)
    policy_metadata = policy.metadata
    rtc_policy = TrainingPaperRtcPolicy(policy, args)

    metadata = dict(policy_metadata or {})
    metadata["paper_rtc"] = rtc_policy.metadata()
    metadata["training_paper_rtc"] = metadata["paper_rtc"]
    if "completion" in metadata["training_paper_rtc"]:
        metadata["completion"] = metadata["training_paper_rtc"]["completion"]
    metadata["ra_image_transport"] = dict(_RA_IMAGE_TRANSPORT_METADATA)
    _log_args(args, policy_metadata, metadata["training_paper_rtc"])

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info(
        "Creating training-paper RTC server host=%s ip=%s port=%d",
        hostname,
        local_ip,
        args.port,
    )

    server = TrainingPaperRtcWebsocketPolicyServer(
        rtc_policy,
        host="0.0.0.0",
        port=args.port,
        metadata=metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
