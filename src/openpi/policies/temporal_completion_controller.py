"""Causal prompt controller for the 2 Hz temporal completion head.

The controller is intentionally model-agnostic.  The action policy supplies one
frozen-prefix feature on each completion tick and a callback scores the oldest
to newest three-feature history.  Prompt changes are latched and take effect for
the next policy forward; the completion history is cleared at every change.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import dataclasses
import numbers
from typing import Any, Protocol

import numpy as np

ScoreFn = Callable[[np.ndarray], float | np.ndarray]


def _trajectory_frame_index(value: Any) -> int:
    """Requires an integer frame index relative to the current trajectory."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"trajectory_frame_index must be an integer, got {value!r}")
    return int(value)


class TemporalFeaturePolicy(Protocol):
    def infer(self, obs: dict[str, Any], **kwargs: Any) -> dict[str, Any]: ...

    def score_temporal_completion(self, prefix_history: np.ndarray, *, return_logit: bool = False) -> float: ...


@dataclasses.dataclass(frozen=True)
class TemporalCompletionDecision:
    """Outcome of one 2 Hz controller tick."""

    frame_index: int
    task_before: int
    task_after: int
    score: float | None
    history_ready: bool
    triggered: bool
    done: bool
    invalidate_action_plan: bool


class TemporalCompletionController:
    """Maintains prompt and exact-gap prefix history for completion inference."""

    def __init__(
        self,
        prompts: tuple[str, ...] | list[str],
        *,
        threshold: float,
        tick_stride_frames: int = 15,
        history_steps: int = 3,
    ) -> None:
        prompts = tuple(str(prompt) for prompt in prompts)
        if not prompts or any(not prompt for prompt in prompts):
            raise ValueError("temporal completion prompts must be non-empty strings")
        maximum_threshold = float(np.nextafter(1.0, np.inf))
        if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= maximum_threshold:
            raise ValueError("temporal completion threshold must lie in [0, nextafter(1,+inf)]")
        if tick_stride_frames != 15:
            raise ValueError("the locked 30 fps/2 Hz scheme requires tick_stride_frames=15")
        if history_steps != 3:
            raise ValueError("the locked temporal completion scheme requires exactly three history steps")

        self._prompts = prompts
        self._threshold = float(threshold)
        self._tick_stride_frames = int(tick_stride_frames)
        self._history_steps = int(history_steps)
        self.reset()

    def reset(self) -> None:
        """Resets the controller to the first subtask without changing config."""

        self._task_index = 0
        self._history: list[np.ndarray] = []
        self._feature_dim: int | None = None
        self._last_frame_index: int | None = None
        self._done = False
        self._triggered_boundaries: list[int] = []

    @property
    def current_task_index(self) -> int:
        return self._task_index

    @property
    def current_prompt(self) -> str:
        return self._prompts[self._task_index]

    @property
    def done(self) -> bool:
        return self._done

    @property
    def triggered_boundaries(self) -> tuple[int, ...]:
        return tuple(self._triggered_boundaries)

    @property
    def history_size(self) -> int:
        return len(self._history)

    def override_prompt(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Returns an observation with the controller prompt explicitly set.

        This must be called before task/default-prompt transforms.  A copy is
        returned so the caller's source observation is not mutated.
        """

        result = dict(observation)
        result["prompt"] = self.current_prompt
        return result

    def step(
        self,
        frame_index: int,
        prefix_feature: np.ndarray,
        score_fn: ScoreFn,
    ) -> TemporalCompletionDecision:
        """Consumes one current-tick feature and optionally advances the prompt."""

        frame_index = _trajectory_frame_index(frame_index)
        task_before = self._task_index
        if self._done:
            raise RuntimeError("temporal completion controller is already done")
        if frame_index < 0 or frame_index % self._tick_stride_frames != 0:
            raise ValueError(f"completion frame {frame_index} is not on the {self._tick_stride_frames}-frame tick grid")
        if self._last_frame_index is not None:
            expected = self._last_frame_index + self._tick_stride_frames
            if frame_index != expected:
                raise ValueError(f"completion ticks must be exact-gap: expected {expected}, got {frame_index}")
        self._last_frame_index = frame_index

        feature = np.asarray(prefix_feature, dtype=np.float32)
        if feature.ndim != 1 or feature.size == 0:
            raise ValueError(f"prefix_feature must have shape [D], got {feature.shape}")
        if not np.isfinite(feature).all():
            raise ValueError("prefix_feature contains non-finite values")
        if self._feature_dim is None:
            self._feature_dim = int(feature.shape[0])
        elif feature.shape != (self._feature_dim,):
            raise ValueError(f"prefix feature dimension changed from {self._feature_dim} to {feature.shape[0]}")

        self._history.append(np.array(feature, copy=True))
        if len(self._history) > self._history_steps:
            self._history.pop(0)
        if len(self._history) < self._history_steps:
            return TemporalCompletionDecision(
                frame_index=frame_index,
                task_before=task_before,
                task_after=self._task_index,
                score=None,
                history_ready=False,
                triggered=False,
                done=False,
                invalidate_action_plan=False,
            )

        history = np.stack(self._history, axis=0).astype(np.float32, copy=False)
        raw_score = np.asarray(score_fn(history))
        if raw_score.size != 1:
            raise ValueError(f"score_fn must return one scalar, got shape {raw_score.shape}")
        score = float(raw_score.reshape(-1)[0])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score_fn must return a probability in [0, 1]")

        triggered = score >= self._threshold
        invalidate_action_plan = False
        if triggered:
            self._triggered_boundaries.append(frame_index)
            invalidate_action_plan = True
            if self._task_index == len(self._prompts) - 1:
                self._done = True
            else:
                self._task_index += 1
                # Never mix frozen-prefix features produced under two prompts.
                self._history.clear()

        return TemporalCompletionDecision(
            frame_index=frame_index,
            task_before=task_before,
            task_after=self._task_index,
            score=score,
            history_ready=True,
            triggered=triggered,
            done=self._done,
            invalidate_action_plan=invalidate_action_plan,
        )


class TemporalCompletionPolicy:
    """Adds causal prompt switching to a prefix-feature-capable action policy.

    The caller supplies the 30 fps frame index relative to the current full
    trajectory, starting at zero.  Completion is scored only on its fixed 2 Hz
    grid.  A trigger applies the new prompt on the next call and marks the
    just-produced old-prompt action plan invalid, allowing a broker/controller
    to discard its pending chunk explicitly.  Instances are single-rollout
    mutable state and must not be shared across concurrent clients.
    """

    def __init__(
        self,
        policy: TemporalFeaturePolicy,
        prompts: tuple[str, ...] | list[str],
        *,
        threshold: float,
        tick_stride_frames: int = 15,
    ) -> None:
        self._policy = policy
        self._controller = TemporalCompletionController(
            prompts,
            threshold=threshold,
            tick_stride_frames=tick_stride_frames,
        )
        self._tick_stride_frames = int(tick_stride_frames)

    @property
    def controller(self) -> TemporalCompletionController:
        return self._controller

    def reset(self) -> None:
        self._controller.reset()

    def infer(self, obs: dict[str, Any], *, trajectory_frame_index: int, **kwargs: Any) -> dict[str, Any]:
        if self._controller.done:
            raise RuntimeError("temporal completion policy is already done")
        if "return_prefix_feature" in kwargs:
            raise ValueError("TemporalCompletionPolicy owns return_prefix_feature")
        frame_index = _trajectory_frame_index(trajectory_frame_index)
        prompted_observation = self._controller.override_prompt(obs)
        on_tick = frame_index >= 0 and frame_index % self._tick_stride_frames == 0
        action_task_index = self._controller.current_task_index
        result = self._policy.infer(
            prompted_observation,
            # Always use the tuple-return action graph.  Alternating between
            # two separately-jitted full-model methods causes a second cold
            # compile; masked-mean pooling off-tick is negligible and its
            # feature is discarded below.
            return_prefix_feature=True,
            **kwargs,
        )
        result = dict(result)
        result["active_prompt"] = prompted_observation["prompt"]
        result["active_task_index"] = action_task_index
        result["next_prompt"] = self._controller.current_prompt
        result["next_task_index"] = self._controller.current_task_index
        result["invalidate_action_plan"] = False
        if not on_tick:
            result.pop("prefix_feature", None)
            result["completion_decision"] = None
            return result
        if "prefix_feature" not in result:
            raise ValueError("wrapped policy did not return prefix_feature on a completion tick")
        decision = self._controller.step(
            frame_index,
            np.asarray(result.pop("prefix_feature"), dtype=np.float32),
            self._policy.score_temporal_completion,
        )
        result["completion_decision"] = dataclasses.asdict(decision)
        result["completion_score"] = decision.score
        result["next_prompt"] = self._controller.current_prompt
        result["next_task_index"] = decision.task_after
        result["invalidate_action_plan"] = decision.invalidate_action_plan
        return result
