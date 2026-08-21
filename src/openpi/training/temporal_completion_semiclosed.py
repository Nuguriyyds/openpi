"""Pure state and boundary helpers for full-trajectory completion evaluation.

The training evaluator works on sealed subtask rows.  This module is deliberately
independent of datasets and model loading so that the full-trajectory evaluator
can exercise the same causal state machine without importing JAX in its tests.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import itertools
import numbers
from typing import Any, Literal

import numpy as np

FPS = 30
TICK_STRIDE_FRAMES = 15
TASK_COUNT = 4
HistoryMode = Literal["history", "current_only"]


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def validate_lengths(lengths: Sequence[int]) -> tuple[int, int, int, int]:
    values = tuple(_integer(value, name="episode length") for value in lengths)
    if len(values) != TASK_COUNT or any(value <= 0 for value in values):
        raise ValueError(f"lengths must contain four positive integers, got {values!r}")
    return values  # type: ignore[return-value]


def gt_end_frames(lengths: Sequence[int]) -> tuple[int, int, int, int]:
    """Returns inclusive global end frames from four subtask lengths."""

    values = validate_lengths(lengths)
    cumulative = 0
    ends: list[int] = []
    for length in values:
        cumulative += length
        ends.append(cumulative - 1)
    return tuple(ends)  # type: ignore[return-value]


def reference_tick(end_frame: int, *, stride: int = TICK_STRIDE_FRAMES) -> int:
    """Returns the locked reference tick ``stride*((E+14)//15)``.

    The formula intentionally uses the inclusive end frame directly.  Thus an
    end at frame 89 maps to the action/completion tick at frame 90.
    """

    end = _integer(end_frame, name="end_frame")
    if end < 0 or stride != TICK_STRIDE_FRAMES:
        raise ValueError("reference_tick requires a non-negative end and the locked stride of 15")
    return stride * ((end + stride - 1) // stride)


def reference_ticks(end_frames: Sequence[int]) -> tuple[int, int, int, int]:
    if len(end_frames) != TASK_COUNT:
        raise ValueError("end_frames must contain exactly four values")
    return tuple(reference_tick(value) for value in end_frames)  # type: ignore[return-value]


def oracle_task_index(frame_index: int, end_frames: Sequence[int]) -> int:
    """Returns the ground-truth task owning a global frame (for statistics)."""

    frame = _integer(frame_index, name="frame_index")
    ends = tuple(_integer(value, name="end_frame") for value in end_frames)
    if len(ends) != TASK_COUNT or any(right <= left for left, right in itertools.pairwise(ends)):
        raise ValueError(f"end_frames must be four strictly increasing values, got {ends!r}")
    for task, end in enumerate(ends):
        if frame <= end:
            return task
    return TASK_COUNT - 1


@dataclasses.dataclass(frozen=True)
class BoundaryResult:
    task_index: int
    reference_tick: int
    reference_tick_available: bool
    predicted_tick: int | None
    classification: Literal["correct", "early", "late", "missed", "unavailable"]
    timing_error_frames: int | None
    timing_error_seconds: float | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def classify_boundary(
    task_index: int,
    reference: int,
    predicted: int | None,
    *,
    full_length: int,
) -> BoundaryResult:
    task = _integer(task_index, name="task_index")
    ref = _integer(reference, name="reference_tick")
    if task not in range(TASK_COUNT) or ref < 0 or full_length <= 0:
        raise ValueError("invalid boundary classification inputs")
    available = ref < int(full_length)
    if predicted is None:
        status: Literal["correct", "early", "late", "missed", "unavailable"] = "missed"
    elif not available:
        status = "unavailable"
    elif predicted == ref:
        status = "correct"
    elif predicted < ref:
        status = "early"
    else:
        status = "late"
    timing_frames = None if predicted is None or not available else int(predicted - ref)
    return BoundaryResult(
        task_index=task,
        reference_tick=ref,
        reference_tick_available=available,
        predicted_tick=None if predicted is None else int(predicted),
        classification=status,
        timing_error_frames=timing_frames,
        timing_error_seconds=None if timing_frames is None else timing_frames / FPS,
    )


@dataclasses.dataclass(frozen=True)
class TickDecision:
    frame_index: int
    active_task_index: int
    active_prompt: str
    task_before: int
    task_after: int
    history_ready: bool
    score: float | None
    threshold: float
    triggered: bool
    done: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


ScoreFunction = Callable[[np.ndarray], float]


class SemiClosedCompletionController:
    """Causal prompt-switching state for one full trajectory.

    The feature passed to :meth:`step` is always generated using the prompt
    returned by ``active_prompt`` before the call.  A trigger changes the task
    only for the next tick and clears history in history mode.  No ground-truth
    boundary is consulted by this class.
    """

    def __init__(self, prompts: Sequence[str], *, threshold: float, mode: HistoryMode) -> None:
        values = tuple(str(prompt) for prompt in prompts)
        if len(values) != TASK_COUNT or any(not prompt.strip() for prompt in values):
            raise ValueError("exactly four non-empty prompts are required")
        maximum = float(np.nextafter(1.0, np.inf))
        if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= maximum:
            raise ValueError("threshold must be in [0, nextafter(1,+inf)]")
        if mode not in ("history", "current_only"):
            raise ValueError(f"unsupported mode {mode!r}")
        self.prompts = values
        self.threshold = float(threshold)
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        self._task_index = 0
        self._history: list[np.ndarray] = []
        self._last_frame: int | None = None
        self._done = False

    @property
    def current_task_index(self) -> int:
        return self._task_index

    @property
    def current_prompt(self) -> str:
        return self.prompts[self._task_index]

    @property
    def done(self) -> bool:
        return self._done

    @property
    def history_size(self) -> int:
        return len(self._history)

    def step(self, frame_index: int, feature: np.ndarray, score_fn: ScoreFunction) -> TickDecision:
        frame = _integer(frame_index, name="frame_index")
        if frame < 0 or frame % TICK_STRIDE_FRAMES != 0:
            raise ValueError("full-trajectory completion frames must be on the 15-frame tick grid")
        if self._last_frame is not None and frame != self._last_frame + TICK_STRIDE_FRAMES:
            raise ValueError(f"completion ticks must be exact-gap: expected {self._last_frame + 15}, got {frame}")
        self._last_frame = frame
        task_before = self._task_index
        prompt = self.current_prompt
        if self._done:
            return TickDecision(
                frame_index=frame,
                active_task_index=task_before,
                active_prompt=prompt,
                task_before=task_before,
                task_after=task_before,
                history_ready=False,
                score=None,
                threshold=self.threshold,
                triggered=False,
                done=True,
            )

        values = np.asarray(feature, dtype=np.float32)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"prefix feature must be a finite non-empty [D] vector, got {values.shape}")
        history_ready = self.mode == "current_only"
        if self.mode == "history":
            self._history.append(np.array(values, copy=True))
            if len(self._history) > 3:
                self._history.pop(0)
            history_ready = len(self._history) == 3
        if not history_ready:
            return TickDecision(
                frame_index=frame,
                active_task_index=task_before,
                active_prompt=prompt,
                task_before=task_before,
                task_after=task_before,
                history_ready=False,
                score=None,
                threshold=self.threshold,
                triggered=False,
                done=False,
            )

        if self.mode == "current_only":
            head_input = np.stack([np.zeros_like(values), np.zeros_like(values), values], axis=0)
        else:
            head_input = np.stack(self._history, axis=0).astype(np.float32, copy=False)
        raw_score = np.asarray(score_fn(head_input))
        if raw_score.size != 1:
            raise ValueError(f"score_fn must return one scalar, got {raw_score.shape}")
        score = float(raw_score.reshape(-1)[0])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score_fn must return a probability in [0, 1]")
        triggered = score >= self.threshold
        task_after = task_before
        if triggered:
            if task_before == TASK_COUNT - 1:
                self._done = True
            else:
                task_after = task_before + 1
                self._task_index = task_after
                if self.mode == "history":
                    self._history.clear()
        return TickDecision(
            frame_index=frame,
            active_task_index=task_before,
            active_prompt=prompt,
            task_before=task_before,
            task_after=task_after,
            history_ready=True,
            score=score,
            threshold=self.threshold,
            triggered=triggered,
            done=self._done,
        )


def summarize_boundary_results(results: Sequence[BoundaryResult]) -> dict[str, Any]:
    counts = dict.fromkeys(("correct", "early", "late", "missed", "unavailable"), 0)
    for result in results:
        counts[result.classification] += 1
    available = [result for result in results if result.reference_tick_available]
    timing = [result.timing_error_seconds for result in available if result.timing_error_seconds is not None]
    correct = counts["correct"]
    return {
        "count": len(results),
        "correct": correct,
        "early": counts["early"],
        "late": counts["late"],
        "missed": counts["missed"],
        "unavailable": counts["unavailable"],
        "correct_rate": correct / len(available) if available else None,
        "timing_error_seconds_mean": float(np.mean(timing)) if timing else None,
        "timing_error_seconds_median": float(np.median(timing)) if timing else None,
    }
