"""Pure state and boundary helpers for full-trajectory completion evaluation.

The training evaluator works on sealed subtask rows.  This module is deliberately
independent of datasets and model loading so that the full-trajectory evaluator
can exercise the same causal state machine without importing JAX in its tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import itertools
import numbers
from typing import Any, Literal

import numpy as np

FPS = 30
TICK_STRIDE_FRAMES = 15
TASK_COUNT = 4
HistoryMode = Literal["history", "current_only", "transition", "raw_prefix_current", "raw_prefix_history"]
GatedClassification = Literal["early", "on_time", "late_trigger", "timeout_forced"]
GatedSwitchReason = Literal["head_early", "head_on_time", "head_late", "timeout"]


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


ScoreFunction = Callable[[Any], Any]


def _parse_score_output(value: Any) -> tuple[float, float | None]:
    """Accepts legacy probability scores or ``(score, logit)`` pairs."""

    logit: float | None = None
    if isinstance(value, tuple) and len(value) == 2:
        value, raw_logit = value
        raw_logit_array = np.asarray(raw_logit)
        if raw_logit_array.size != 1:
            raise ValueError(f"score_fn logit must be scalar, got {raw_logit_array.shape}")
        logit = float(raw_logit_array.reshape(-1)[0])
        if not np.isfinite(logit):
            raise ValueError("score_fn logit must be finite")
    raw_score = np.asarray(value)
    if raw_score.size != 1:
        raise ValueError(f"score_fn must return one scalar, got {raw_score.shape}")
    score = float(raw_score.reshape(-1)[0])
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score_fn must return a probability in [0, 1]")
    return score, logit


def _validate_raw_prefix_input(feature: Any) -> dict[str, np.ndarray]:
    if not isinstance(feature, Mapping):
        raise ValueError("raw_prefix_current expects a mapping with prefix_out, prefix_mask, and layout ids")
    required = {"prefix_out", "prefix_mask", "prefix_segment_ids", "prefix_position_ids"}
    if not required.issubset(feature):
        raise ValueError(f"raw_prefix_current input is missing {sorted(required - set(feature))}")
    prefix_out = np.asarray(feature["prefix_out"], dtype=np.float32)
    prefix_mask = np.asarray(feature["prefix_mask"], dtype=np.bool_)
    segment_ids = np.asarray(feature["prefix_segment_ids"], dtype=np.int32)
    position_ids = np.asarray(feature["prefix_position_ids"], dtype=np.int32)
    if prefix_out.ndim != 2 or prefix_out.shape[-1] <= 0 or not np.isfinite(prefix_out).all():
        raise ValueError(f"raw_prefix_current prefix_out must be finite [S, D], got {prefix_out.shape}")
    if prefix_mask.shape != prefix_out.shape[:1]:
        raise ValueError("raw_prefix_current prefix_mask must have shape [S]")
    if segment_ids.shape != prefix_mask.shape or position_ids.shape != prefix_mask.shape:
        raise ValueError("raw_prefix_current layout ids must have shape [S]")
    return {
        "prefix_out": prefix_out,
        "prefix_mask": prefix_mask,
        "prefix_segment_ids": segment_ids,
        "prefix_position_ids": position_ids,
    }


def _copy_raw_prefix_input(feature: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.array(value, copy=True) for name, value in feature.items()}


class SemiClosedCompletionController:
    """Causal prompt-switching state for one full trajectory.

    The feature passed to :meth:`step` is always generated using the prompt
    returned by ``active_prompt`` before the call.  A trigger changes the task
    only for the next tick.  ``history`` mode clears the prefix history after
    a switch, while ``transition`` mode keeps it so the next inputs naturally
    evolve through ``[old, old, new]``, ``[old, new, new]``, and
    ``[new, new, new]``.  No ground-truth boundary is consulted by this class.
    """

    def __init__(self, prompts: Sequence[str], *, threshold: float, mode: HistoryMode) -> None:
        values = tuple(str(prompt) for prompt in prompts)
        if len(values) != TASK_COUNT or any(not prompt.strip() for prompt in values):
            raise ValueError("exactly four non-empty prompts are required")
        maximum = float(np.nextafter(1.0, np.inf))
        if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= maximum:
            raise ValueError("threshold must be in [0, nextafter(1,+inf)]")
        if mode not in ("history", "current_only", "transition", "raw_prefix_current", "raw_prefix_history"):
            raise ValueError(f"unsupported mode {mode!r}")
        self.prompts = values
        self.threshold = float(threshold)
        self.mode = mode
        self.reset()

    def reset(self) -> None:
        self._task_index = 0
        self._history: list[Any] = []
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

        if self.mode == "raw_prefix_current":
            values = None
            head_input = _validate_raw_prefix_input(feature)
            history_ready = True
        elif self.mode == "raw_prefix_history":
            values = None
            current = _validate_raw_prefix_input(feature)
            self._history.append(_copy_raw_prefix_input(current))
            if len(self._history) > 3:
                self._history.pop(0)
            history_ready = len(self._history) == 3
            head_input = tuple(self._history) if history_ready else None
        else:
            values = np.asarray(feature, dtype=np.float32)
            if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"prefix feature must be a finite non-empty [D] vector, got {values.shape}")
            history_ready = self.mode == "current_only"
            if self.mode in ("history", "transition"):
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

        if self.mode not in ("raw_prefix_current", "raw_prefix_history"):
            if self.mode == "current_only":
                head_input = np.stack([np.zeros_like(values), np.zeros_like(values), values], axis=0)
            else:
                head_input = np.stack(self._history, axis=0).astype(np.float32, copy=False)
        score, _logit = _parse_score_output(score_fn(head_input))
        triggered = score >= self.threshold
        task_after = task_before
        if triggered:
            if task_before == TASK_COUNT - 1:
                self._done = True
            else:
                task_after = task_before + 1
                self._task_index = task_after
                if self.mode in ("history", "raw_prefix_history"):
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


@dataclasses.dataclass(frozen=True)
class GatedTaskResult:
    """Outcome of one task under gated terminal-hold replay."""

    task_index: int
    playback_start_frame: int
    gt_end_frame: int
    terminal_arrival_rollout_tick: int | None
    trigger_rollout_tick: int | None
    trigger_source_frame: int | None
    classification: GatedClassification
    head_triggered: bool
    forced_by_timeout: bool
    trigger_score: float | None
    remaining_source_frames: int | None
    late_delay_ticks: int | None
    late_delay_seconds: float | None
    switch_effective_rollout_tick: int | None
    trigger_logit: float | None = None

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        if value["trigger_logit"] is None:
            # Keep legacy history/current-only reports byte-for-byte shaped;
            # raw-prefix reports include this field when a logit was scored.
            value.pop("trigger_logit")
        return value


@dataclasses.dataclass(frozen=True)
class GatedTickDecision:
    """One 2 Hz gated replay decision and the state after scoring it."""

    rollout_tick: int
    rollout_time_seconds: float
    source_frame_index: int
    source_task_index: int
    active_task_index: int
    active_prompt: str
    target: int
    terminal_hold: bool
    terminal_hold_tick: int | None
    history_ready: bool
    score: float | None
    threshold: float
    triggered: bool
    timeout_forced: bool
    switch_reason: GatedSwitchReason | None
    task_before: int
    task_after: int
    done: bool
    logit: float | None = None

    def to_dict(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        if value["logit"] is None:
            value.pop("logit")
        return value


class GatedCompletionController:
    """Prompt-gated 2 Hz replay with terminal hold and timeout switching.

    This controller owns only replay state.  It never consults a predicted
    label or ground-truth boundary to trigger a switch; ``gt_end_frames`` is
    used solely to gate the source frame and to classify the resulting event.
    Early triggers use isolated recovery: the next task starts from its own
    standard playback frame, so an early switch cannot feed the next task an
    image under the wrong prompt.
    """

    def __init__(
        self,
        prompts: Sequence[str],
        *,
        playback_start_frames: Sequence[int],
        gt_end_frames: Sequence[int],
        threshold: float,
        mode: HistoryMode,
        timeout_seconds: float = 2.0,
    ) -> None:
        values = tuple(str(prompt) for prompt in prompts)
        if len(values) != TASK_COUNT or any(not prompt.strip() for prompt in values):
            raise ValueError("exactly four non-empty prompts are required")
        starts = tuple(_integer(value, name="playback_start_frame") for value in playback_start_frames)
        ends = tuple(_integer(value, name="gt_end_frame") for value in gt_end_frames)
        if len(starts) != TASK_COUNT or len(ends) != TASK_COUNT:
            raise ValueError("playback_start_frames and gt_end_frames must contain four values")
        if any(start < 0 or end < start for start, end in zip(starts, ends, strict=True)):
            raise ValueError("each playback start must be non-negative and no later than its gt end")
        maximum = float(np.nextafter(1.0, np.inf))
        if not np.isfinite(threshold) or not 0.0 <= float(threshold) <= maximum:
            raise ValueError("threshold must be in [0, nextafter(1,+inf)]")
        if mode not in ("history", "current_only", "transition", "raw_prefix_current", "raw_prefix_history"):
            raise ValueError(f"unsupported mode {mode!r}")
        seconds = float(timeout_seconds)
        units = seconds * 2.0
        if not np.isfinite(seconds) or seconds <= 0.0 or not np.isclose(units, round(units), rtol=0.0, atol=1.0e-8):
            raise ValueError("timeout_seconds must be positive and an integer multiple of 0.5 seconds")
        self.prompts = values
        self.playback_start_frames = starts
        self.gt_end_frames = ends
        self.threshold = float(threshold)
        self.mode = mode
        self.timeout_seconds = seconds
        self.timeout_ticks = round(units)
        self.reset()

    def reset(self) -> None:
        self._task_index = 0
        self._source_frame = self.playback_start_frames[0]
        self._history: list[Any] = []
        self._terminal_arrival_tick: int | None = None
        self._last_rollout_tick: int | None = None
        self._done = False
        self._done_rollout_tick: int | None = None
        self._task_results: list[GatedTaskResult] = []

    @property
    def current_task_index(self) -> int:
        return self._task_index

    @property
    def current_prompt(self) -> str:
        return self.prompts[self._task_index]

    @property
    def current_source_frame(self) -> int:
        return self._source_frame

    @property
    def done(self) -> bool:
        return self._done

    @property
    def done_rollout_tick(self) -> int | None:
        return self._done_rollout_tick

    @property
    def history_size(self) -> int:
        return len(self._history)

    @property
    def task_results(self) -> tuple[GatedTaskResult, ...]:
        return tuple(self._task_results)

    def _head_input(self, values: Any) -> tuple[Any | None, bool]:
        if self.mode == "raw_prefix_current":
            return _validate_raw_prefix_input(values), True
        if self.mode == "raw_prefix_history":
            current = _validate_raw_prefix_input(values)
            self._history.append(_copy_raw_prefix_input(current))
            if len(self._history) > 3:
                self._history.pop(0)
            history_ready = len(self._history) == 3
            return (tuple(self._history), True) if history_ready else (None, False)
        history_ready = self.mode == "current_only"
        if self.mode in ("history", "transition"):
            self._history.append(np.array(values, copy=True))
            if len(self._history) > 3:
                self._history.pop(0)
            history_ready = len(self._history) == 3
        if not history_ready:
            return None, False
        if self.mode == "current_only":
            return np.stack([np.zeros_like(values), np.zeros_like(values), values], axis=0), True
        return np.stack(self._history, axis=0).astype(np.float32, copy=False), True

    def _append_result(
        self,
        *,
        task: int,
        rollout_tick: int,
        source_frame: int,
        terminal_hold_tick: int | None,
        score: float | None,
        logit: float | None,
        classification: GatedClassification,
        switch_reason: GatedSwitchReason,
    ) -> GatedTaskResult:
        head_triggered = classification != "timeout_forced"
        forced = classification == "timeout_forced"
        terminal_arrival = self._terminal_arrival_tick
        trigger_tick = rollout_tick if head_triggered else None
        trigger_frame = source_frame if head_triggered else None
        remaining = self.gt_end_frames[task] - source_frame if head_triggered and classification == "early" else None
        late_ticks = None
        late_seconds = None
        if classification == "on_time":
            late_ticks = 0
            late_seconds = 0.0
        elif classification in ("late_trigger", "timeout_forced"):
            late_ticks = self.timeout_ticks if forced else int(terminal_hold_tick or 0)
            late_seconds = late_ticks / 2.0
        switch_tick = None if task == TASK_COUNT - 1 else rollout_tick + 1
        result = GatedTaskResult(
            task_index=task,
            playback_start_frame=self.playback_start_frames[task],
            gt_end_frame=self.gt_end_frames[task],
            terminal_arrival_rollout_tick=terminal_arrival,
            trigger_rollout_tick=trigger_tick,
            trigger_source_frame=trigger_frame,
            classification=classification,
            head_triggered=head_triggered,
            forced_by_timeout=forced,
            trigger_score=score,
            remaining_source_frames=remaining,
            late_delay_ticks=late_ticks,
            late_delay_seconds=late_seconds,
            switch_effective_rollout_tick=switch_tick,
            trigger_logit=logit,
        )
        self._task_results.append(result)
        return result

    def _switch_or_finish(self, task: int, rollout_tick: int) -> None:
        if task == TASK_COUNT - 1:
            self._done = True
            self._done_rollout_tick = rollout_tick
            return
        self._task_index = task + 1
        self._source_frame = self.playback_start_frames[self._task_index]
        self._terminal_arrival_tick = None
        if self.mode in ("history", "raw_prefix_history"):
            self._history.clear()

    def step(self, rollout_tick: int, feature: np.ndarray, score_fn: ScoreFunction) -> GatedTickDecision:
        tick = _integer(rollout_tick, name="rollout_tick")
        if tick < 0 or (self._last_rollout_tick is not None and tick != self._last_rollout_tick + 1):
            expected = 0 if self._last_rollout_tick is None else self._last_rollout_tick + 1
            raise ValueError(f"rollout ticks must be consecutive from zero: expected {expected}, got {tick}")
        if self._done:
            raise ValueError("cannot step a completed gated controller")
        self._last_rollout_tick = tick
        task = self._task_index
        source_frame = self._source_frame
        end_frame = self.gt_end_frames[task]
        prompt = self.current_prompt
        if not self.playback_start_frames[task] <= source_frame <= end_frame:
            raise ValueError("gated source frame escaped the active task playback range")
        terminal = source_frame == end_frame
        if terminal:
            if self._terminal_arrival_tick is None:
                self._terminal_arrival_tick = tick
            terminal_hold_tick = tick - self._terminal_arrival_tick
        else:
            if self._terminal_arrival_tick is not None:
                raise ValueError("non-terminal source frame observed after terminal hold began")
            terminal_hold_tick = None
        if self.mode in ("raw_prefix_current", "raw_prefix_history"):
            values = feature
            head_input, history_ready = self._head_input(values)
        else:
            values = np.asarray(feature, dtype=np.float32)
            if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"prefix feature must be a finite non-empty [D] vector, got {values.shape}")
            head_input, history_ready = self._head_input(values)
        score: float | None = None
        logit: float | None = None
        if head_input is not None:
            score, logit = _parse_score_output(score_fn(head_input))
        triggered = score is not None and score >= self.threshold
        classification: GatedClassification | None = None
        switch_reason: GatedSwitchReason | None = None
        timeout_forced = False
        if triggered:
            if source_frame < end_frame:
                classification = "early"
                switch_reason = "head_early"
            elif terminal_hold_tick == 0:
                classification = "on_time"
                switch_reason = "head_on_time"
            else:
                classification = "late_trigger"
                switch_reason = "head_late"
        elif terminal and terminal_hold_tick is not None and terminal_hold_tick >= self.timeout_ticks:
            classification = "timeout_forced"
            switch_reason = "timeout"
            timeout_forced = True
        if classification is not None:
            self._append_result(
                task=task,
                rollout_tick=tick,
                source_frame=source_frame,
                terminal_hold_tick=terminal_hold_tick,
                score=score,
                logit=logit,
                classification=classification,
                switch_reason=switch_reason,
            )
            self._switch_or_finish(task, tick)
        elif source_frame < end_frame:
            self._source_frame = min(source_frame + TICK_STRIDE_FRAMES, end_frame)
        task_after = self._task_index if classification is not None else task
        return GatedTickDecision(
            rollout_tick=tick,
            rollout_time_seconds=tick / 2.0,
            source_frame_index=source_frame,
            source_task_index=task,
            active_task_index=task,
            active_prompt=prompt,
            target=int(terminal),
            terminal_hold=terminal,
            terminal_hold_tick=terminal_hold_tick,
            history_ready=history_ready,
            score=score,
            threshold=self.threshold,
            triggered=bool(triggered),
            timeout_forced=timeout_forced,
            switch_reason=switch_reason,
            task_before=task,
            task_after=task_after,
            done=self._done,
            logit=logit,
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
