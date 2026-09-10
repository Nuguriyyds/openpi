"""Training-Paper RTC server with same-forward temporal completion scoring."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import serve_training_paper_rtc_base as _base  # noqa: E402
import serve_training_paper_rtc_policy as _rtc  # noqa: E402
from deploy_completion.completion_head_runtime import CompletionHeadRuntime


_DEFAULT_HEAD_DIR = "/home/geekplus/develop/ra_ttrtc/openpi/checkpoints/done_head_h768"
_TASK_PROMPT_ORDER = (
    "load_bread_into_toaster",
    "activate_toaster",
    "pour_drink_into_cup",
    "place_toasted_bread_on_plate",
)


@dataclasses.dataclass
class Args(_base.Args):
    completion_head_dir: str = _DEFAULT_HEAD_DIR
    completion_history_tolerance_s: float = 0.2
    completion_history_window_s: float = 2.0


@dataclasses.dataclass(frozen=True)
class PrefixEntry:
    observation_monotonic_s: float
    prefix_out: jax.Array
    prefix_mask: jax.Array


class TimestampedPrefixHistory:
    """Per-connection, per-prompt-generation raw-prefix history."""

    def __init__(self, head: CompletionHeadRuntime, *, tolerance_s: float, window_s: float):
        self._head = head
        self._tolerance_s = float(tolerance_s)
        self._window_s = float(window_s)
        self._generation: int | None = None
        self._entries: list[PrefixEntry] = []

    def clear(self) -> None:
        self._generation = None
        self._entries.clear()

    def append_and_score(
        self,
        *,
        generation: int,
        observation_monotonic_s: float,
        prefix_out: jax.Array,
        prefix_mask: jax.Array,
    ) -> dict[str, Any]:
        if self._generation != generation:
            self._generation = generation
            self._entries.clear()

        current = PrefixEntry(
            observation_monotonic_s=float(observation_monotonic_s),
            prefix_out=prefix_out,
            prefix_mask=prefix_mask,
        )
        self._entries.append(current)
        oldest_kept = current.observation_monotonic_s - self._window_s
        self._entries = [entry for entry in self._entries if entry.observation_monotonic_s >= oldest_kept]

        target_0 = current.observation_monotonic_s - 1.0
        target_1 = current.observation_monotonic_s - 0.5
        candidates = self._entries[:-1]
        best: tuple[float, PrefixEntry, PrefixEntry, float, float] | None = None
        for index, first in enumerate(candidates):
            for second in candidates[index + 1 :]:
                if not (
                    first.observation_monotonic_s
                    < second.observation_monotonic_s
                    < current.observation_monotonic_s
                ):
                    continue
                error_0 = abs(first.observation_monotonic_s - target_0)
                error_1 = abs(second.observation_monotonic_s - target_1)
                if error_0 > self._tolerance_s or error_1 > self._tolerance_s:
                    continue
                candidate = (error_0 + error_1, first, second, error_0, error_1)
                if best is None or candidate[0] < best[0]:
                    best = candidate

        base = {
            "prompt_generation": generation,
            "observation_monotonic_s": current.observation_monotonic_s,
            "history_size": len(self._entries),
            "history_ready": best is not None,
        }
        if best is None:
            near_t_minus_1 = any(
                abs(entry.observation_monotonic_s - target_0) <= self._tolerance_s for entry in candidates
            )
            near_t_minus_half = any(
                abs(entry.observation_monotonic_s - target_1) <= self._tolerance_s for entry in candidates
            )
            if not near_t_minus_1:
                reason = "missing_t_minus_1s"
            elif not near_t_minus_half:
                reason = "missing_t_minus_0_5s"
            else:
                reason = "no_ordered_distinct_triplet"
            return {
                **base,
                "score": None,
                "logit": None,
                "history_not_ready_reason": reason,
                "selected_observation_times": [],
                "relative_times": [],
                "target_time_errors": [],
                "head_score_ms": 0.0,
            }

        _, first, second, error_0, error_1 = best
        selected = (first, second, current)
        prefix_history = jnp.stack([entry.prefix_out for entry in selected], axis=0)
        prefix_mask = jnp.stack([entry.prefix_mask for entry in selected], axis=0)
        score, logit, head_ms = self._head.score_device(prefix_history, prefix_mask)
        selected_times = [entry.observation_monotonic_s for entry in selected]
        return {
            **base,
            "score": score,
            "logit": logit,
            "selected_observation_times": selected_times,
            "relative_times": [value - current.observation_monotonic_s for value in selected_times],
            "target_time_errors": [error_0, error_1, 0.0],
            "head_score_ms": head_ms,
        }


class IntegratedTrainingPaperRtcPolicy(_base.TrainingPaperRtcPolicy):
    def __init__(self, policy: Any, args: Args) -> None:
        super().__init__(policy, args)
        self._completion_head = CompletionHeadRuntime(args.completion_head_dir)
        self._completion_history = TimestampedPrefixHistory(
            self._completion_head,
            tolerance_s=args.completion_history_tolerance_s,
            window_s=args.completion_history_window_s,
        )
        dummy_history = jnp.zeros(
            (
                3,
                self._completion_head.metadata.token_count,
                self._completion_head.metadata.feature_dim,
            ),
            dtype=jnp.bfloat16,
        )
        dummy_mask = jnp.ones(dummy_history.shape[:2], dtype=jnp.bool_)
        _, _, warmup_ms = self._completion_head.score_device(dummy_history, dummy_mask)
        logging.info("Completion head loaded and warmed up in %.1f ms", warmup_ms)

    def reset(self) -> None:
        super().reset()
        self._completion_history.clear()

    @staticmethod
    def _strip_private_request_keys(obs: dict) -> dict:
        return {
            key: value
            for key, value in obs.items()
            if not str(key).startswith(("_paper_rtc_", "_training_paper_rtc_", "_completion_"))
        }

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        raw = self._completion_head.raw_metadata
        cache = raw.get("cache_metadata") if isinstance(raw.get("cache_metadata"), dict) else {}
        prompt_map = cache.get("task_prompts") if isinstance(cache.get("task_prompts"), dict) else {}
        prompts = [prompt_map[key] for key in _TASK_PROMPT_ORDER]
        metadata["completion"] = {
            **self._completion_head.metadata.as_dict(),
            "task_prompts": prompts,
            "history_tolerance_s": self._args.completion_history_tolerance_s,
            "history_window_s": self._args.completion_history_window_s,
            "source": "same_action_prefix_forward",
        }
        return metadata

    def _infer_client_chunk(
        self,
        clean_obs: dict,
        *,
        action_prefix_model: Any,
        prefix_steps: int,
        request_obs: dict,
    ) -> dict[str, Any]:
        result = self._sampler.infer_chunk_with_raw_prefix(
            clean_obs,
            action_prefix_model=action_prefix_model,
            prefix_steps=prefix_steps,
        )
        prefix_out = result.pop("raw_prefix_out")
        prefix_mask = result.pop("raw_prefix_mask")
        generation = int(request_obs.get("_completion_prompt_generation", 0))
        task_index = int(request_obs.get("_completion_task_index", 0))
        request_sequence = int(request_obs.get("_completion_request_sequence", -1))
        observation_time = float(request_obs.get("_completion_observation_monotonic_s", time.monotonic()))

        if bool(request_obs.get("_completion_skip_history", False)):
            completion = {
                "score": None,
                "logit": None,
                "history_ready": False,
                "history_size": 0,
                "history_not_ready_reason": "warmup_not_recorded",
                "prompt_generation": generation,
                "observation_monotonic_s": observation_time,
                "selected_observation_times": [],
                "relative_times": [],
                "target_time_errors": [],
                "head_score_ms": 0.0,
            }
        else:
            completion = self._completion_history.append_and_score(
                generation=generation,
                observation_monotonic_s=observation_time,
                prefix_out=prefix_out,
                prefix_mask=prefix_mask,
            )
        completion["task_index"] = task_index
        completion["request_sequence"] = request_sequence
        result["completion"] = completion
        return result


def main(args: Args) -> None:
    if args.completion_history_tolerance_s <= 0.0:
        raise ValueError("completion_history_tolerance_s must be positive")
    if args.completion_history_window_s < 1.0 + args.completion_history_tolerance_s:
        raise ValueError("completion_history_window_s is too short for the t-1.0s target")
    _rtc._install_training_paper_rtc_overrides()
    _base.TrainingPaperRtcPolicy = IntegratedTrainingPaperRtcPolicy
    jax.config.update("jax_default_matmul_precision", "float32")
    _base.main(args)


if __name__ == "__main__":
    _base.logging.basicConfig(level=_base.logging.INFO, force=True)
    main(_base.tyro.cli(Args))
