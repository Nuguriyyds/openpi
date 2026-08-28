"""Evaluate a completion head on complete 30 fps trajectories.

This is an independent, inference-only evaluator.  It keeps the controller's
prompt and prefix history causal while using the subtask manifest only for
episode identity, lengths, and reporting reference boundaries.  The model is
run only on global 2 Hz ticks; no action or flow-matching sampling is done.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from openpi.training import temporal_completion_data as temporal_data
from openpi.training.temporal_completion_semiclosed import GatedCompletionController
from openpi.training.temporal_completion_semiclosed import classify_boundary
from openpi.training.temporal_completion_semiclosed import reference_ticks

DEFAULT_MANIFEST = Path("/mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v4.json")
DEFAULT_FULL_DATASET_ROOT = Path("/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_730")
DEFAULT_SUBTASK_DATASET_ROOT = Path("/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730")
REPORT_SCHEMA_VERSION = 2
EVALUATION_PROTOCOL = "gated_terminal_hold_stall_v2"


def validate_max_terminal_hold_seconds(value: float) -> tuple[float, int]:
    seconds = float(value)
    units = seconds * 2.0
    if not np.isfinite(seconds) or seconds <= 0.0 or not np.isclose(units, round(units), rtol=0.0, atol=1.0e-8):
        raise ValueError("--max-terminal-hold-seconds must be positive and an integer multiple of 0.5 seconds")
    return seconds, round(units)


def load_threshold(
    *, validation_report: Path | None, explicit_threshold: float | None
) -> tuple[float, str, str | None]:
    """Resolves exactly one validation-selected or explicitly supplied threshold."""

    if (validation_report is None) == (explicit_threshold is None):
        raise ValueError("provide exactly one of --validation-report or --threshold")
    source = "validation_report" if validation_report is not None else "explicit_cli"
    report_path: str | None = None
    if validation_report is not None:
        value = json.loads(validation_report.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not isinstance(value.get("threshold_selection"), dict):
            raise ValueError("validation report must contain threshold_selection")
        selection = value["threshold_selection"]
        if "threshold" not in selection:
            raise ValueError("validation report threshold_selection lacks threshold")
        threshold = float(selection["threshold"])
        report_path = str(validation_report.resolve())
    else:
        threshold = float(explicit_threshold)
    maximum = float(np.nextafter(1.0, np.inf))
    if not math.isfinite(threshold) or not 0.0 <= threshold <= maximum:
        raise ValueError("completion threshold must be finite and in [0, nextafter(1,+inf)]")
    return threshold, source, report_path


def _scalar_int(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _episode_bounds(dataset: Any, episode_id: int) -> tuple[int, int]:
    episode_from = _scalar_int(dataset.episode_data_index["from"][episode_id])
    episode_to = _scalar_int(dataset.episode_data_index["to"][episode_id])
    if episode_to <= episode_from:
        raise ValueError(f"dataset episode {episode_id} is empty")
    return episode_from, episode_to


def _resolve_logical_prompts(tasks: Any) -> tuple[str, str, str, str]:
    prompts: list[str] = []
    for task_index in range(4):
        key: Any = task_index if task_index in tasks else str(task_index)
        if key not in tasks:
            raise ValueError(f"subtask metadata has no prompt for logical task {task_index}")
        prompt = tasks[key]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"subtask metadata prompt {task_index} is not a non-empty string")
        prompts.append(prompt)
    return tuple(prompts)  # type: ignore[return-value]


def _load_manifest(path: Path) -> temporal_data.TemporalCompletionManifest:
    manifest = temporal_data.load_temporal_manifest(path)
    if manifest.trajectory_source not in ("full_identity", "subtask_logical"):
        raise ValueError(f"unsupported manifest trajectory_source: {manifest.trajectory_source!r}")
    if manifest.trajectory_source == "full_identity" and (
        manifest.source_full_root is None or manifest.source_full_repo_id is None
    ):
        raise ValueError("full_identity manifest does not bind a full dataset")
    return manifest


def _evaluation_repack() -> Any:
    # Reuse the extraction script's exact repack transform.  The import stays
    # local so pure controller/threshold tests do not initialize model stacks.
    try:
        from scripts.extract_temporal_completion_features import _evaluation_repack
    except ModuleNotFoundError:
        from extract_temporal_completion_features import _evaluation_repack

    return _evaluation_repack()


@dataclasses.dataclass(frozen=True)
class _EpisodeSpec:
    group_id: int
    full_episode_id: int
    subtask_episode_ids: tuple[int, int, int, int]
    lengths: tuple[int, int, int, int]
    subtask_start_frames: tuple[int, int, int, int]
    gt_end_frames: tuple[int, int, int, int]
    full_length: int
    prompts: tuple[str, str, str, str]


def _numeric_episode(
    root: Path,
    episode_id: int,
    *,
    layout_cache: dict[Path, tuple[str, int]],
    episode_cache: dict[tuple[Path, int], np.ndarray],
) -> np.ndarray:
    """Loads only numeric state/action columns for sequence alignment."""

    key = (root.resolve(), int(episode_id))
    if key in episode_cache:
        return episode_cache[key]
    import pyarrow.parquet as parquet

    root_key = root.resolve()
    if root_key not in layout_cache:
        info = json.loads((root_key / "meta/info.json").read_text(encoding="utf-8"))
        layout_cache[root_key] = (str(info["data_path"]), int(info.get("chunks_size", 1000)))
    template, chunk_size = layout_cache[root_key]
    parquet_path = root_key / template.format(
        episode_chunk=int(episode_id) // chunk_size,
        episode_index=int(episode_id),
    )
    if not parquet_path.is_file():
        raise FileNotFoundError(f"episode parquet not found for alignment: {parquet_path}")
    table = parquet.read_table(
        parquet_path,
        # Pass a list rather than a tuple: newer PyArrow releases validate
        # the ``columns`` argument strictly and reject tuples.
        columns=["observation.state.joint", "observation.gripper_position", "actions"],
    )
    columns = [
        np.asarray(table[name].combine_chunks().to_pylist(), dtype=np.float32)
        for name in ("observation.state.joint", "observation.gripper_position", "actions")
    ]
    values = np.concatenate(columns, axis=1)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError(f"numeric alignment data is invalid: {parquet_path}")
    episode_cache[key] = values
    return values


def _align_subtasks_to_full(
    *,
    full_root: Path,
    full_episode_id: int,
    subtask_root: Path,
    subtask_episode_ids: tuple[int, int, int, int],
    layout_cache: dict[Path, tuple[str, int]],
    episode_cache: dict[tuple[Path, int], np.ndarray],
    tolerance: float = 1.0e-5,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Finds exact contiguous subtask segments inside one full trajectory."""

    full = _numeric_episode(
        full_root,
        full_episode_id,
        layout_cache=layout_cache,
        episode_cache=episode_cache,
    )
    subtasks = tuple(
        _numeric_episode(
            subtask_root,
            episode_id,
            layout_cache=layout_cache,
            episode_cache=episode_cache,
        )
        for episode_id in subtask_episode_ids
    )
    first = subtasks[0][0]
    candidate_starts = np.flatnonzero(np.max(np.abs(full - first), axis=1) <= tolerance)
    solutions: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for candidate in candidate_starts.tolist():
        starts: list[int] = []
        ends: list[int] = []
        cursor = int(candidate)
        valid = True
        for subtask in subtasks:
            stop = cursor + len(subtask)
            if stop > len(full) or not np.allclose(full[cursor:stop], subtask, rtol=0.0, atol=tolerance):
                valid = False
                break
            starts.append(cursor)
            ends.append(stop - 1)
            cursor = stop
        if valid:
            solutions.append((tuple(starts), tuple(ends)))  # type: ignore[arg-type]
    if len(solutions) != 1:
        raise ValueError(
            f"could not find a unique contiguous alignment for full episode {full_episode_id} "
            f"and subtask episodes {subtask_episode_ids}: solutions={len(solutions)}"
        )
    return solutions[0]


def _episode_specs(
    manifest: temporal_data.TemporalCompletionManifest,
    *,
    split: str,
    subtask_dataset: Any,
    full_dataset: Any,
    subtask_root: Path,
    full_root: Path,
) -> tuple[_EpisodeSpec, ...]:
    if split != "test":
        raise ValueError("semi-closed evaluation is locked to the manifest test split")
    specs: list[_EpisodeSpec] = []
    layout_cache: dict[Path, tuple[str, int]] = {}
    episode_cache: dict[tuple[Path, int], np.ndarray] = {}
    used_full_ids: set[int] = set()
    for record in manifest.trajectories:
        if record.split != split:
            continue
        if record.group_id is None or record.mapping_status not in ("matched", "subtask_only"):
            continue
        group_id = int(record.group_id)
        if manifest.trajectory_source == "full_identity":
            if record.full_episode_id is None:
                raise ValueError(f"matched group {group_id} has no full_episode_id")
            full_episode_id = int(record.full_episode_id)
        else:
            # The logical training manifest has no full-trajectory identity
            # field.  Start with the nominal group id and resolve the one-off
            # full-dataset insertion by exact numeric sequence alignment.
            full_episode_id = group_id
        subtask_ids = tuple(4 * group_id + task for task in range(4))
        lengths: list[int] = []
        for episode_id in subtask_ids:
            start, stop = _episode_bounds(subtask_dataset, episode_id)
            lengths.append(stop - start)
        if manifest.trajectory_source == "subtask_logical":
            candidate_ids = [group_id]
            for delta in range(1, 9):
                candidate_ids.extend((group_id + delta, group_id - delta))
            candidate_ids = [
                candidate
                for candidate in candidate_ids
                if candidate >= 0 and candidate < len(full_dataset.episode_data_index["from"])
            ]
            aligned: list[tuple[int, tuple[int, int, int, int], tuple[int, int, int, int]]] = []
            for candidate in candidate_ids:
                if candidate in used_full_ids:
                    continue
                try:
                    starts, ends = _align_subtasks_to_full(
                        full_root=full_root,
                        full_episode_id=candidate,
                        subtask_root=subtask_root,
                        subtask_episode_ids=subtask_ids,  # type: ignore[arg-type]
                        layout_cache=layout_cache,
                        episode_cache=episode_cache,
                    )
                except (FileNotFoundError, ValueError):
                    continue
                aligned.append((candidate, starts, ends))
            if len(aligned) != 1:
                raise ValueError(
                    f"could not resolve full episode for logical group {group_id}; "
                    f"candidate alignments={[(item[0], item[2][-1]) for item in aligned]}"
                )
            full_episode_id, starts, ends = aligned[0]
        else:
            starts, ends = _align_subtasks_to_full(
                full_root=full_root,
                full_episode_id=full_episode_id,
                subtask_root=subtask_root,
                subtask_episode_ids=subtask_ids,  # type: ignore[arg-type]
                layout_cache=layout_cache,
                episode_cache=episode_cache,
            )
        used_full_ids.add(full_episode_id)
        full_start, full_stop = _episode_bounds(full_dataset, full_episode_id)
        full_length = full_stop - full_start
        if full_length < sum(lengths) or ends[-1] >= full_length:
            raise ValueError(f"aligned subtask segments exceed full episode {full_episode_id}")
        specs.append(
            _EpisodeSpec(
                group_id=group_id,
                full_episode_id=full_episode_id,
                subtask_episode_ids=subtask_ids,  # type: ignore[arg-type]
                lengths=tuple(lengths),  # type: ignore[arg-type]
                subtask_start_frames=starts,
                gt_end_frames=ends,
                full_length=full_length,
                prompts=manifest.task_prompts,
            )
        )
    if not specs:
        raise ValueError(f"manifest contains no matched {split!r} groups")
    return tuple(sorted(specs, key=lambda item: item.full_episode_id))


def _prepare_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Loads the requested head and returns policy, dataset model, and JAX prefix fn."""

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    import openpi.models.model as model_api
    from openpi.policies import policy_config
    from openpi.training import config as training_config

    config = training_config.get_config(args.config_name)
    if not bool(getattr(config.completion, "uses_temporal_completion", False)):
        raise ValueError(f"config {args.config_name!r} is not a temporal completion-head config")
    configured_mode = str(getattr(config.completion, "temporal_input_mode", "history"))
    configured_protocol = str(getattr(config.completion, "temporal_sampling_protocol", "subtask_local"))
    if args.mode == "transition":
        # Transition evaluation uses the same [B, 3, D] history head as the
        # history-carry training ablation.  ``transition`` is an evaluator
        # state-machine mode, not a separate model input mode.
        if configured_mode != "history" or configured_protocol != "history_carry":
            raise ValueError(
                "transition mode requires a history-carry config "
                "(temporal_input_mode='history', temporal_sampling_protocol='history_carry')"
            )
    elif configured_mode != args.mode:
        raise ValueError(f"config temporal_input_mode={configured_mode!r} does not match --mode={args.mode!r}")
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"checkpoint params not found: {checkpoint / 'params'}")
    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        completion_head_params=getattr(args, "done_params", None),
        repack_transforms=_evaluation_repack(),
        sample_kwargs={},
    )
    if bool(getattr(policy, "_is_pytorch_model", False)):
        raise ValueError("semi-closed evaluator requires a JAX Pi0.5 checkpoint")
    model = policy._model  # noqa: SLF001
    token_mode = getattr(model, "completion_head_variant", None) == "token_query_attention"
    method_name = "compute_prefix_tokens" if token_mode else "compute_prefix_feature"
    if not hasattr(model, method_name):
        raise ValueError(f"loaded model lacks {method_name}")
    graphdef, state = nnx.split(model)

    def compute_prefix(state_value: Any, observation: Any) -> Any:
        module = nnx.merge(graphdef, state_value)
        method = getattr(module, method_name)
        return method(jax.random.key(0), observation, train=False)

    compute_fn = jax.jit(compute_prefix)
    return policy, model_api, jax, jnp, compute_fn, state


def _prefix_feature(
    *,
    policy: Any,
    model_api: Any,
    jax: Any,
    jnp: Any,
    compute_fn: Any,
    state: Any,
    dataset: Any,
    episode_id: int,
    frame_index: int,
    prompt: str,
) -> np.ndarray:
    start, stop = _episode_bounds(dataset, episode_id)
    if frame_index < 0 or start + frame_index >= stop:
        raise ValueError(f"full episode {episode_id} frame {frame_index} is out of range")
    sample = dict(dataset[start + frame_index])
    if _scalar_int(sample["episode_index"]) != episode_id or _scalar_int(sample["frame_index"]) != frame_index:
        raise ValueError("full dataset episode/frame metadata disagrees with the requested coordinate")
    # Prompt override is deliberately before repack, prompt injection, image,
    # normalization, and tokenization transforms.
    sample["prompt"] = prompt
    transformed = policy._input_transform(sample)  # noqa: SLF001
    batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
    observation = model_api.Observation.from_dict(batched)
    computed = jax.block_until_ready(compute_fn(state, observation))
    if isinstance(computed, tuple):
        tokens, mask = computed
        tokens = np.asarray(tokens, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.bool_)
        if tokens.ndim != 3 or tokens.shape[0] != 1 or tokens.shape[-1] != int(policy._model.prefix_feature_dim):  # noqa: SLF001
            raise ValueError(f"compute_prefix_tokens returned unexpected shape {tokens.shape}")
        if mask.shape != tokens.shape[:2]:
            raise ValueError(f"compute_prefix_tokens returned unexpected mask shape {mask.shape}")
        if not np.isfinite(tokens).all():
            raise ValueError("compute_prefix_tokens returned non-finite values")
        return np.concatenate((tokens[0], mask[0, :, None].astype(np.float32)), axis=-1)
    feature = np.asarray(computed, dtype=np.float32)
    if feature.ndim != 2 or feature.shape[0] != 1 or feature.shape[1] != int(policy._model.prefix_feature_dim):  # noqa: SLF001
        raise ValueError(f"compute_prefix_feature returned unexpected shape {feature.shape}")
    if not np.isfinite(feature).all():
        raise ValueError("compute_prefix_feature returned non-finite values")
    return feature[0]


def _evaluate_episode(
    spec: _EpisodeSpec,
    *,
    args: argparse.Namespace,
    policy: Any,
    model_api: Any,
    jax: Any,
    jnp: Any,
    compute_fn: Any,
    state: Any,
    full_dataset: Any,
    threshold: float,
) -> dict[str, Any]:
    ends = spec.gt_end_frames
    refs = reference_ticks(ends)
    playback_starts = (0, *spec.subtask_start_frames[1:])
    controller = GatedCompletionController(
        spec.prompts,
        playback_start_frames=playback_starts,
        gt_end_frames=ends,
        threshold=threshold,
        mode=args.mode,
        max_terminal_hold_seconds=args.max_terminal_hold_seconds,
    )
    ticks: list[dict[str, Any]] = []
    rollout_tick = 0
    while not controller.terminated:
        source_frame = controller.current_source_frame
        prompt = controller.current_prompt
        feature = _prefix_feature(
            policy=policy,
            model_api=model_api,
            jax=jax,
            jnp=jnp,
            compute_fn=compute_fn,
            state=state,
            dataset=full_dataset,
            episode_id=spec.full_episode_id,
            frame_index=source_frame,
            prompt=prompt,
        )
        decision = controller.step(rollout_tick, feature, policy.score_temporal_completion)
        if decision.active_task_index != decision.source_task_index:
            raise RuntimeError(
                "gated replay invariant violated: active_task_index and source_task_index differ "
                f"at rollout tick {rollout_tick}"
            )
        tick = decision.to_dict()
        tick["prompt_mismatch"] = False
        tick["wrong_prompt"] = False
        ticks.append(tick)
        rollout_tick += 1

    task_results = list(controller.task_results)
    if controller.done and len(task_results) != 4:
        raise RuntimeError(f"completed gated replay has {len(task_results)} task results instead of four")
    if controller.stalled and (not task_results or task_results[-1].classification != "missed"):
        raise RuntimeError("stalled gated replay must end with a missed task result")
    predicted: list[int | None] = [None] * 4
    for result in task_results:
        predicted[result.task_index] = result.trigger_source_frame
    boundaries = [
        classify_boundary(task, refs[task], predicted[task], full_length=spec.full_length) for task in range(4)
    ]
    boundary_dicts = [result.to_dict() for result in boundaries]
    task_dicts = [result.to_dict() for result in task_results]
    classifications = [result.classification for result in task_results]
    first_non_on_time = next((task for task, value in enumerate(classifications) if value != "on_time"), None)
    early_count = classifications.count("early")
    on_time_count = classifications.count("on_time")
    late_count = classifications.count("late_trigger")
    missed_count = classifications.count("missed")
    not_reached_count = 4 - len(task_results)
    mismatch_count = sum(int(bool(tick["prompt_mismatch"])) for tick in ticks)
    warmup_count = sum(int(not bool(tick["history_ready"])) for tick in ticks)
    return {
        "full_episode_id": spec.full_episode_id,
        "group_id": spec.group_id,
        "full_length": spec.full_length,
        "subtask_episode_ids": list(spec.subtask_episode_ids),
        "lengths": list(spec.lengths),
        "subtask_start_frames": list(spec.subtask_start_frames),
        "playback_start_frames": list(playback_starts),
        "gt_end_frames": list(ends),
        "reference_ticks": list(refs),
        "reference_tick_available": [ref < spec.full_length for ref in refs],
        "predicted_ticks": predicted,
        "boundary_results": boundary_dicts,
        "task_results": task_dicts,
        "all_correct": all(result.classification in ("correct", "unavailable") for result in boundaries),
        "all_on_time": on_time_count == 4,
        "fully_autonomous": controller.done,
        "first_failure_task": first_non_on_time,
        "first_non_on_time_task": first_non_on_time,
        "early_count": early_count,
        "on_time_count": on_time_count,
        "late_trigger_count": late_count,
        "missed_count": missed_count,
        "not_reached_count": not_reached_count,
        "late_count": late_count,
        "done": controller.done,
        "done_frame": ticks[-1]["source_frame_index"] if ticks else None,
        "done_rollout_tick": controller.done_rollout_tick,
        "has_early_switch": early_count > 0,
        "has_late_trigger": late_count > 0,
        "has_missed": missed_count > 0,
        "stalled": controller.stalled,
        "max_terminal_hold_seconds": args.max_terminal_hold_seconds,
        "max_terminal_hold_ticks": controller.max_terminal_hold_ticks,
        "prompt_mismatch_count": mismatch_count,
        "prompt_mismatch_rate": mismatch_count / len(ticks) if ticks else None,
        "wrong_prompt_tick_count": mismatch_count,
        "history_not_ready_count": warmup_count,
        "history_not_ready_rate": warmup_count / len(ticks) if ticks else None,
        "ticks": ticks,
    }


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
    }


def _gated_summary(results: list[dict[str, Any]], *, planned_count: int | None = None) -> dict[str, Any]:
    counts = dict.fromkeys(("early", "on_time", "late_trigger", "missed"), 0)
    for result in results:
        counts[str(result["classification"])] += 1
    evaluated_count = len(results)
    task_count = evaluated_count if planned_count is None else planned_count
    if task_count < evaluated_count:
        raise ValueError("planned task count cannot be smaller than evaluated task count")
    not_reached_count = task_count - evaluated_count
    autonomous = counts["early"] + counts["on_time"] + counts["late_trigger"]
    early_remaining = [
        float(result["remaining_source_frames"])
        for result in results
        if result["classification"] == "early" and result["remaining_source_frames"] is not None
    ]
    late_delays = [
        float(result["late_delay_seconds"])
        for result in results
        if result["classification"] == "late_trigger" and result["late_delay_seconds"] is not None
    ]
    remaining_stats = _distribution(early_remaining)
    late_stats = _distribution(late_delays)
    return {
        "task_count": task_count,
        "early_count": counts["early"],
        "on_time_count": counts["on_time"],
        "late_trigger_count": counts["late_trigger"],
        "missed_count": counts["missed"],
        "not_reached_count": not_reached_count,
        "evaluated_task_count": evaluated_count,
        "early_rate": counts["early"] / task_count if task_count else None,
        "on_time_rate": counts["on_time"] / task_count if task_count else None,
        "late_trigger_rate": counts["late_trigger"] / task_count if task_count else None,
        "missed_rate": counts["missed"] / task_count if task_count else None,
        "not_reached_rate": not_reached_count / task_count if task_count else None,
        "autonomous_switch_count": autonomous,
        "autonomous_switch_rate": autonomous / task_count if task_count else None,
        "remaining_source_frames": remaining_stats,
        "remaining_source_frames_mean": remaining_stats["mean"],
        "remaining_source_frames_median": remaining_stats["median"],
        "remaining_source_frames_p90": remaining_stats["p90"],
        "remaining_source_frames_max": remaining_stats["max"],
        "late_delay_seconds": late_stats,
        "late_delay_seconds_mean": late_stats["mean"],
        "late_delay_seconds_median": late_stats["median"],
        "late_delay_seconds_p90": late_stats["p90"],
        "late_delay_seconds_max": late_stats["max"],
    }


def _summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    task_results = [result for episode in episodes for result in episode["task_results"]]
    episode_count = len(episodes)
    per_task = {
        str(task): _gated_summary(
            [result for result in task_results if int(result["task_index"]) == task],
            planned_count=episode_count,
        )
        for task in range(4)
    }
    all_ticks = sum(len(item["ticks"]) for item in episodes)
    mismatches = sum(int(item["wrong_prompt_tick_count"]) for item in episodes)
    warmups = sum(int(item["history_not_ready_count"]) for item in episodes)
    all_on_time_ids = sorted(int(item["full_episode_id"]) for item in episodes if item["all_on_time"])
    early_ids = sorted(int(item["full_episode_id"]) for item in episodes if item["has_early_switch"])
    late_ids = sorted(int(item["full_episode_id"]) for item in episodes if item["has_late_trigger"])
    stalled_ids = sorted(int(item["full_episode_id"]) for item in episodes if item["stalled"])
    not_autonomous_ids = sorted(int(item["full_episode_id"]) for item in episodes if not item["fully_autonomous"])
    fully_autonomous_count = sum(bool(item["fully_autonomous"]) for item in episodes)
    all_on_time_count = len(all_on_time_ids)
    overall = _gated_summary(task_results, planned_count=episode_count * 4)
    return {
        "episode_count": episode_count,
        "fully_autonomous_episode_count": fully_autonomous_count,
        "fully_autonomous_episode_rate": fully_autonomous_count / episode_count if episode_count else None,
        "all_on_time_episode_count": all_on_time_count,
        "all_on_time_episode_rate": all_on_time_count / episode_count if episode_count else None,
        "episode_with_early_count": len(early_ids),
        "episode_with_late_trigger_count": len(late_ids),
        "stalled_episode_count": len(stalled_ids),
        "done_count": sum(bool(item["done"]) for item in episodes),
        "done_rate": sum(bool(item["done"]) for item in episodes) / episode_count if episode_count else None,
        "prompt_mismatch_count": mismatches,
        "prompt_mismatch_rate": mismatches / all_ticks if all_ticks else None,
        "wrong_prompt_tick_count": mismatches,
        "history_not_ready_count": warmups,
        "history_not_ready_rate": warmups / all_ticks if all_ticks else None,
        "task_count": overall["task_count"],
        "early_count": overall["early_count"],
        "on_time_count": overall["on_time_count"],
        "late_trigger_count": overall["late_trigger_count"],
        "missed_count": overall["missed_count"],
        "not_reached_count": overall["not_reached_count"],
        "evaluated_task_count": overall["evaluated_task_count"],
        "early_rate": overall["early_rate"],
        "on_time_rate": overall["on_time_rate"],
        "late_trigger_rate": overall["late_trigger_rate"],
        "missed_rate": overall["missed_rate"],
        "not_reached_rate": overall["not_reached_rate"],
        "autonomous_switch_count": overall["autonomous_switch_count"],
        "autonomous_switch_rate": overall["autonomous_switch_rate"],
        "remaining_source_frames_mean": overall["remaining_source_frames_mean"],
        "remaining_source_frames_median": overall["remaining_source_frames_median"],
        "remaining_source_frames_p90": overall["remaining_source_frames_p90"],
        "remaining_source_frames_max": overall["remaining_source_frames_max"],
        "late_delay_seconds_mean": overall["late_delay_seconds_mean"],
        "late_delay_seconds_median": overall["late_delay_seconds_median"],
        "late_delay_seconds_p90": overall["late_delay_seconds_p90"],
        "late_delay_seconds_max": overall["late_delay_seconds_max"],
        "overall": overall,
        "per_task": per_task,
        "all_on_time_episode_ids": all_on_time_ids,
        "early_episode_ids": early_ids,
        "late_trigger_episode_ids": late_ids,
        "stalled_episode_ids": stalled_ids,
        "not_fully_autonomous_episode_ids": not_autonomous_ids,
        # Compatibility aliases retained for old report consumers; the new
        # classification fields above are the primary metrics.
        "incorrect_episode_ids": not_autonomous_ids,
        "late_episode_ids": late_ids,
        "missed_episode_ids": stalled_ids,
    }


def evaluate(args: argparse.Namespace) -> Path:
    args.max_terminal_hold_seconds, max_terminal_hold_ticks = validate_max_terminal_hold_seconds(
        args.max_terminal_hold_seconds
    )
    threshold, threshold_source, validation_path = load_threshold(
        validation_report=args.validation_report,
        explicit_threshold=args.threshold,
    )
    manifest = _load_manifest(args.manifest.resolve())
    if (
        manifest.source_full_root is not None
        and Path(manifest.source_full_root).resolve() != args.full_dataset_root.resolve()
    ):
        raise ValueError("--full-dataset-root does not match manifest source_full_root")
    if Path(manifest.source_subtask_root).resolve() != args.subtask_dataset_root.resolve():
        raise ValueError("--subtask-dataset-root does not match manifest source_subtask_root")

    os.environ.setdefault("HF_LEROBOT_HOME", str(args.full_dataset_root.resolve().parents[1]))
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    subtask_dataset = lerobot_dataset.LeRobotDataset(manifest.source_subtask_repo_id, root=args.subtask_dataset_root)
    full_repo_id = manifest.source_full_repo_id or "modanqing/agilex_make_breakfast_730"
    full_dataset = lerobot_dataset.LeRobotDataset(full_repo_id, root=args.full_dataset_root)
    metadata = lerobot_dataset.LeRobotDatasetMetadata(manifest.source_subtask_repo_id, root=args.subtask_dataset_root)
    ordered_prompts = _resolve_logical_prompts(metadata.tasks)
    if ordered_prompts != manifest.task_prompts:
        raise ValueError("subtask metadata prompts do not match manifest task_prompts")
    specs = _episode_specs(
        manifest,
        split=args.split,
        subtask_dataset=subtask_dataset,
        full_dataset=full_dataset,
        subtask_root=args.subtask_dataset_root,
        full_root=args.full_dataset_root,
    )
    policy, model_api, jax, jnp, compute_fn, state = _prepare_model(args)
    episodes = [
        _evaluate_episode(
            spec,
            args=args,
            policy=policy,
            model_api=model_api,
            jax=jax,
            jnp=jnp,
            compute_fn=compute_fn,
            state=state,
            full_dataset=full_dataset,
            threshold=threshold,
        )
        for spec in specs
    ]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": args.mode,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "max_terminal_hold_seconds": args.max_terminal_hold_seconds,
        "max_terminal_hold_ticks": max_terminal_hold_ticks,
        "scheduler_hz": 2,
        "fps": 30,
        "prompt_mismatch_note": "prompt mismatch is prevented by the gated replay protocol",
        "full_dataset_root": str(args.full_dataset_root.resolve()),
        "subtask_dataset_root": str(args.subtask_dataset_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "config_name": args.config_name,
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_report": validation_path,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "fps30": 30,
        "stride15": 15,
        "split": args.split,
        "summary": _summary(episodes),
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return args.output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("history", "current_only", "transition"), required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--done-params", type=Path)
    parser.add_argument("--full-dataset-root", type=Path, default=DEFAULT_FULL_DATASET_ROOT)
    parser.add_argument("--subtask-dataset-root", type=Path, default=DEFAULT_SUBTASK_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-terminal-hold-seconds", type=float, default=2.0)
    threshold_group = parser.add_mutually_exclusive_group(required=True)
    threshold_group.add_argument("--validation-report", type=Path)
    threshold_group.add_argument("--threshold", type=float)
    return parser


def main() -> None:
    output = evaluate(_parser().parse_args())
    print(f"Wrote semi-closed temporal completion report: {output}")


if __name__ == "__main__":
    main()
