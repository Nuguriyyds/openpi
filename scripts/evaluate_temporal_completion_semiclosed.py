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
from openpi.training.temporal_completion_semiclosed import BoundaryResult
from openpi.training.temporal_completion_semiclosed import SemiClosedCompletionController
from openpi.training.temporal_completion_semiclosed import classify_boundary
from openpi.training.temporal_completion_semiclosed import gt_end_frames
from openpi.training.temporal_completion_semiclosed import oracle_task_index
from openpi.training.temporal_completion_semiclosed import reference_ticks
from openpi.training.temporal_completion_semiclosed import summarize_boundary_results

DEFAULT_MANIFEST = Path("/mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v4.json")
DEFAULT_FULL_DATASET_ROOT = Path("/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_730")
DEFAULT_SUBTASK_DATASET_ROOT = Path("/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730")
REPORT_SCHEMA_VERSION = 1


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
        from scripts.extract_temporal_completion_features import _evaluation_repack  # noqa: PLC0415
    except ModuleNotFoundError:
        from extract_temporal_completion_features import _evaluation_repack  # noqa: PLC0415

    return _evaluation_repack()


@dataclasses.dataclass(frozen=True)
class _EpisodeSpec:
    group_id: int
    full_episode_id: int
    subtask_episode_ids: tuple[int, int, int, int]
    lengths: tuple[int, int, int, int]
    full_length: int
    prompts: tuple[str, str, str, str]


def _episode_specs(
    manifest: temporal_data.TemporalCompletionManifest,
    *,
    split: str,
    subtask_dataset: Any,
    full_dataset: Any,
) -> tuple[_EpisodeSpec, ...]:
    if split != "test":
        raise ValueError("semi-closed evaluation is locked to the manifest test split")
    specs: list[_EpisodeSpec] = []
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
            # field.  This evaluation protocol deliberately uses the user's
            # fixed dataset convention: full episode id equals group id.
            full_episode_id = group_id
        if full_episode_id != group_id:
            raise ValueError(
                "semi-closed fixed mapping requires full_episode_id == group_id; "
                f"got group={group_id}, full={full_episode_id}"
            )
        subtask_ids = tuple(4 * group_id + task for task in range(4))
        lengths: list[int] = []
        for episode_id in subtask_ids:
            start, stop = _episode_bounds(subtask_dataset, episode_id)
            lengths.append(stop - start)
        full_start, full_stop = _episode_bounds(full_dataset, full_episode_id)
        full_length = full_stop - full_start
        if full_length != sum(lengths):
            raise ValueError(
                f"full episode {full_episode_id} length {full_length} does not equal subtask sum {sum(lengths)}"
            )
        specs.append(
            _EpisodeSpec(
                group_id=group_id,
                full_episode_id=full_episode_id,
                subtask_episode_ids=subtask_ids,  # type: ignore[arg-type]
                lengths=tuple(lengths),  # type: ignore[arg-type]
                full_length=full_length,
                prompts=manifest.task_prompts,
            )
        )
    if not specs:
        raise ValueError(f"manifest contains no matched {split!r} groups")
    return tuple(sorted(specs, key=lambda item: item.full_episode_id))


def _prepare_model(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Loads the requested head and returns policy, dataset model, and JAX prefix fn."""

    import flax.nnx as nnx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    import openpi.models.model as model_api  # noqa: PLC0415
    from openpi.policies import policy_config  # noqa: PLC0415
    from openpi.training import config as training_config  # noqa: PLC0415

    config = training_config.get_config(args.config_name)
    if not bool(getattr(config.completion, "uses_temporal_completion", False)):
        raise ValueError(f"config {args.config_name!r} is not a temporal completion-head config")
    configured_mode = str(getattr(config.completion, "temporal_input_mode", "history"))
    if configured_mode != args.mode:
        raise ValueError(f"config temporal_input_mode={configured_mode!r} does not match --mode={args.mode!r}")
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"checkpoint params not found: {checkpoint / 'params'}")
    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        repack_transforms=_evaluation_repack(),
        sample_kwargs={},
    )
    if bool(getattr(policy, "_is_pytorch_model", False)):
        raise ValueError("semi-closed evaluator requires a JAX Pi0.5 checkpoint")
    model = policy._model  # noqa: SLF001
    if not hasattr(model, "compute_prefix_feature"):
        raise ValueError("loaded model lacks compute_prefix_feature")
    graphdef, state = nnx.split(model)

    def compute_prefix(state_value: Any, observation: Any) -> Any:
        module = nnx.merge(graphdef, state_value)
        return module.compute_prefix_feature(jax.random.key(0), observation, train=False)

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
    feature = np.asarray(jax.block_until_ready(compute_fn(state, observation)), dtype=np.float32)
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
    ends = gt_end_frames(spec.lengths)
    refs = reference_ticks(ends)
    controller = SemiClosedCompletionController(spec.prompts, threshold=threshold, mode=args.mode)
    feature_cache: dict[tuple[int, int, int], np.ndarray] = {}
    predicted: list[int | None] = [None] * 4
    ticks: list[dict[str, Any]] = []
    warmup_count = 0
    mismatch_count = 0
    done_frame: int | None = None
    for frame in range(0, spec.full_length, 15):
        task_before = controller.current_task_index
        prompt = controller.current_prompt
        if controller.done:
            feature = np.zeros(int(policy._model.prefix_feature_dim), dtype=np.float32)  # noqa: SLF001
        else:
            cache_key = (spec.full_episode_id, frame, task_before)
            feature = feature_cache.get(cache_key)
            if feature is None:
                feature = _prefix_feature(
                    policy=policy,
                    model_api=model_api,
                    jax=jax,
                    jnp=jnp,
                    compute_fn=compute_fn,
                    state=state,
                    dataset=full_dataset,
                    episode_id=spec.full_episode_id,
                    frame_index=frame,
                    prompt=prompt,
                )
                feature_cache[cache_key] = feature
        decision = controller.step(frame, feature, policy.score_temporal_completion)
        oracle_task = oracle_task_index(frame, ends)
        mismatch = decision.active_task_index != oracle_task
        mismatch_count += int(mismatch)
        if not decision.history_ready and not decision.done:
            warmup_count += 1
        if decision.triggered and predicted[decision.task_before] is None:
            predicted[decision.task_before] = frame
        if decision.done and done_frame is None:
            done_frame = frame
        ticks.append(
            {
                "frame_index": frame,
                "active_task_index": decision.active_task_index,
                "active_prompt": decision.active_prompt,
                "oracle_task_index": oracle_task,
                "history_ready": decision.history_ready,
                "target": int(frame >= refs[decision.task_before]),
                "score": decision.score,
                "threshold": threshold,
                "triggered": decision.triggered,
                "task_before": decision.task_before,
                "task_after": decision.task_after,
                "done": decision.done,
                "prompt_mismatch": mismatch,
            }
        )
    boundaries = [
        classify_boundary(task, refs[task], predicted[task], full_length=spec.full_length) for task in range(4)
    ]
    boundary_dicts = [result.to_dict() for result in boundaries]
    # A terminal boundary whose reference tick lies beyond the video is
    # reported separately as ``unavailable`` rather than counted as a failure.
    # A missing trigger at that boundary is still a real miss.
    first_failure = next(
        (result.task_index for result in boundaries if result.classification not in ("correct", "unavailable")),
        None,
    )
    return {
        "full_episode_id": spec.full_episode_id,
        "group_id": spec.group_id,
        "full_length": spec.full_length,
        "subtask_episode_ids": list(spec.subtask_episode_ids),
        "lengths": list(spec.lengths),
        "gt_end_frames": list(ends),
        "reference_ticks": list(refs),
        "reference_tick_available": [ref < spec.full_length for ref in refs],
        "predicted_ticks": predicted,
        "boundary_results": boundary_dicts,
        "all_correct": all(result.classification in ("correct", "unavailable") for result in boundaries),
        "first_failure_task": first_failure,
        "early_count": sum(result.classification == "early" for result in boundaries),
        "late_count": sum(result.classification == "late" for result in boundaries),
        "missed_count": sum(result.classification == "missed" for result in boundaries),
        "done": controller.done,
        "done_frame": done_frame,
        "prompt_mismatch_count": mismatch_count,
        "prompt_mismatch_rate": mismatch_count / len(ticks) if ticks else None,
        "history_not_ready_count": warmup_count,
        "history_not_ready_rate": warmup_count / len(ticks) if ticks else None,
        "ticks": ticks,
    }


def _summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    boundary_objects = [BoundaryResult(**boundary) for episode in episodes for boundary in episode["boundary_results"]]
    per_task: dict[str, Any] = {}
    for task in range(4):
        per_task[str(task)] = summarize_boundary_results([item for item in boundary_objects if item.task_index == task])
    all_ticks = sum(len(item["ticks"]) for item in episodes)
    mismatches = sum(int(item["prompt_mismatch_count"]) for item in episodes)
    warmups = sum(int(item["history_not_ready_count"]) for item in episodes)
    incorrect = [int(item["full_episode_id"]) for item in episodes if not item["all_correct"]]
    early = [int(item["full_episode_id"]) for item in episodes if item["early_count"]]
    late = [int(item["full_episode_id"]) for item in episodes if item["late_count"]]
    missed = [int(item["full_episode_id"]) for item in episodes if item["missed_count"]]
    unavailable = [
        int(item["full_episode_id"])
        for item in episodes
        if any(not boundary["reference_tick_available"] for boundary in item["boundary_results"])
    ]
    return {
        "episode_count": len(episodes),
        "all_correct_count": sum(bool(item["all_correct"]) for item in episodes),
        "all_correct_rate": sum(bool(item["all_correct"]) for item in episodes) / len(episodes) if episodes else None,
        "done_count": sum(bool(item["done"]) for item in episodes),
        "done_rate": sum(bool(item["done"]) for item in episodes) / len(episodes) if episodes else None,
        "prompt_mismatch_count": mismatches,
        "prompt_mismatch_rate": mismatches / all_ticks if all_ticks else None,
        "history_not_ready_count": warmups,
        "history_not_ready_rate": warmups / all_ticks if all_ticks else None,
        "boundaries": summarize_boundary_results(boundary_objects),
        "per_task": per_task,
        "incorrect_episode_ids": sorted(set(incorrect)),
        "early_episode_ids": sorted(set(early)),
        "late_episode_ids": sorted(set(late)),
        "missed_episode_ids": sorted(set(missed)),
        "terminal_unavailable_episode_ids": sorted(set(unavailable)),
    }


def evaluate(args: argparse.Namespace) -> Path:
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
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: PLC0415

    subtask_dataset = lerobot_dataset.LeRobotDataset(manifest.source_subtask_repo_id, root=args.subtask_dataset_root)
    full_repo_id = manifest.source_full_repo_id or "modanqing/agilex_make_breakfast_730"
    full_dataset = lerobot_dataset.LeRobotDataset(full_repo_id, root=args.full_dataset_root)
    metadata = lerobot_dataset.LeRobotDatasetMetadata(manifest.source_subtask_repo_id, root=args.subtask_dataset_root)
    ordered_prompts = _resolve_logical_prompts(metadata.tasks)
    if ordered_prompts != manifest.task_prompts:
        raise ValueError("subtask metadata prompts do not match manifest task_prompts")
    specs = _episode_specs(manifest, split=args.split, subtask_dataset=subtask_dataset, full_dataset=full_dataset)
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
    parser.add_argument("--mode", choices=("history", "current_only"), required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--full-dataset-root", type=Path, default=DEFAULT_FULL_DATASET_ROOT)
    parser.add_argument("--subtask-dataset-root", type=Path, default=DEFAULT_SUBTASK_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--output", type=Path, required=True)
    threshold_group = parser.add_mutually_exclusive_group(required=True)
    threshold_group.add_argument("--validation-report", type=Path)
    threshold_group.add_argument("--threshold", type=float)
    return parser


def main() -> None:
    output = evaluate(_parser().parse_args())
    print(f"Wrote semi-closed temporal completion report: {output}")


if __name__ == "__main__":
    main()
