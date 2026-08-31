"""Run gated semi-closed replay from breakfast boundary annotations."""

# ruff: noqa: E402, I001 -- configure XLA before importing JAX-backed model modules.

from __future__ import annotations

import os


_DETERMINISTIC_XLA_FLAGS = (
    "--xla_gpu_deterministic_ops=true",
    "--xla_gpu_exclude_nondeterministic_ops=true",
)


def _with_default_deterministic_xla_flags(flags: str) -> str:
    tokens = flags.split()
    configured = {token.split("=", maxsplit=1)[0] for token in tokens if token.startswith("--")}
    for flag in _DETERMINISTIC_XLA_FLAGS:
        if flag.split("=", maxsplit=1)[0] not in configured:
            tokens.append(flag)
    return " ".join(tokens)


os.environ["XLA_FLAGS"] = _with_default_deterministic_xla_flags(os.environ.get("XLA_FLAGS", ""))

import argparse
from collections import defaultdict
from collections.abc import Sequence
import itertools
import json
from pathlib import Path

from openpi.training import breakfast_done_data

try:
    from scripts import evaluate_temporal_completion_semiclosed as base
except ImportError:
    import evaluate_temporal_completion_semiclosed as base


DEFAULT_ANNOTATION_ROOT = Path(
    "/home/geek/share3/breakfest_data/rule_split_330-2_action_state_delay05_cut2dist10_overlap10/split"
)
DEFAULT_DATASET_ROOT = Path("/home/geek/share3/breakfest_data/agilex_make_breakfast_330-2")
DEFAULT_CHECKPOINT = Path("/home/geek/share3/vla_done/v2/39999")
DEFAULT_DONE_PARAMS = Path(
    "/home/geek/share3/vla_done/v2/done_head_h768_deterministic_full_seed42_20260828/"
    "checkpoints/step_001400/params"
)
DEFAULT_OUTPUT = Path(
    "/home/geek/share3/vla_done/v2/done_head_h768_deterministic_full_seed42_20260828/"
    "semiclosed_step1400/report.json"
)
DEFAULT_CONFIG_NAME = "pi05_agilex_breakfast_token_query_completion_head_h768"


def _build_specs(
    *,
    samples: Sequence[breakfast_done_data.DoneSample],
    test_episode_ids: Sequence[int],
    task_prompts: dict[str, str],
    dataset: object,
) -> tuple[tuple[base._EpisodeSpec, ...], tuple[int, ...]]:
    grouped: dict[int, list[breakfast_done_data.DoneSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.episode_index].append(sample)
    if set(grouped) != set(test_episode_ids):
        raise ValueError("validation samples do not match the generated episode split")

    task_names = tuple(task_prompts)
    if len(task_names) != 4:
        raise ValueError("semi-closed breakfast replay requires exactly four ordered tasks")
    prompts = tuple(task_prompts[name] for name in task_names)
    specs: list[base._EpisodeSpec] = []
    missing_terminal: list[int] = []
    for episode_id in test_episode_ids:
        positives = [sample for sample in grouped[episode_id] if sample.label == 1]
        transitions = {
            sample.current_sub_task: sample.query_frame for sample in positives if sample.sample_type == "transition"
        }
        terminals = [sample.query_frame for sample in positives if sample.sample_type == "terminal"]
        if set(transitions) != set(task_names[:3]):
            raise ValueError(f"episode {episode_id} does not contain exactly three task transitions")
        if not terminals:
            missing_terminal.append(episode_id)
            continue
        if len(terminals) != 1:
            raise ValueError(f"episode {episode_id} contains multiple terminal labels")

        ends = (*tuple(transitions[name] for name in task_names[:3]), terminals[0])
        starts = (0, ends[0] + 15, ends[1] + 15, ends[2] + 15)
        if any(right <= left for left, right in itertools.pairwise(ends)):
            raise ValueError(f"episode {episode_id} task boundaries are not increasing")
        episode_start, episode_stop = base._episode_bounds(dataset, episode_id)  # noqa: SLF001
        full_length = episode_stop - episode_start
        if ends[-1] >= full_length or any(start > end for start, end in zip(starts, ends, strict=True)):
            raise ValueError(f"episode {episode_id} boundaries exceed the full trajectory")
        lengths = tuple(end - start + 1 for start, end in zip(starts, ends, strict=True))
        specs.append(
            base._EpisodeSpec(  # noqa: SLF001
                group_id=episode_id,
                full_episode_id=episode_id,
                subtask_episode_ids=(episode_id,) * 4,
                lengths=lengths,
                subtask_start_frames=starts,
                gt_end_frames=ends,
                full_length=full_length,
                prompts=prompts,
            )
        )
    if not specs:
        raise ValueError("validation split contains no fully annotated terminal episodes")
    return tuple(specs), tuple(missing_terminal)


def evaluate(args: argparse.Namespace) -> Path:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite semi-closed report: {args.output}")
    max_terminal_hold_seconds, max_terminal_hold_ticks = base.validate_max_terminal_hold_seconds(
        args.max_terminal_hold_seconds
    )
    args.max_terminal_hold_seconds = max_terminal_hold_seconds
    args.mode = "history"
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("max_episodes must be positive")

    dataset_root = args.dataset_root.resolve()
    annotation_root = args.annotation_root.resolve()
    done_dataset = breakfast_done_data.load_breakfast_done_dataset(dataset_root, annotation_root)
    task_prompts = done_dataset.task_prompts
    samples = tuple(sample for sample in done_dataset.samples if sample.split == "val")
    test_episode_ids = done_dataset.split_episode_ids["val"]

    os.environ.setdefault("HF_LEROBOT_HOME", str(dataset_root.parent))
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    dataset = lerobot_dataset.LeRobotDataset(args.dataset_repo_id, root=dataset_root)
    specs, missing_terminal = _build_specs(
        samples=samples,
        test_episode_ids=test_episode_ids,
        task_prompts=task_prompts,
        dataset=dataset,
    )
    if args.max_episodes is not None:
        specs = specs[: args.max_episodes]

    policy, model_api, jax, jnp, compute_fn, state = base._prepare_model(args)  # noqa: SLF001
    episodes = []
    for index, spec in enumerate(specs, start=1):
        result = base._evaluate_episode(  # noqa: SLF001
            spec,
            args=args,
            policy=policy,
            model_api=model_api,
            jax=jax,
            jnp=jnp,
            compute_fn=compute_fn,
            state=state,
            full_dataset=dataset,
            threshold=args.threshold,
        )
        episodes.append(result)
        print(
            f"Evaluated {index}/{len(specs)} episode={spec.full_episode_id} "
            f"tasks={[item['classification'] for item in result['task_results']]}",
            flush=True,
        )

    report = {
        "schema_version": 2,
        "evaluation_protocol": "breakfast_gated_terminal_hold_stall_v2",
        "mode": "history",
        "scheduler_hz": 2,
        "fps": 30,
        "threshold": args.threshold,
        "threshold_source": "fixed_validation_rule",
        "max_terminal_hold_seconds": max_terminal_hold_seconds,
        "max_terminal_hold_ticks": max_terminal_hold_ticks,
        "checkpoint": str(args.checkpoint.resolve()),
        "done_params": str(args.done_params.resolve()) if args.done_params is not None else None,
        "config_name": args.config_name,
        "annotation_root": str(annotation_root),
        "dataset_root": str(dataset_root),
        "split": "test",
        "fully_annotated_episode_count": len(specs),
        "excluded_missing_terminal_episode_ids": list(missing_terminal),
        "summary": base._summary(episodes),  # noqa: SLF001
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return args.output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default="local/agilex_make_breakfast_330_2")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--done-params", type=Path, default=DEFAULT_DONE_PARAMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-terminal-hold-seconds", type=float, default=2.0)
    parser.add_argument("--max-episodes", type=int)
    return parser


def main() -> None:
    output = evaluate(_parser().parse_args())
    print(f"Wrote breakfast semi-closed report: {output}")


if __name__ == "__main__":
    main()
