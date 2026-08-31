"""Semi-closed evaluation for the standalone joint progress/done head."""

# ruff: noqa: E402, I001, PLC0415 -- configure XLA before importing JAX-backed model modules.

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
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np

from openpi.models import completion as completion_model
from openpi.models import progress_done
from openpi.training import breakfast_done_data

try:
    from scripts import evaluate_breakfast_done_semiclosed as breakfast_eval
    from scripts import evaluate_temporal_completion_semiclosed as base
except ImportError:
    import evaluate_breakfast_done_semiclosed as breakfast_eval
    import evaluate_temporal_completion_semiclosed as base


DEFAULT_HEAD_PARAMS = Path(
    "/home/geek/share3/vla_done/v2/progress_done_head_h768_seed42_20260831/checkpoints/step_001400/params"
)
DEFAULT_OUTPUT = Path(
    "/home/geek/share3/vla_done/v2/progress_done_head_h768_seed42_20260831/"
    "semiclosed_step1400_training_aligned/report.json"
)
DEFAULT_NORM_STATS_ASSET_ID = (
    "modanqing/agilex_make_breakfast_generalize_720_subtasks_pickbread600_water400_button300_putbread200"
)


class _ProgressDonePolicy:
    """Adds standalone dual-head scoring to an unchanged backbone policy."""

    def __init__(self, backbone: Any, score_fn: Any):
        self._backbone = backbone
        self._model = backbone._model  # noqa: SLF001
        self._input_transform = backbone._input_transform  # noqa: SLF001
        self._score_fn = score_fn
        self.progress_scores: list[float] = []

    def score_temporal_completion(self, history: np.ndarray) -> float:
        input_dim = int(self._model.prefix_feature_dim)
        if history.ndim != 3 or history.shape[0] != 3 or history.shape[-1] != input_dim + 1:
            raise ValueError(f"token history must have shape [3, N, {input_dim + 1}], got {history.shape}")
        tokens = history[..., :input_dim]
        masks = history[..., input_dim] > 0.5
        done_logits, progress_logits = self._score_fn(tokens[None, ...], masks[None, ...])
        done_score = float(np.asarray(done_logits)[0])
        progress_score = float(np.asarray(progress_logits)[0])
        done_probability = 1.0 / (1.0 + np.exp(-done_score))
        progress_probability = 1.0 / (1.0 + np.exp(-progress_score))
        self.progress_scores.append(progress_probability)
        return done_probability


def _spearman_correlation(predictions: np.ndarray, targets: np.ndarray) -> float | None:
    def average_ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        ranks = np.empty(len(values), dtype=np.float64)
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            ranks[order[start:stop]] = (start + stop - 1) / 2.0
            start = stop
        return ranks

    if len(predictions) < 2:
        return None
    prediction_ranks = average_ranks(predictions)
    target_ranks = average_ranks(targets)
    if np.ptp(prediction_ranks) == 0.0 or np.ptp(target_ranks) == 0.0:
        return None
    return float(np.corrcoef(prediction_ranks, target_ranks)[0, 1])


def progress_summary(episodes: Sequence[dict[str, Any]]) -> dict[str, float | int | None]:
    scored = [tick for episode in episodes for tick in episode["ticks"] if tick.get("progress_score") is not None]
    if not scored:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "spearman": None,
            "early_mae": None,
            "middle_mae": None,
            "late_mae": None,
            "backward_step_rate": None,
            "trigger_progress_mean": None,
            "trigger_progress_min": None,
            "trigger_progress_max": None,
        }
    predictions = np.asarray([tick["progress_score"] for tick in scored], dtype=np.float64)
    targets = np.asarray([tick["progress_target"] for tick in scored], dtype=np.float64)
    errors = predictions - targets
    absolute_errors = np.abs(errors)
    early = targets < 1.0 / 3.0
    middle = (targets >= 1.0 / 3.0) & (targets < 2.0 / 3.0)
    late = targets >= 2.0 / 3.0
    backward = 0
    comparable = 0
    for episode in episodes:
        previous: dict[int, float] = {}
        for tick in episode["ticks"]:
            if tick.get("progress_score") is None:
                continue
            task = int(tick["source_task_index"])
            score = float(tick["progress_score"])
            if task in previous:
                comparable += 1
                backward += int(score < previous[task])
            previous[task] = score
    triggered = [float(tick["progress_score"]) for tick in scored if bool(tick["triggered"])]
    return {
        "count": len(scored),
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "spearman": _spearman_correlation(predictions, targets),
        "early_mae": float(np.mean(absolute_errors[early])) if np.any(early) else None,
        "middle_mae": float(np.mean(absolute_errors[middle])) if np.any(middle) else None,
        "late_mae": float(np.mean(absolute_errors[late])) if np.any(late) else None,
        "backward_step_rate": backward / comparable if comparable else None,
        "trigger_progress_mean": float(np.mean(triggered)) if triggered else None,
        "trigger_progress_min": float(np.min(triggered)) if triggered else None,
        "trigger_progress_max": float(np.max(triggered)) if triggered else None,
    }


def _query_playback_starts(
    samples: Sequence[breakfast_done_data.DoneSample],
    *,
    episode_id: int,
) -> tuple[int, int, int, int]:
    episode_samples = [sample for sample in samples if sample.episode_index == episode_id]
    return tuple(  # type: ignore[return-value]
        min(sample.query_frame for sample in episode_samples if sample.current_sub_task == task_id)
        for task_id in breakfast_done_data.SUB_TASK_IDS
    )


def _attach_progress(
    result: dict[str, Any],
    scores: Sequence[float],
    annotation_starts: tuple[int, int, int, int],
) -> None:
    score_iterator = iter(scores)
    result["annotation_start_frames"] = list(annotation_starts)
    for tick in result["ticks"]:
        if tick["history_ready"]:
            tick["progress_score"] = float(next(score_iterator))
        else:
            tick["progress_score"] = None
        task = int(tick["source_task_index"])
        start = annotation_starts[task]
        end = int(result["gt_end_frames"][task])
        tick["progress_target"] = float(
            np.clip((int(tick["source_frame_index"]) - start) / max(end - start, 1), 0.0, 1.0)
        )
    try:
        next(score_iterator)
    except StopIteration:
        return
    raise ValueError("head produced more progress scores than evaluated history rows")


def _prepare(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    import openpi.models.model as model_api
    from openpi.policies import policy_config
    from openpi.training import checkpoints as training_checkpoints
    from openpi.training import config as training_config

    config = training_config.get_config(args.config_name)
    checkpoint = args.checkpoint.resolve()
    norm_stats = training_checkpoints.load_norm_stats(checkpoint / "assets", args.norm_stats_asset_id)
    backbone = policy_config.create_trained_policy(
        config,
        checkpoint,
        repack_transforms=base._evaluation_repack(),  # noqa: SLF001
        sample_kwargs={},
        norm_stats=norm_stats,
    )
    model = backbone._model  # noqa: SLF001
    model.eval()
    graphdef, state = nnx.split(model)

    def compute_prefix(state_value: Any, observation: Any) -> Any:
        module = nnx.merge(graphdef, state_value)
        return module.compute_prefix_tokens(jax.random.key(0), observation, train=False)

    head_config = completion_model.CompletionHeadConfig(
        enabled=True,
        variant="token_query_attention",
        hidden_dim=768,
        query_count=32,
        attention_heads=12,
        temporal_layers=3,
        dropout_rate=0.1,
    )
    head = progress_done.TokenQueryProgressDoneHead(
        input_dim=int(model.prefix_feature_dim), config=head_config, rngs=nnx.Rngs(0)
    )
    head_graphdef, head_state = nnx.split(head)
    restored = model_api.restore_params(args.head_params.resolve(), dtype=jnp.float32)
    if set(restored) != {"completion_head"}:
        raise ValueError("head checkpoint must contain only completion_head")
    head_state.replace_by_pure_dict(restored["completion_head"])

    @jax.jit
    def score(tokens: Any, masks: Any) -> Any:
        module = nnx.merge(head_graphdef, head_state)
        return module(tokens, masks, train=False)

    return backbone, model_api, jax, jnp, jax.jit(compute_prefix), state, score


def evaluate(args: argparse.Namespace) -> Path:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite semi-closed report: {args.output}")
    max_terminal_hold_seconds, max_terminal_hold_ticks = base.validate_max_terminal_hold_seconds(
        args.max_terminal_hold_seconds
    )
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("max-episodes must be positive")
    args.max_terminal_hold_seconds = max_terminal_hold_seconds
    args.mode = "history"

    dataset_root = args.dataset_root.resolve()
    annotation_root = args.annotation_root.resolve()
    done_dataset = breakfast_done_data.load_breakfast_done_dataset(dataset_root, annotation_root)
    samples = tuple(sample for sample in done_dataset.samples if sample.split == "val")
    episode_by_id = {episode.index: episode for episode in done_dataset.episodes}
    os.environ.setdefault("HF_LEROBOT_HOME", str(dataset_root.parent))
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    dataset = lerobot_dataset.LeRobotDataset(args.dataset_repo_id, root=dataset_root)
    specs, missing_terminal = breakfast_eval._build_specs(  # noqa: SLF001
        samples=samples,
        test_episode_ids=done_dataset.split_episode_ids["val"],
        task_prompts=done_dataset.task_prompts,
        dataset=dataset,
    )
    if args.max_episodes is not None:
        specs = specs[: args.max_episodes]

    backbone, model_api, jax, jnp, compute_fn, state, score_fn = _prepare(args)
    policy = _ProgressDonePolicy(backbone, score_fn)
    episodes = []
    for index, spec in enumerate(specs, start=1):
        policy.progress_scores.clear()
        playback_starts = _query_playback_starts(samples, episode_id=spec.full_episode_id)
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
            playback_start_frames=playback_starts,
            use_query_history=True,
        )
        _attach_progress(
            result,
            policy.progress_scores,
            episode_by_id[spec.full_episode_id].stage_starts,
        )
        episodes.append(result)
        print(
            f"Evaluated {index}/{len(specs)} episode={spec.full_episode_id} "
            f"tasks={[item['classification'] for item in result['task_results']]}",
            flush=True,
        )

    report = {
        "schema_version": 2,
        "evaluation_protocol": "breakfast_progress_done_training_aligned_gated_v2",
        "history_source": "query_frame_triplet_seed_then_rolling",
        "switch_output": "done",
        "scheduler_hz": 2,
        "fps": 30,
        "threshold": args.threshold,
        "max_terminal_hold_seconds": max_terminal_hold_seconds,
        "max_terminal_hold_ticks": max_terminal_hold_ticks,
        "checkpoint": str(args.checkpoint.resolve()),
        "head_params": str(args.head_params.resolve()),
        "annotation_root": str(annotation_root),
        "dataset_root": str(dataset_root),
        "fully_annotated_episode_count": len(specs),
        "excluded_missing_terminal_episode_ids": list(missing_terminal),
        "summary": {
            **base._summary(episodes),  # noqa: SLF001
            "progress": progress_summary(episodes),
        },
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return args.output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, default=breakfast_eval.DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=breakfast_eval.DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default="local/agilex_make_breakfast_330_2")
    parser.add_argument("--config-name", default="pi05_730_breakfast_subtasks")
    parser.add_argument("--checkpoint", type=Path, default=breakfast_eval.DEFAULT_CHECKPOINT)
    parser.add_argument("--head-params", type=Path, default=DEFAULT_HEAD_PARAMS)
    parser.add_argument("--norm-stats-asset-id", default=DEFAULT_NORM_STATS_ASSET_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-terminal-hold-seconds", type=float, default=2.0)
    parser.add_argument("--max-episodes", type=int)
    return parser


def main() -> None:
    output = evaluate(_parser().parse_args())
    print(f"Wrote progress+done semi-closed report: {output}")


if __name__ == "__main__":
    main()
