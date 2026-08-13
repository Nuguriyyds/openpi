"""Checks whether the completion window's positive frames are even locally
distinguishable from the rest of their OWN episode.

Complements ``diagnose_frozen_feature_ambiguity.py``, which asks the
cross-episode/generalization question ("does 'near the end' look consistent
across different episodes of the same task"). This script asks the more
basic, necessary-but-not-sufficient question first: within one single
episode, are the labeled-positive frames (the last ``window_seconds``)
even distinguishable from that SAME episode's own earlier, negative frames?

If a positive frame's closest look-alike -- searched only within its own
episode -- is just as often an ordinary early frame as another positive
frame, then no head can detect the window even in principle, before ever
asking whether that detection generalizes to other episodes.

Sampling is stratified per episode (the last --tail-frames local frames,
which should cover the true positive window plus a margin, plus
--frames-per-episode random earlier frames) rather than uniform random,
because uniform random sampling would rarely land on a positive frame
(the window is only ~6% of a typical episode).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import time
from typing import Any

import numpy as np


def _scalar(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _sample_frame_specs(
    *,
    episode_from,
    episode_to,
    episode_ids: list[int],
    num_episodes: int,
    tail_frames: int,
    frames_per_episode: int,
    seed: int,
) -> list[tuple[int, int, int]]:
    rng = random.Random(seed)
    sampled_episodes = rng.sample(episode_ids, min(num_episodes, len(episode_ids)))
    frame_specs: list[tuple[int, int, int]] = []
    for episode_index in sampled_episodes:
        length = int(episode_to[episode_index]) - int(episode_from[episode_index])
        if length <= 0:
            continue
        tail_start = max(0, length - tail_frames)
        tail_local_frames = list(range(tail_start, length))
        earlier_pool = range(0, tail_start)
        earlier_local_frames = (
            rng.sample(earlier_pool, min(frames_per_episode, len(earlier_pool))) if earlier_pool else []
        )
        for local_frame in {*tail_local_frames, *earlier_local_frames}:
            dataset_index = int(episode_from[episode_index]) + local_frame
            frame_specs.append((episode_index, local_frame, dataset_index))
    return frame_specs


def _evaluation_repack():
    import openpi.transforms as transforms  # noqa: PLC0415

    return transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "images": {
                        "cam_top": "observation.image.top",
                        "cam_left_wrist": "observation.image.left_wrist",
                        "cam_right_wrist": "observation.image.right_wrist",
                    },
                    "state": "observation.state.joint",
                    "gripper_position": "observation.gripper_position",
                    "actions": "actions",
                    "prompt": "prompt",
                }
            )
        ]
    )


def extract_features(args: argparse.Namespace) -> Path:
    """Runs the frozen prefix + mean-pool over stratified-sampled frames; saves an .npz."""

    # Must be set before the first import of anything under lerobot.common.datasets --
    # it resolves the local cache root at import time, not per-call.
    os.environ["HF_LEROBOT_HOME"] = str(args.hf_lerobot_home.resolve())

    import jax
    import jax.numpy as jnp
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    import openpi.models.model as model_api
    from openpi.models.pi0 import make_attn_mask
    from openpi.policies import policy_config
    import flax.nnx as nnx
    from openpi.training import config as training_config

    config = training_config.get_config(args.config_name)
    checkpoint_dir = args.checkpoint_base / args.config_name / args.exp_name / str(args.checkpoint_step)
    if not (checkpoint_dir / "params").is_dir():
        raise FileNotFoundError(f"checkpoint params not found: {checkpoint_dir / 'params'}")
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("training config has no LeRobot repo_id")
    label_key = config.completion.label_key

    policy = policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        repack_transforms=_evaluation_repack(),
        sample_kwargs={},
    )
    model = policy._model  # noqa: SLF001
    model.eval()

    graphdef, state = nnx.split(model)

    def _pool(state, rng, observation):
        module = nnx.merge(graphdef, state)
        observation = model_api.preprocess_observation(rng, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = module.embed_prefix(observation)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = module.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
        prefix_out = prefix_out.astype(jnp.float32)
        mask_f = prefix_mask.astype(jnp.float32)[..., None]
        return jnp.sum(prefix_out * mask_f, axis=1) / jnp.maximum(jnp.sum(mask_f, axis=1), 1e-8)

    compute_fn = jax.jit(_pool)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id)
    tasks = dataset_meta.tasks
    episode_from = dataset.episode_data_index["from"]
    episode_to = dataset.episode_data_index["to"]
    all_episode_ids = sorted(dataset_meta.episodes)

    frame_specs = _sample_frame_specs(
        episode_from=episode_from,
        episode_to=episode_to,
        episode_ids=all_episode_ids,
        num_episodes=args.num_episodes,
        tail_frames=args.tail_frames,
        frames_per_episode=args.frames_per_episode,
        seed=args.seed,
    )
    if not frame_specs:
        raise ValueError("no frames were sampled")
    print(f"Sampled {len(frame_specs)} frames from {len({spec[0] for spec in frame_specs})} episodes")

    rng = jax.random.key(args.seed)
    episode_indices: list[int] = []
    frame_indices: list[int] = []
    targets: list[float] = []
    features: list[np.ndarray] = []

    started = time.monotonic()
    for batch_start in range(0, len(frame_specs), args.batch_size):
        batch_specs = frame_specs[batch_start : batch_start + args.batch_size]
        transformed_items: list[dict[str, Any]] = []
        batch_metadata: list[tuple[int, int, float]] = []
        for episode_index, local_frame_index, dataset_index in batch_specs:
            sample = dict(dataset[dataset_index])
            task_index = _scalar(sample["task_index"])
            sample["prompt"] = tasks[task_index]
            target = float(np.asarray(sample[label_key]).reshape(-1)[0])
            transformed_items.append(policy._input_transform(sample))  # noqa: SLF001
            batch_metadata.append((episode_index, local_frame_index, target))
        valid_count = len(transformed_items)
        while len(transformed_items) < args.batch_size:
            transformed_items.append(jax.tree.map(lambda value: np.array(value, copy=True), transformed_items[-1]))
        batched_inputs = jax.tree.map(
            lambda *values: jnp.asarray(np.stack([np.asarray(value) for value in values], axis=0)),
            *transformed_items,
        )
        observation = model_api.Observation.from_dict(batched_inputs)
        pooled = np.asarray(jax.block_until_ready(compute_fn(state, rng, observation)))
        for index, (episode_index, local_frame_index, target) in enumerate(batch_metadata):
            episode_indices.append(episode_index)
            frame_indices.append(local_frame_index)
            targets.append(target)
            features.append(pooled[index])
        if (batch_start // args.batch_size) % 10 == 0:
            elapsed = time.monotonic() - started
            print(f"  {batch_start + valid_count}/{len(frame_specs)} frames ({elapsed:.1f}s elapsed)")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        episode_index=np.asarray(episode_indices, dtype=np.int32),
        frame_index=np.asarray(frame_indices, dtype=np.int32),
        target=np.asarray(targets, dtype=np.float32),
        feature=np.asarray(features, dtype=np.float32),
        label_key=label_key,
    )
    print(f"Saved {len(features)} pooled features to {output_path}")
    return output_path


def _compute_within_episode_metrics(
    episode_index: np.ndarray,
    target: np.ndarray,
    feature: np.ndarray,
) -> dict[str, Any]:
    """Core within-episode nearest-neighbor analysis, free of I/O.

    For each frame, its nearest neighbor is searched ONLY within its own
    episode. Returns counts needed to answer "are the window's positive
    frames even locally distinguishable from that same episode's earlier,
    negative frames". Separated from ``analyze`` so it can be tested on
    synthetic features without touching disk.
    """

    label = (target >= 0.5).astype(np.int32)
    norms = np.linalg.norm(feature, axis=1, keepdims=True)
    normed = feature / np.maximum(norms, 1e-8)

    same_label_flags: list[bool] = []
    worst: list[tuple[float, int, int]] = []  # (distance, self_index, neighbor_index), positive->negative only
    positive_total = 0
    positive_confused = 0

    for episode in np.unique(episode_index):
        rows = np.flatnonzero(episode_index == episode)
        if len(rows) < 2:
            continue
        sub_normed = normed[rows]
        sub_label = label[rows]
        distance = 1.0 - sub_normed @ sub_normed.T
        np.fill_diagonal(distance, np.inf)
        nearest = np.argmin(distance, axis=1)
        for local_i, local_j in enumerate(nearest):
            same_label_flags.append(bool(sub_label[local_i] == sub_label[local_j]))
            if sub_label[local_i] == 1:
                positive_total += 1
                if sub_label[local_j] == 0:
                    positive_confused += 1
                    worst.append((float(distance[local_i, local_j]), int(rows[local_i]), int(rows[local_j])))

    return {
        "frame_count": int(len(label)),
        "positive_count": int(label.sum()),
        "negative_count": int((1 - label).sum()),
        "same_label_rate": float(np.mean(same_label_flags)) if same_label_flags else float("nan"),
        "positive_total": positive_total,
        "positive_confused": positive_confused,
        "worst": worst,
    }


def analyze(npz_path: Path, *, top_k: int) -> None:
    """Within-episode nearest-neighbor analysis: is the window locally distinguishable?"""

    with np.load(npz_path, allow_pickle=False) as values:
        episode_index = values["episode_index"]
        frame_index = values["frame_index"]
        target = values["target"].astype(np.float64)
        feature = values["feature"].astype(np.float64)

    metrics = _compute_within_episode_metrics(episode_index, target, feature)

    print(
        f"\nFrames analyzed: {metrics['frame_count']}  "
        f"(positive={metrics['positive_count']}, negative={metrics['negative_count']})"
    )
    sampled_base_rate = metrics["positive_count"] / max(metrics["frame_count"], 1)
    print(f"Sampled positive base rate: {sampled_base_rate:.3f}  (inflated by stratified tail sampling)")
    print(
        f"Fraction whose within-episode nearest neighbor shares the same label: "
        f"{metrics['same_label_rate']:.3f}"
    )
    print("(near 1.0 => positive/negative frames form locally separable clusters within their own episode;")
    print(" near the sampled base rate => the window is NOT even locally distinguishable from the rest")
    print(" of its own episode, before ever asking about generalization across episodes)\n")

    positive_total = metrics["positive_total"]
    positive_confused = metrics["positive_confused"]
    if positive_total:
        print(
            f"Of {positive_total} positive (last-window) frames, "
            f"{positive_confused} ({positive_confused / positive_total:.1%}) have an ordinary EARLIER frame from "
            "the SAME episode as their closest look-alike, not another positive frame."
        )
        print("(high % => a frozen single frame cannot tell 'in the window' from 'earlier in this same episode';")
        print(" no head -- this one or a bigger one -- can be expected to resolve that.)\n")

    worst = metrics["worst"]
    worst.sort(key=lambda item: item[0])
    print(f"Worst offenders (positive frame whose closest same-episode look-alike is a negative frame) -- top {top_k}:")
    for distance, i, j in worst[:top_k]:
        print(
            f"  episode {episode_index[i]}: positive frame {frame_index[i]} (target={target[i]:.3f})  <->  "
            f"negative frame {frame_index[j]} (target={target[j]:.3f})  distance={distance:.4f}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_agilex_breakfast_frozen_head_s2_completion_window")
    parser.add_argument("--exp-name", default="s2_completion_window")
    parser.add_argument("--checkpoint-step", default="7999")
    parser.add_argument("--checkpoint-base", type=Path, default=Path("/mnt/data/models/wyt/checkpoints"))
    parser.add_argument("--hf-lerobot-home", type=Path, default=Path("/mnt/data/models/wyt/data"))
    parser.add_argument("--num-episodes", type=int, default=150)
    parser.add_argument("--tail-frames", type=int, default=25, help="Local frames from the end to always include.")
    parser.add_argument("--frames-per-episode", type=int, default=20, help="Random EARLIER frames per episode.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("/tmp/window_within_episode.npz"))
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Re-analyze an existing --output file instead of re-running the model.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    npz_path = args.output if args.skip_extract else extract_features(args)
    analyze(npz_path, top_k=args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
