"""Checks whether progress/completion is even recoverable from one frozen frame.

This is deliberately independent of any trained completion head. For a random
sample of frames drawn from many episodes, it:

  1. Runs the frozen PaliGemma prefix forward pass (the same code path
     ``Pi0.compute_completion_logits`` uses) and mean-pools the prefix tokens
     into one feature vector per frame -- NOT using a trained
     ``CompletionHead``'s learned attention query, so this measures what is
     present in the frozen representation itself, not what one particular
     head happened to extract from it.
  2. For every frame, finds its nearest neighbor (by cosine distance) among
     frames from OTHER episodes.
  3. Reports how often "this frame looks nearly identical to that frame"
     coincides with "but their true labels are very different". If that
     happens often among the closest pairs, no head -- this one or a bigger
     one -- can be expected to resolve it, because the ambiguity is already
     in the frozen input, not in how it is read out.

The VLM backbone is frozen through both S1 and S2 training, so any of the S2
checkpoints yield identical prefix features; --config-name only changes which
labeled dataset supplies the ground-truth target. The full-episode linear
``progress`` label is the default because it gives a real-valued target for
every frame of every episode, which is the richest signal to test ambiguity
against (window labels are mostly 0 with a rare nonzero region).
"""

from __future__ import annotations

import argparse
import json
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
        local_frames = rng.sample(range(length), min(frames_per_episode, length))
        for local_frame in local_frames:
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
    """Runs the frozen prefix + mean-pool over the sampled frames; saves an .npz."""

    # Must be set before the first import of anything under lerobot.common.datasets
    # -- it resolves the local cache root at import time, not per-call, so setting
    # it after importing the module (even earlier in this same function) is too late
    # and silently falls back to the default ~/.cache/huggingface/lerobot root.
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


def analyze(npz_path: Path, *, top_k: int) -> None:
    """Nearest-cross-episode-neighbor analysis: does 'looks the same' imply 'similar label'?"""

    with np.load(npz_path, allow_pickle=False) as values:
        episode_index = values["episode_index"]
        frame_index = values["frame_index"]
        target = values["target"].astype(np.float64)
        feature = values["feature"].astype(np.float64)

    norms = np.linalg.norm(feature, axis=1, keepdims=True)
    normed = feature / np.maximum(norms, 1e-8)
    similarity = normed @ normed.T
    distance = 1.0 - similarity
    n = distance.shape[0]
    same_episode = episode_index[:, None] == episode_index[None, :]
    np.fill_diagonal(same_episode, True)
    distance[same_episode] = np.inf

    nearest_index = np.argmin(distance, axis=1)
    nearest_distance_full = distance[np.arange(n), nearest_index]
    label_gap_full = np.abs(target - target[nearest_index])

    finite = np.isfinite(nearest_distance_full)
    original_index = np.arange(n)[finite]
    neighbor_original_index = nearest_index[finite]
    nearest_distance = nearest_distance_full[finite]
    label_gap = label_gap_full[finite]
    order = np.argsort(nearest_distance)

    correlation = np.corrcoef(nearest_distance, label_gap)[0, 1]
    print(f"\nFrames analyzed: {n}")
    print(f"Correlation(nearest-neighbor distance, |label gap|) = {correlation:.3f}")
    print("(near 0 or negative => 'looks the same' does NOT predict 'similar label';")
    print(" the frozen representation does not disambiguate the target on its own)\n")

    print("Label gap by closeness percentile (closest pairs first):")
    for fraction in (0.01, 0.05, 0.10, 0.25, 0.50, 1.00):
        count = max(1, int(fraction * len(order)))
        bucket = label_gap[order[:count]]
        print(
            f"  closest {fraction * 100:5.1f}% ({count:5d} pairs): "
            f"mean |gap|={bucket.mean():.3f}  median |gap|={np.median(bucket):.3f}  "
            f"frac(|gap|>0.3)={(bucket > 0.3).mean():.3f}"
        )

    print(f"\nWorst offenders (smallest feature distance, largest label gap) -- top {top_k}:")
    worst_score = -nearest_distance + label_gap  # small distance AND large gap ranks first
    worst_order = np.argsort(-worst_score)
    for rank in worst_order[:top_k]:
        i = original_index[rank]
        neighbor = neighbor_original_index[rank]
        print(
            f"  episode {episode_index[i]} frame {frame_index[i]} (target={target[i]:.3f})  <->  "
            f"episode {episode_index[neighbor]} frame {frame_index[neighbor]} (target={target[neighbor]:.3f})  "
            f"distance={nearest_distance[rank]:.4f}  gap={label_gap[rank]:.3f}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="pi05_agilex_breakfast_frozen_head_s2_progress_head")
    parser.add_argument("--exp-name", default="s2_progress_head_full")
    parser.add_argument("--checkpoint-step", default="7999")
    parser.add_argument("--checkpoint-base", type=Path, default=Path("/mnt/data/models/wyt/checkpoints"))
    parser.add_argument("--hf-lerobot-home", type=Path, default=Path("/mnt/data/models/wyt/data"))
    parser.add_argument("--num-episodes", type=int, default=250)
    parser.add_argument("--frames-per-episode", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--output", type=Path, default=Path("/tmp/frozen_feature_ambiguity.npz"))
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
