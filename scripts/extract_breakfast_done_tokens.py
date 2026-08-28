"""Extract Pi0.5 prefix-token caches from breakfast boundary annotations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from openpi.training import breakfast_done_data

DEFAULT_ANNOTATION_ROOT = Path(
    "/home/geek/share3/breakfest_data/rule_split_330-2_action_state_delay05_cut2dist10_overlap10/split"
)
DEFAULT_DATASET_ROOT = Path("/home/geek/share3/breakfest_data/agilex_make_breakfast_330-2")
DEFAULT_CONFIG_NAME = "pi05_730_breakfast_subtasks"
DEFAULT_CHECKPOINT = Path(
    "/mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999"
)
DEFAULT_OUTPUT = Path("/home/geek/share3/vla_done/v2/qwen_done_v2/qwen_style_done_tokens_v2_shards")


def _evaluation_repack() -> Any:
    import openpi.transforms as transforms

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


def _scalar_int(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", default="local/agilex_make_breakfast_330_2")
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--norm-stats-asset-id")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--token-limit", type=int, default=968)
    return parser.parse_args(argv)


def extract(args: argparse.Namespace) -> Path:
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    import openpi.models.model as model_api
    from openpi.policies import policy_config
    from openpi.training import checkpoints as training_checkpoints
    from openpi.training import config as training_config

    if args.batch_size <= 0 or args.num_workers <= 0 or args.token_limit <= 0:
        raise ValueError("batch-size, num-workers, and token-limit must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")

    dataset_root = args.dataset_root.resolve()
    annotation_root = args.annotation_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve() / f"shard_{args.shard_index}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite token shard: {output}")

    done_dataset = breakfast_done_data.load_breakfast_done_dataset(dataset_root, annotation_root)
    task_prompts = done_dataset.task_prompts
    samples = done_dataset.samples
    train_count = sum(sample.split == "train" for sample in samples)
    val_count = sum(sample.split == "val" for sample in samples)
    plan = breakfast_done_data.build_feature_plan(samples)
    feature_start = len(plan.keys) * args.shard_index // args.shard_count
    feature_stop = len(plan.keys) * (args.shard_index + 1) // args.shard_count

    config = training_config.get_config(args.config_name)
    norm_stats = (
        None
        if args.norm_stats_asset_id is None
        else training_checkpoints.load_norm_stats(checkpoint / "assets", args.norm_stats_asset_id)
    )
    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        repack_transforms=_evaluation_repack(),
        sample_kwargs={},
        norm_stats=norm_stats,
    )
    if bool(getattr(policy, "_is_pytorch_model", False)):
        raise ValueError("token extraction requires a JAX checkpoint")
    model = policy._model  # noqa: SLF001
    model.eval()
    graphdef, state = nnx.split(model)

    def compute_prefix_tokens(model_state: Any, observation: model_api.Observation) -> tuple[Any, Any]:
        module = nnx.merge(graphdef, model_state)
        return module.compute_prefix_tokens(jax.random.key(0), observation, train=False)

    compute_fn = jax.jit(compute_prefix_tokens)
    dataset = lerobot_dataset.LeRobotDataset(args.dataset_repo_id, root=dataset_root)
    episode_from = dataset.episode_data_index["from"]
    episode_to = dataset.episode_data_index["to"]

    output.mkdir(parents=True, exist_ok=False)
    try:
        local_count = feature_stop - feature_start
        tokens = np.lib.format.open_memmap(
            output / "tokens.npy",
            mode="w+",
            dtype=np.float16,
            shape=(local_count, args.token_limit, int(model.prefix_feature_dim)),
        )
        masks = np.lib.format.open_memmap(
            output / "masks.npy",
            mode="w+",
            dtype=np.bool_,
            shape=(local_count, args.token_limit),
        )

        def prepare(key: breakfast_done_data.FeatureKey) -> dict[str, Any]:
            start = _scalar_int(episode_from[key.episode_index])
            stop = _scalar_int(episode_to[key.episode_index])
            dataset_index = start + key.frame_index
            if dataset_index < start or dataset_index >= stop:
                raise ValueError(f"source frame {key.episode_index}:{key.frame_index} is outside [{start}, {stop})")
            sample = dict(dataset[dataset_index])
            if _scalar_int(sample["episode_index"]) != key.episode_index:
                raise ValueError("source episode does not match boundary annotations")
            if _scalar_int(sample["frame_index"]) != key.frame_index:
                raise ValueError("source frame does not match boundary annotations")
            sample["prompt"] = key.prompt
            return policy._input_transform(sample)  # noqa: SLF001

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for batch_start in range(feature_start, feature_stop, args.batch_size):
                batch_stop = min(batch_start + args.batch_size, feature_stop)
                transformed = list(executor.map(prepare, plan.keys[batch_start:batch_stop]))
                inputs = jax.tree.map(
                    lambda *values: jnp.asarray(np.stack([np.asarray(value) for value in values])),
                    *transformed,
                )
                observation = model_api.Observation.from_dict(inputs)
                batch_tokens, batch_masks = jax.block_until_ready(compute_fn(state, observation))
                batch_tokens = np.asarray(batch_tokens)
                batch_masks = np.asarray(batch_masks, dtype=np.bool_)
                if batch_tokens.ndim != 3 or batch_tokens.shape[-1] != int(model.prefix_feature_dim):
                    raise ValueError(f"unexpected prefix-token shape: {batch_tokens.shape}")
                if batch_tokens.shape[1] < args.token_limit:
                    raise ValueError("token-limit exceeds model prefix length")
                if np.any(batch_masks[:, args.token_limit :]):
                    raise ValueError("token-limit truncates valid prefix tokens")
                local_start = batch_start - feature_start
                local_stop = local_start + len(transformed)
                tokens[local_start:local_stop] = batch_tokens[:, : args.token_limit].astype(np.float16)
                masks[local_start:local_stop] = batch_masks[:, : args.token_limit]
                tokens.flush()
                masks.flush()
                print(
                    f"Extracted {batch_stop - feature_start}/{local_count} token features "
                    f"for shard {args.shard_index + 1}/{args.shard_count}",
                    flush=True,
                )

        np.save(output / "history_indices.npy", plan.history_indices, allow_pickle=False)
        np.save(output / "labels.npy", np.asarray([sample.label for sample in samples], dtype=np.uint8))
        np.save(output / "splits.npy", np.asarray([sample.split for sample in samples], dtype="<U5"))
        np.save(output / "sample_ids.npy", np.asarray([sample.sample_id for sample in samples], dtype="<U96"))
        metadata = {
            "schema_version": 1,
            "source_protocol": "breakfast_boundary_task_end_full_tokens",
            "model_config_name": args.config_name,
            "checkpoint_path": str(checkpoint),
            "norm_stats_asset_id": args.norm_stats_asset_id,
            "dataset_root": str(dataset_root),
            "annotation_root": str(annotation_root),
            "task_prompts": task_prompts,
            "row_count": len(samples),
            "train_count": train_count,
            "val_count": val_count,
            "feature_count": len(plan.keys),
            "feature_dim": int(model.prefix_feature_dim),
            "token_count": args.token_limit,
            "storage_dtype": "float16",
            "history_times_seconds": [-1.0, -0.5, 0.0],
            "feature_plan_sha256": hashlib.sha256(
                json.dumps(
                    [(key.episode_index, key.frame_index, key.prompt) for key in plan.keys],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "feature_shard": {
                "index": args.shard_index,
                "count": args.shard_count,
                "start": feature_start,
                "stop": feature_stop,
            },
        }
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(output)
        raise
    print(f"Saved breakfast done token shard: {output}")
    return output


def main(argv: Sequence[str] | None = None) -> None:
    extract(_parse_args(argv))


if __name__ == "__main__":
    main()
