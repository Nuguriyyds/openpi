"""Build a three-frame raw-prefix history index and missing-feature extension.

Missing prefix batches are written directly to bounded NPY shards.  The
extractor never materializes the complete extension array or creates a large
write-side mmap, which keeps the workflow usable on mounted filesystems.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np

from openpi.training import raw_prefix_completion_features as raw_features
from openpi.training import temporal_completion_data as temporal_data
from openpi.training import temporal_raw_prefix_completion_features as temporal_raw_features

DEFAULT_CONFIG_NAME = "pi05_730_breakfast_subtasks"
DEFAULT_CHECKPOINT = Path(
    "/mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999"
)
DEFAULT_DATASET_ROOT = Path("/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730")
DEFAULT_HF_LEROBOT_HOME = Path("/mnt/data/dataset/ei/huggingface")
DEFAULT_MANIFEST = Path("/mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v4.json")
DEFAULT_BASE_CACHE = Path("/mnt/data/models/wyt/evaluations/current_raw_prefix_tokens_v1")
DEFAULT_OUTPUT = Path("/mnt/data/models/wyt/evaluations/temporal_raw_prefix_history_v1")


def _scalar_int(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _extract_missing_history(
    args: argparse.Namespace,
    *,
    manifest: temporal_data.TemporalCompletionManifest,
    base_cache: raw_features.RawPrefixCompletionCache,
    plan: temporal_raw_features.TemporalRawPrefixHistoryPlan,
    writer: temporal_raw_features.TemporalRawPrefixHistoryWriter,
) -> None:
    """Extracts only missing keys and streams each model batch to the writer."""

    if not plan.extension_count:
        return

    # Heavy model/dataset imports are kept behind the missing-feature gate; a
    # complete sidecar can therefore be built without another VLM pass.
    import flax.nnx as nnx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: PLC0415

    import openpi.models.model as model_api  # noqa: PLC0415
    from openpi.policies import policy_config  # noqa: PLC0415
    from openpi.training import config as training_config  # noqa: PLC0415

    checkpoint = args.checkpoint.resolve()
    os.environ["HF_LEROBOT_HOME"] = str(args.hf_lerobot_home.resolve())
    config = training_config.get_config(args.config_name)
    # Reuse the clean-runtime and prompt/repack checks from the original
    # current-prefix extractor.  The base cache remains read-only.
    try:
        from scripts.extract_current_raw_prefix_tokens import (  # noqa: I001, PLC0415
            _evaluation_repack,
            _resolve_prompts,
            _validate_clean_runtime,
        )
    except ModuleNotFoundError:
        from extract_current_raw_prefix_tokens import (  # noqa: I001, PLC0415
            _evaluation_repack,
            _resolve_prompts,
            _validate_clean_runtime,
        )

    data_config = _validate_clean_runtime(config, checkpoint=checkpoint, manifest=manifest)
    dataset_root = args.dataset_root.resolve()
    if dataset_root != Path(manifest.source_subtask_root).resolve():
        raise ValueError("--dataset-root does not match manifest source_subtask_root")
    configured_root = (
        Path(data_config.lerobot_home).resolve() / str(data_config.repo_id)
        if data_config.lerobot_home is not None
        else args.hf_lerobot_home.resolve() / str(data_config.repo_id)
    )
    if configured_root != dataset_root:
        raise ValueError(f"clean config resolves dataset to {configured_root}, not {dataset_root}")

    metadata = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=dataset_root)
    prompts = _resolve_prompts(metadata.tasks)
    if prompts != manifest.task_prompts:
        raise ValueError("runtime task prompts differ from sealed manifest task_prompts")
    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        repack_transforms=_evaluation_repack(),
        sample_kwargs={},
    )
    if bool(getattr(policy, "_is_pytorch_model", False)):
        raise ValueError("temporal raw-prefix extraction requires a JAX clean checkpoint")
    model = policy._model  # noqa: SLF001
    if not hasattr(model, "compute_prefix_outputs"):
        raise ValueError("clean model lacks compute_prefix_outputs")
    model.eval()
    graphdef, state = nnx.split(model)

    def compute_prefix(state_value: Any, observation: Any) -> Any:
        module = nnx.merge(graphdef, state_value)
        return module.compute_prefix_outputs(jax.random.key(0), observation, train=False)

    compute_fn = jax.jit(compute_prefix)
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, root=dataset_root)
    episode_from = dataset.episode_data_index["from"]
    episode_to = dataset.episode_data_index["to"]
    for batch_start in range(0, plan.extension_count, args.batch_size):
        batch_keys = plan.missing_keys[batch_start : batch_start + args.batch_size]
        transformed: list[dict[str, Any]] = []
        for key in batch_keys:
            start = _scalar_int(episode_from[key.source_episode_id])
            stop = _scalar_int(episode_to[key.source_episode_id])
            dataset_index = start + key.source_frame_index
            if dataset_index < start or dataset_index >= stop:
                raise ValueError(f"source frame {key.source_episode_id}:{key.source_frame_index} is out of range")
            sample = dict(dataset[dataset_index])
            if _scalar_int(sample["episode_index"]) != key.source_episode_id:
                raise ValueError("source dataset episode_index disagrees with history key")
            if _scalar_int(sample["frame_index"]) != key.source_frame_index:
                raise ValueError("source dataset frame_index disagrees with history key")
            # The prompt is set before the exact existing repack/injection/
            # image/norm/tokenizer transform chain.
            sample["prompt"] = manifest.task_prompts[key.prompt_index]
            transformed.append(policy._input_transform(sample))  # noqa: SLF001

        batched_inputs = jax.tree.map(
            lambda *values: jnp.asarray(np.stack([np.asarray(value) for value in values], axis=0)),
            *transformed,
        )
        observation = model_api.Observation.from_dict(batched_inputs)
        prefix_out, prefix_mask, segment_ids, position_ids = compute_fn(state, observation)
        prefix_out_np = np.asarray(jax.block_until_ready(prefix_out), dtype=np.float32)
        prefix_mask_np = np.asarray(jax.block_until_ready(prefix_mask), dtype=np.bool_)
        segment_ids_np = np.asarray(jax.block_until_ready(segment_ids), dtype=np.int32)
        position_ids_np = np.asarray(jax.block_until_ready(position_ids), dtype=np.int32)
        if prefix_out_np.ndim != 3 or prefix_out_np.shape[0] != len(batch_keys):
            raise ValueError(f"compute_prefix_outputs returned invalid shape {prefix_out_np.shape}")
        if prefix_out_np.shape[1:] != (base_cache.metadata.token_count, base_cache.metadata.input_dim):
            raise ValueError("missing history feature shape differs from the base cache")
        if not np.isfinite(prefix_out_np).all():
            raise ValueError("compute_prefix_outputs returned non-finite values")
        if prefix_mask_np.shape != prefix_out_np.shape[:2]:
            raise ValueError("compute_prefix_outputs returned a mask with the wrong shape")
        if segment_ids_np.shape != (base_cache.metadata.token_count,) or position_ids_np.shape != segment_ids_np.shape:
            raise ValueError("compute_prefix_outputs returned invalid layout ids")
        if not np.array_equal(segment_ids_np, base_cache.prefix_segment_ids) or not np.array_equal(
            position_ids_np,
            base_cache.prefix_position_ids,
        ):
            raise ValueError("missing history prefix layout differs from the base cache")
        writer.append(prefix_out_np, prefix_mask_np, segment_ids_np, position_ids_np)
        stop = batch_start + len(batch_keys)
        print(f"Extracted and cached {stop}/{plan.extension_count} unique missing history prefixes")


def extract_history(args: argparse.Namespace) -> Path:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_shard_bytes <= 0:
        raise ValueError("--max-shard-bytes must be positive")
    output = args.output.resolve()

    manifest = temporal_data.load_temporal_manifest(args.manifest.resolve())
    base_cache_path = args.base_cache.resolve()
    base_cache = raw_features.load_raw_prefix_cache(
        base_cache_path,
        manifest=manifest,
        expected_checkpoint_path=str(args.checkpoint.resolve()),
        expected_model_config_name=args.config_name,
    )
    plan = temporal_raw_features.build_temporal_raw_prefix_history_plan(
        manifest,
        base_cache,
        sampling_protocol=args.sampling_protocol,
    )
    base_reused_slots = int(np.sum(plan.history_location_kind == temporal_raw_features.BASE_LOCATION))
    print(f"row_count: {plan.row_count}")
    print(f"base_reused_slots: {base_reused_slots}")
    print(f"unique_missing_feature_count: {plan.extension_count}")

    writer = temporal_raw_features.TemporalRawPrefixHistoryWriter(
        output,
        manifest=manifest,
        base_cache=base_cache,
        plan=plan,
        base_cache_path=base_cache_path,
        max_shard_bytes=args.max_shard_bytes,
        sampling_protocol=args.sampling_protocol,
    )
    try:
        _extract_missing_history(args, manifest=manifest, base_cache=base_cache, plan=plan, writer=writer)
        metadata = writer.finalize()
    except Exception:
        writer.abort()
        raise
    print(f"extension_shard_count: {metadata.extension_shard_count}")
    print(f"token_shape: {(metadata.token_count, metadata.input_dim)}")
    print(f"output: {output}")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--hf-lerobot-home", type=Path, default=DEFAULT_HF_LEROBOT_HOME)
    parser.add_argument("--base-cache", type=Path, default=DEFAULT_BASE_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sampling-protocol",
        choices=("subtask_local", "start_terminal"),
        default="subtask_local",
        help="history row protocol to materialize (default: subtask_local)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=temporal_raw_features.DEFAULT_MAX_SHARD_BYTES,
        help="maximum extension payload per NPY shard (default: 1 GiB)",
    )
    return parser


def main() -> None:
    extract_history(_parser().parse_args())


if __name__ == "__main__":
    main()
