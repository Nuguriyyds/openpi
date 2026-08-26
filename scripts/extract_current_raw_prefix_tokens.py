"""Extract the current-frame, token-preserving raw-prefix completion cache."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import dataclasses
import os
from pathlib import Path
from typing import Any

import numpy as np

from openpi.training import raw_prefix_completion_features as raw_features
from openpi.training import temporal_completion_data as temporal_data

DEFAULT_CONFIG_NAME = "pi05_730_breakfast_subtasks"
DEFAULT_CHECKPOINT = Path(
    "/mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999"
)
DEFAULT_DATASET_ROOT = Path("/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730")
DEFAULT_HF_LEROBOT_HOME = Path("/mnt/data/dataset/ei/huggingface")
DEFAULT_MANIFEST = Path("/mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v4.json")
DEFAULT_OUTPUT = Path("/mnt/data/models/wyt/evaluations/current_raw_prefix_tokens_v1")


@dataclasses.dataclass(frozen=True, order=True)
class RawPrefixKey:
    source_episode_id: int
    source_frame_index: int
    prompt: str
    prompt_index: int = dataclasses.field(compare=False)


@dataclasses.dataclass(frozen=True)
class RawPrefixPlan:
    keys: tuple[RawPrefixKey, ...]
    row_key_indices: np.ndarray


def _resolve_prompts(tasks: Mapping[Any, Any]) -> tuple[str, str, str, str]:
    prompts: list[str] = []
    for task_index in range(temporal_data.TASKS_PER_TRAJECTORY):
        key: Any = task_index if task_index in tasks else str(task_index)
        if key not in tasks:
            raise ValueError(f"dataset metadata has no prompt for logical task {task_index}")
        prompt = tasks[key]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"dataset metadata prompt {task_index} is not a non-empty string")
        prompts.append(prompt)
    return tuple(prompts)  # type: ignore[return-value]


def build_raw_prefix_plan(
    rows: Sequence[temporal_data.TemporalSampleRow],
    prompts: Sequence[str],
) -> RawPrefixPlan:
    """Builds one current-frame request per canonical row with de-duplication."""

    if len(prompts) != temporal_data.TASKS_PER_TRAJECTORY:
        raise ValueError("raw-prefix extraction requires four task prompts")
    unique: set[RawPrefixKey] = set()
    row_keys: list[RawPrefixKey] = []
    for row in rows:
        if row.sample_kind == "transition_negative":
            raise ValueError("raw-prefix extraction does not accept transition rows")
        prompt_index = int(row.prompt_index)
        key = RawPrefixKey(
            source_episode_id=int(row.source_episode_ids[-1]),
            source_frame_index=int(row.source_frame_indices[-1]),
            prompt=str(prompts[prompt_index]),
            prompt_index=prompt_index,
        )
        row_keys.append(key)
        unique.add(key)
    ordered = tuple(sorted(unique))
    key_to_index = {key: index for index, key in enumerate(ordered)}
    row_key_indices = np.asarray([key_to_index[key] for key in row_keys], dtype=np.int64)
    return RawPrefixPlan(keys=ordered, row_key_indices=row_key_indices)


def _evaluation_repack() -> Any:
    # Keep one preprocessing/repack implementation for all frozen-prefix
    # extraction paths.  The import is local so planning tests stay light.
    try:
        from scripts.extract_temporal_completion_features import _evaluation_repack  # noqa: PLC0415
    except ModuleNotFoundError:
        from extract_temporal_completion_features import _evaluation_repack  # noqa: PLC0415

    return _evaluation_repack()


def _scalar_int(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _validate_clean_runtime(config: Any, *, checkpoint: Path, manifest: Any) -> Any:
    if config.name != DEFAULT_CONFIG_NAME:
        raise ValueError(f"raw-prefix extraction is locked to clean config {DEFAULT_CONFIG_NAME!r}")
    if bool(getattr(config.training_time_rtc, "enabled", False)):
        raise ValueError("raw-prefix extraction forbids TTRTC")
    completion_head = getattr(config.model, "completion_head", None)
    if completion_head is not None and bool(getattr(completion_head, "enabled", False)):
        raise ValueError("raw-prefix extraction forbids a completion head")
    if config.completion.stage != "disabled" or not bool(config.model.pi05):
        raise ValueError("raw-prefix extraction requires a clean pi0.5 config with completion disabled")
    expected_params = (checkpoint / "params").resolve()
    configured_params = Path(config.weight_loader.params_path).resolve()
    if configured_params != expected_params:
        raise ValueError(f"clean config params path {configured_params} does not match {expected_params}")
    if not expected_params.is_dir() or not (checkpoint / "assets").is_dir():
        raise FileNotFoundError(f"clean checkpoint is incomplete: {checkpoint}")
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id != manifest.source_subtask_repo_id or not bool(data_config.prompt_from_task):
        raise ValueError("clean config dataset/prompt settings do not match the sealed manifest")
    return data_config


def extract_raw_prefix(args: argparse.Namespace) -> Path:
    # Heavy JAX/LeRobot imports belong only in the actual extraction path.
    import flax.nnx as nnx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: PLC0415

    import openpi.models.model as model_api  # noqa: PLC0415
    from openpi.policies import policy_config  # noqa: PLC0415
    from openpi.training import config as training_config  # noqa: PLC0415

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sealed raw-prefix cache: {output}")
    manifest = temporal_data.load_temporal_manifest(args.manifest.resolve())
    dataset_root = args.dataset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if dataset_root != Path(manifest.source_subtask_root).resolve():
        raise ValueError("--dataset-root does not match manifest source_subtask_root")
    os.environ["HF_LEROBOT_HOME"] = str(args.hf_lerobot_home.resolve())
    config = training_config.get_config(args.config_name)
    data_config = _validate_clean_runtime(config, checkpoint=checkpoint, manifest=manifest)
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
    rows = raw_features.manifest_rows(manifest)
    plan = build_raw_prefix_plan(rows, prompts)
    if not plan.keys:
        raise ValueError("sealed manifest produced no raw-prefix requests")

    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        repack_transforms=_evaluation_repack(),
        sample_kwargs={},
    )
    if bool(getattr(policy, "_is_pytorch_model", False)):
        raise ValueError("raw-prefix extraction requires a JAX clean checkpoint")
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
    layout_segment_ids: np.ndarray | None = None
    layout_position_ids: np.ndarray | None = None
    writer = raw_features.RawPrefixCacheWriter(
        output,
        manifest=manifest,
        row_feature_indices=plan.row_key_indices,
        feature_count=len(plan.keys),
        model_config_name=args.config_name,
        checkpoint_path=str(checkpoint),
        max_shard_bytes=args.max_shard_bytes,
    )

    for batch_start in range(0, len(plan.keys), args.batch_size):
        batch_keys = plan.keys[batch_start : batch_start + args.batch_size]
        transformed: list[dict[str, Any]] = []
        for key in batch_keys:
            start = _scalar_int(episode_from[key.source_episode_id])
            stop = _scalar_int(episode_to[key.source_episode_id])
            dataset_index = start + key.source_frame_index
            if dataset_index < start or dataset_index >= stop:
                raise ValueError(f"source frame {key.source_episode_id}:{key.source_frame_index} is out of range")
            sample = dict(dataset[dataset_index])
            if _scalar_int(sample["episode_index"]) != key.source_episode_id:
                raise ValueError("source dataset episode_index disagrees with manifest reference")
            if _scalar_int(sample["frame_index"]) != key.source_frame_index:
                raise ValueError("source dataset frame_index disagrees with manifest reference")
            # Prompt override precedes the shared repack, AgileX, norm, resize,
            # and tokenizer transforms.
            sample["prompt"] = key.prompt
            transformed.append(policy._input_transform(sample))  # noqa: SLF001

        batched_inputs = jax.tree.map(
            lambda *values: jnp.asarray(np.stack([np.asarray(value) for value in values], axis=0)),
            *transformed,
        )
        observation = model_api.Observation.from_dict(batched_inputs)
        prefix_out, prefix_mask, segment_ids, position_ids = compute_fn(state, observation)
        prefix_out_np = np.asarray(jax.block_until_ready(prefix_out))
        prefix_mask_np = np.asarray(jax.block_until_ready(prefix_mask), dtype=np.bool_)
        segment_ids_np = np.asarray(jax.block_until_ready(segment_ids), dtype=np.int32)
        position_ids_np = np.asarray(jax.block_until_ready(position_ids), dtype=np.int32)
        expected_shape = (len(batch_keys), prefix_out_np.shape[1], int(model.prefix_feature_dim))
        if prefix_out_np.shape != expected_shape or not np.isfinite(prefix_out_np).all():
            raise ValueError(f"compute_prefix_outputs returned invalid shape/values: {prefix_out_np.shape}")
        if prefix_mask_np.shape != prefix_out_np.shape[:2]:
            raise ValueError("compute_prefix_outputs returned a mask with the wrong shape")
        if segment_ids_np.shape != (prefix_out_np.shape[1],) or position_ids_np.shape != segment_ids_np.shape:
            raise ValueError("compute_prefix_outputs returned invalid layout ids")
        if layout_segment_ids is None:
            layout_segment_ids = segment_ids_np
            layout_position_ids = position_ids_np
            bytes_per_feature = prefix_out_np.shape[1] * (
                prefix_out_np.shape[2] * np.dtype(np.float16).itemsize + np.dtype(np.bool_).itemsize
            )
            estimated_gib = len(plan.keys) * bytes_per_feature / float(1 << 30)
            print(f"Estimated sharded cache payload: {estimated_gib:.2f} GiB")
        elif not np.array_equal(layout_segment_ids, segment_ids_np) or not np.array_equal(
            layout_position_ids, position_ids_np
        ):
            raise ValueError("prefix layout changed between extraction batches")
        writer.append(prefix_out_np, prefix_mask_np, segment_ids_np, position_ids_np)
        print(f"Extracted {min(batch_start + len(batch_keys), len(plan.keys))}/{len(plan.keys)} unique current prefixes")

    assert layout_segment_ids is not None
    assert layout_position_ids is not None
    cache_metadata = writer.finalize()

    split_counts = Counter(row.split for row in rows)
    task_counts = Counter(row.task_index for row in rows)
    kind_counts = Counter(row.sample_kind for row in rows)
    print(f"rows: {len(rows)}")
    print(f"split_counts: {dict(sorted(split_counts.items()))}")
    print(f"task_counts: {dict(sorted(task_counts.items()))}")
    print(
        "sample_kind_counts: "
        f"positive={kind_counts['positive']} hard_negative={kind_counts['hard_negative']} "
        f"ordinary_negative={kind_counts['ordinary_negative']}"
    )
    print(f"unique_feature_count: {cache_metadata.feature_count}")
    print(f"shard_count: {cache_metadata.shard_count}")
    print(f"token_shape: {(cache_metadata.token_count, cache_metadata.input_dim)}")
    print(f"dtype: {cache_metadata.storage_dtype}")
    print(f"output: {output}")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--hf-lerobot-home", type=Path, default=DEFAULT_HF_LEROBOT_HOME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16, help="number of unique current prefixes per VLA batch")
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=raw_features.DEFAULT_MAX_SHARD_BYTES,
        help="maximum prefix payload per NPY shard (default: 1 GiB)",
    )
    return parser


def main() -> None:
    extract_raw_prefix(_parser().parse_args())


if __name__ == "__main__":
    main()
