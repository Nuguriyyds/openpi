"""Quantify boundary/progress information in a clean pi0.5 checkpoint.

The script deliberately does not require a derived completion dataset.  It
constructs the boundary sample set virtually from the original four-subtask
episodes:

* ordinary negatives: every ``negative_stride`` source frames;
* subtasks 1/2/3: five original tail positives plus five observations copied
  virtually from the next subtask and evaluated under the previous prompt;
* subtask 4: ten original tail positives;
* subtasks 2/3/4: the first five frames under their own prompt are forced
  negatives, producing pixel-identical / different-prompt contrast pairs.

``extract`` restores an action-only checkpoint, freezes it, and saves three
representations per sample: mean-pooled prefix output, the last valid prefix
token, and mean-pooled final action-expert hidden state at a fixed flow time.
The action hidden state is averaged over a common deterministic noise bank so
noise cannot encode a label or group identity.

``analyze`` fits group-disjoint linear probes for both binary boundary and
continuous progress targets.  It reports overall and hard-boundary metrics,
paired-prompt accuracy, episode-level early-trigger/detection metrics, and the
probe-dependent usable information in bits.  Thresholds and regularization
are selected on validation groups only; test groups are touched once.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any

import numpy as np

DEFAULT_CONFIG_NAME = "pi05_730_breakfast_subtasks"
DEFAULT_CHECKPOINT = Path(
    "/mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999"
)
DEFAULT_DATASET_ROOT = Path("/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730")
DEFAULT_HF_LEROBOT_HOME = Path("/mnt/data/dataset/ei/huggingface")
DEFAULT_OUTPUT = Path("/mnt/data/models/wyt/evaluations/clean_completion_features")

SPLIT_TRAIN = 0
SPLIT_VAL = 1
SPLIT_TEST = 2

KIND_ORDINARY_NEGATIVE = 0
KIND_HARD_NEGATIVE = 1
KIND_FORCED_HEAD_NEGATIVE = 2
KIND_TAIL_POSITIVE = 3
KIND_COPY_POSITIVE = 4
KIND_SUBTASK4_POSITIVE = 5

KIND_NAMES = {
    KIND_ORDINARY_NEGATIVE: "ordinary_negative",
    KIND_HARD_NEGATIVE: "hard_negative",
    KIND_FORCED_HEAD_NEGATIVE: "forced_head_negative",
    KIND_TAIL_POSITIVE: "tail_positive",
    KIND_COPY_POSITIVE: "copy_positive",
    KIND_SUBTASK4_POSITIVE: "subtask4_positive",
}


@dataclasses.dataclass(frozen=True)
class SampleSpec:
    logical_episode: int
    source_episode: int
    source_frame: int
    logical_frame: int
    task_index: int
    group_index: int
    split: int
    completion: int
    progress: float
    kind: int
    pair_id: int = -1


def _split_group_ids(
    group_ids: list[int],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> dict[int, int]:
    """Returns a deterministic whole-group split assignment."""

    if not 0.0 < val_fraction < 1.0 or not 0.0 < test_fraction < 1.0:
        raise ValueError("val_fraction and test_fraction must lie in (0, 1)")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val_fraction + test_fraction must be less than 1")
    shuffled = list(group_ids)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_fraction))
    val_count = max(1, round(len(shuffled) * val_fraction))
    if test_count + val_count >= len(shuffled):
        raise ValueError("not enough groups for non-empty train/val/test splits")
    result = dict.fromkeys(shuffled, SPLIT_TRAIN)
    for group in shuffled[:test_count]:
        result[group] = SPLIT_TEST
    for group in shuffled[test_count : test_count + val_count]:
        result[group] = SPLIT_VAL
    return result


def build_virtual_sample_specs(
    episode_lengths: dict[int, int],
    *,
    negative_stride: int = 15,
    copy_frames: int = 5,
    positive_tail: int = 5,
    subtask4_tail: int = 10,
    hard_negative_frames: int = 60,
    split_seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> tuple[list[SampleSpec], dict[str, Any]]:
    """Builds the exact sparse boundary set without copying images or videos."""

    if negative_stride <= 0:
        raise ValueError("negative_stride must be positive")
    if min(copy_frames, positive_tail, subtask4_tail, hard_negative_frames) <= 0:
        raise ValueError("boundary widths must be positive")

    all_episode_ids = sorted(episode_lengths)
    group_ids = sorted({episode // 4 for episode in all_episode_ids})
    valid_groups: list[int] = []
    skipped_groups: list[int] = []
    for group in group_ids:
        episodes = [group * 4 + position for position in range(4)]
        if any(episode not in episode_lengths for episode in episodes):
            skipped_groups.append(group)
            continue
        required = [positive_tail, positive_tail, positive_tail, subtask4_tail]
        if any(episode_lengths[episode] < required[position] for position, episode in enumerate(episodes)):
            skipped_groups.append(group)
            continue
        if any(episode_lengths[episodes[position + 1]] < copy_frames for position in range(3)):
            skipped_groups.append(group)
            continue
        valid_groups.append(group)

    split_by_group = _split_group_ids(
        valid_groups,
        seed=split_seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    specs_by_key: dict[tuple[int, int, int], SampleSpec] = {}

    def add(spec: SampleSpec, *, replace: bool = False) -> None:
        key = (spec.logical_episode, spec.source_episode, spec.source_frame)
        if replace or key not in specs_by_key:
            specs_by_key[key] = spec

    for group in valid_groups:
        split = split_by_group[group]
        group_episodes = [group * 4 + position for position in range(4)]
        for position, episode in enumerate(group_episodes):
            length = episode_lengths[episode]
            tail_count = subtask4_tail if position == 3 else positive_tail
            tail_start = length - tail_count

            # Every-15-frame ordinary negatives.  Frames close to the boundary
            # are marked separately so easy middle negatives cannot hide a
            # failure on the actual decision boundary.
            for frame in range(0, tail_start, negative_stride):
                kind = KIND_HARD_NEGATIVE if frame >= tail_start - hard_negative_frames else KIND_ORDINARY_NEGATIVE
                progress = frame / max(length - 1, 1)
                add(
                    SampleSpec(
                        logical_episode=episode,
                        source_episode=episode,
                        source_frame=frame,
                        logical_frame=frame,
                        task_index=position,
                        group_index=group,
                        split=split,
                        completion=0,
                        progress=progress,
                        kind=kind,
                    )
                )

            if position > 0:
                # These rows are the negative half of the same-pixel,
                # different-prompt pair generated by the previous subtask's
                # virtual boundary copies.
                for frame in range(copy_frames):
                    pair_id = episode * 100 + frame
                    add(
                        SampleSpec(
                            logical_episode=episode,
                            source_episode=episode,
                            source_frame=frame,
                            logical_frame=frame,
                            task_index=position,
                            group_index=group,
                            split=split,
                            completion=0,
                            progress=frame / max(length - 1, 1),
                            kind=KIND_FORCED_HEAD_NEGATIVE,
                            pair_id=pair_id,
                        ),
                        replace=True,
                    )

            positive_kind = KIND_SUBTASK4_POSITIVE if position == 3 else KIND_TAIL_POSITIVE
            for frame in range(tail_start, length):
                add(
                    SampleSpec(
                        logical_episode=episode,
                        source_episode=episode,
                        source_frame=frame,
                        logical_frame=frame,
                        task_index=position,
                        group_index=group,
                        split=split,
                        completion=1,
                        progress=frame / max(length - 1, 1),
                        kind=positive_kind,
                    ),
                    replace=True,
                )

            if position < 3:
                next_episode = group_episodes[position + 1]
                for frame in range(copy_frames):
                    add(
                        SampleSpec(
                            logical_episode=episode,
                            source_episode=next_episode,
                            source_frame=frame,
                            logical_frame=length + frame,
                            task_index=position,
                            group_index=group,
                            split=split,
                            completion=1,
                            progress=1.0,
                            kind=KIND_COPY_POSITIVE,
                            pair_id=next_episode * 100 + frame,
                        )
                    )

    specs = sorted(
        specs_by_key.values(),
        key=lambda spec: (spec.split, spec.group_index, spec.logical_episode, spec.logical_frame, spec.source_episode),
    )
    counts_by_split = {
        name: sum(spec.split == split for spec in specs)
        for name, split in (("train", SPLIT_TRAIN), ("val", SPLIT_VAL), ("test", SPLIT_TEST))
    }
    positives_by_split = {
        name: sum(spec.split == split and spec.completion == 1 for spec in specs)
        for name, split in (("train", SPLIT_TRAIN), ("val", SPLIT_VAL), ("test", SPLIT_TEST))
    }
    audit = {
        "valid_group_count": len(valid_groups),
        "skipped_groups": skipped_groups,
        "counts_by_split": counts_by_split,
        "positives_by_split": positives_by_split,
        "kind_counts": {KIND_NAMES[kind]: sum(spec.kind == kind for spec in specs) for kind in sorted(KIND_NAMES)},
    }
    return specs, audit


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


def _read_episode_lengths(meta_dir: Path) -> dict[int, int]:
    result: dict[int, int] = {}
    for line in (meta_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            result[int(record["episode_index"])] = int(record["length"])
    return result


def _scalar(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _savez_compressed_mount_safe(path: Path, **arrays: np.ndarray) -> None:
    """Writes NPZ via local scratch for mounts that reject ZIP seek-back."""

    scratch_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="completion_features_", suffix=".npz", delete=False) as scratch:
            scratch_path = Path(scratch.name)
            np.savez_compressed(scratch, **arrays)
        shutil.copyfile(scratch_path, path)
    finally:
        if scratch_path is not None:
            scratch_path.unlink(missing_ok=True)


def _get_clean_config(training_config, config_name: str):
    """Loads a registered config or constructs the known clean fallback.

    The original clean checkpoint was produced before this diagnostic branch
    registered its staged breakfast configs.  Some training worktrees contain
    ``pi05_730_breakfast_subtasks`` and some do not, so keep the fallback local
    to the read-only diagnostic rather than mutating a shared config catalog.
    """

    try:
        return training_config.get_config(config_name)
    except ValueError:
        if config_name != DEFAULT_CONFIG_NAME:
            raise
        from openpi.models import pi0_config  # noqa: PLC0415

        return training_config.TrainConfig(
            name=DEFAULT_CONFIG_NAME,
            model=pi0_config.Pi0Config(pi05=True),
            data=training_config.LeRobotAGILEXDataConfig(
                repo_id="modanqing/agilex_make_breakfast_subtask_730",
                assets=training_config.AssetsConfig(asset_id="agilex_make_breakfast_subtask_730"),
                base_config=training_config.DataConfig(
                    prompt_from_task=True,
                ),
            ),
            num_train_steps=50_000,
            batch_size=64,
            num_workers=4,
        )


def extract_features(args: argparse.Namespace) -> Path:
    """Restores the clean model and saves frozen prefix/action representations."""

    os.environ["HF_LEROBOT_HOME"] = str(args.hf_lerobot_home.resolve())

    import flax.nnx as nnx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: PLC0415

    import openpi.models.model as model_api  # noqa: PLC0415
    from openpi.models.pi0 import make_attn_mask  # noqa: PLC0415
    from openpi.policies import policy_config  # noqa: PLC0415
    from openpi.training import config as training_config  # noqa: PLC0415

    dataset_root = args.dataset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    if not (dataset_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"dataset metadata not found: {dataset_root}")
    if not (checkpoint / "params").is_dir():
        raise FileNotFoundError(f"checkpoint params not found: {checkpoint / 'params'}")

    config = _get_clean_config(training_config, args.config_name)
    if getattr(getattr(config, "training_time_rtc", None), "enabled", False):
        raise ValueError(f"config {args.config_name!r} enables TTRTC; a clean action-only config is required")
    if getattr(getattr(config.model, "completion_head", None), "enabled", False):
        raise ValueError(f"config {args.config_name!r} enables a completion head; the probe must use a clean model")
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("training config has no LeRobot repo_id")
    expected_root = (args.hf_lerobot_home.resolve() / data_config.repo_id).resolve()
    if dataset_root != expected_root:
        raise ValueError(f"dataset_root {dataset_root} does not match HF_LEROBOT_HOME/repo_id {expected_root}")

    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        repack_transforms=_evaluation_repack(),
        sample_kwargs={},
    )
    model = policy._model  # noqa: SLF001
    model.eval()
    graphdef, state = nnx.split(model)

    noise_bank = jax.random.normal(
        jax.random.key(args.noise_seed),
        (args.noise_samples, model.action_horizon, model.action_dim),
        dtype=jnp.float32,
    )
    action_timestep = float(args.action_timestep)
    noise_samples = int(args.noise_samples)

    def _compute_probe_features(state, observation):
        module = nnx.merge(graphdef, state)
        observation = model_api.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = module.embed_prefix(observation)
        batch_size = prefix_tokens.shape[0]

        repeated_observation = jax.tree.map(
            lambda value: jnp.repeat(value, noise_samples, axis=0),
            observation,
        )
        repeated_prefix_tokens = jnp.repeat(prefix_tokens, noise_samples, axis=0)
        repeated_prefix_mask = jnp.repeat(prefix_mask, noise_samples, axis=0)
        repeated_prefix_ar_mask = prefix_ar_mask
        repeated_noise = jnp.tile(noise_bank, (batch_size, 1, 1))
        timestep = jnp.full((batch_size * noise_samples,), action_timestep, dtype=jnp.float32)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = module.embed_suffix(
            repeated_observation,
            repeated_noise,
            timestep,
        )
        input_mask = jnp.concatenate([repeated_prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([repeated_prefix_ar_mask, suffix_ar_mask], axis=0)
        attention_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = module.PaliGemma.llm(
            [repeated_prefix_tokens, suffix_tokens],
            mask=attention_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        prefix_out = prefix_out.reshape(batch_size, noise_samples, prefix_out.shape[1], prefix_out.shape[2])[:, 0]
        prefix_mask_f = prefix_mask.astype(jnp.float32)[..., None]
        prefix_mean = jnp.sum(prefix_out.astype(jnp.float32) * prefix_mask_f, axis=1) / jnp.maximum(
            jnp.sum(prefix_mask_f, axis=1), 1.0
        )
        last_index = jnp.maximum(jnp.sum(prefix_mask, axis=1).astype(jnp.int32) - 1, 0)
        prefix_last = jnp.take_along_axis(prefix_out, last_index[:, None, None], axis=1)[:, 0].astype(jnp.float32)

        action_hidden = suffix_out[:, -module.action_horizon :].astype(jnp.float32)
        action_hidden = action_hidden.reshape(
            batch_size,
            noise_samples,
            module.action_horizon,
            action_hidden.shape[-1],
        )
        action_hidden_per_noise = jnp.mean(action_hidden, axis=2)
        action_hidden_mean = jnp.mean(action_hidden_per_noise, axis=1)
        action_hidden_noise_std = jnp.mean(jnp.std(action_hidden_per_noise, axis=1), axis=-1)
        return {
            "prefix_mean": prefix_mean,
            "prefix_last": prefix_last,
            "action_hidden": action_hidden_mean,
            "action_hidden_noise_std": action_hidden_noise_std,
        }

    compute_fn = jax.jit(_compute_probe_features)

    episode_lengths = _read_episode_lengths(dataset_root / "meta")
    specs, audit = build_virtual_sample_specs(
        episode_lengths,
        negative_stride=args.negative_stride,
        copy_frames=args.copy_frames,
        positive_tail=args.positive_tail,
        subtask4_tail=args.subtask4_tail,
        hard_negative_frames=args.hard_negative_frames,
        split_seed=args.split_seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    if args.max_groups_per_split:
        retained_groups: set[int] = set()
        for split in (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST):
            split_groups = sorted({spec.group_index for spec in specs if spec.split == split})
            retained_groups.update(split_groups[: args.max_groups_per_split])
        specs = [spec for spec in specs if spec.group_index in retained_groups]
        audit["max_groups_per_split"] = args.max_groups_per_split
        audit["retained_group_count"] = len(retained_groups)
        audit["counts_after_group_limit"] = {
            name: sum(spec.split == split for spec in specs)
            for name, split in (("train", SPLIT_TRAIN), ("val", SPLIT_VAL), ("test", SPLIT_TEST))
        }
    if args.num_shards > 1:
        specs = [spec for spec in specs if spec.group_index % args.num_shards == args.shard_index]
        audit["num_shards"] = args.num_shards
        audit["shard_index"] = args.shard_index
        audit["shard_group_count"] = len({spec.group_index for spec in specs})
        audit["shard_sample_count"] = len(specs)
    if not specs:
        raise ValueError("sample selection is empty")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id)
    tasks = dataset_meta.tasks
    episode_from = dataset.episode_data_index["from"]

    output_arrays: dict[str, list[np.ndarray]] = {
        "prefix_mean": [],
        "prefix_last": [],
        "action_hidden": [],
        "action_hidden_noise_std": [],
    }
    metadata: dict[str, list[int | float]] = {
        "logical_episode": [],
        "source_episode": [],
        "source_frame": [],
        "logical_frame": [],
        "task_index": [],
        "group_index": [],
        "split": [],
        "completion": [],
        "progress": [],
        "kind": [],
        "pair_id": [],
    }

    for batch_start in range(0, len(specs), args.batch_size):
        valid_specs = specs[batch_start : batch_start + args.batch_size]
        transformed_items: list[dict[str, Any]] = []
        for spec in valid_specs:
            dataset_index = int(episode_from[spec.source_episode]) + spec.source_frame
            sample = dict(dataset[dataset_index])
            if spec.task_index not in tasks:
                raise ValueError(f"task_index {spec.task_index} is absent from dataset metadata")
            sample["prompt"] = tasks[spec.task_index]
            transformed_items.append(policy._input_transform(sample))  # noqa: SLF001
        valid_count = len(transformed_items)
        while len(transformed_items) < args.batch_size:
            transformed_items.append(jax.tree.map(lambda value: np.array(value, copy=True), transformed_items[-1]))
        batched_inputs = jax.tree.map(
            lambda *values: jnp.asarray(np.stack([np.asarray(value) for value in values], axis=0)),
            *transformed_items,
        )
        observation = model_api.Observation.from_dict(batched_inputs)
        batch_features = jax.block_until_ready(compute_fn(state, observation))
        for name, values in batch_features.items():
            output_arrays[name].append(np.asarray(values[:valid_count]))
        for spec in valid_specs:
            for field, field_values in metadata.items():
                field_values.append(getattr(spec, field))
        completed = batch_start + valid_count
        if batch_start == 0 or completed % (args.batch_size * 25) == 0 or completed == len(specs):
            print(f"Extracted {completed}/{len(specs)} samples")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_name = (
        f"features_shard_{args.shard_index:03d}_of_{args.num_shards:03d}.npz"
        if args.num_shards > 1
        else "features.npz"
    )
    feature_path = args.output_dir / feature_name
    _savez_compressed_mount_safe(
        feature_path,
        **{
            name: np.concatenate(chunks, axis=0).astype(np.float32 if name == "action_hidden_noise_std" else np.float16)
            for name, chunks in output_arrays.items()
        },
        logical_episode=np.asarray(metadata["logical_episode"], dtype=np.int32),
        source_episode=np.asarray(metadata["source_episode"], dtype=np.int32),
        source_frame=np.asarray(metadata["source_frame"], dtype=np.int32),
        logical_frame=np.asarray(metadata["logical_frame"], dtype=np.int32),
        task_index=np.asarray(metadata["task_index"], dtype=np.int8),
        group_index=np.asarray(metadata["group_index"], dtype=np.int32),
        split=np.asarray(metadata["split"], dtype=np.int8),
        completion=np.asarray(metadata["completion"], dtype=np.int8),
        progress=np.asarray(metadata["progress"], dtype=np.float32),
        kind=np.asarray(metadata["kind"], dtype=np.int8),
        pair_id=np.asarray(metadata["pair_id"], dtype=np.int64),
    )
    manifest = {
        "config_name": args.config_name,
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "feature_file": str(feature_path),
        "action_timestep": action_timestep,
        "noise_samples": noise_samples,
        "noise_seed": args.noise_seed,
        "sample_audit": audit,
    }
    manifest_name = (
        f"extract_manifest_shard_{args.shard_index:03d}_of_{args.num_shards:03d}.json"
        if args.num_shards > 1
        else "extract_manifest.json"
    )
    (args.output_dir / manifest_name).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved features to {feature_path}")
    return feature_path


def merge_feature_shards(args: argparse.Namespace) -> Path:
    """Validates and combines independently extracted group shards."""

    if args.num_shards <= 1:
        raise ValueError("merge requires --num-shards greater than 1")
    shard_paths = [
        args.output_dir / f"features_shard_{index:03d}_of_{args.num_shards:03d}.npz"
        for index in range(args.num_shards)
    ]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing feature shards: {missing}")

    chunks: dict[str, list[np.ndarray]] = {}
    shard_rows: list[int] = []
    expected_keys: tuple[str, ...] | None = None
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as values:
            keys = tuple(values.files)
            if expected_keys is None:
                expected_keys = keys
            elif keys != expected_keys:
                raise ValueError(f"feature keys differ in {path}")
            row_count = len(values["completion"])
            shard_rows.append(row_count)
            for key in keys:
                array = np.asarray(values[key])
                if len(array) != row_count:
                    raise ValueError(f"row count mismatch for {key} in {path}")
                chunks.setdefault(key, []).append(array)

    assert expected_keys is not None
    merged = {key: np.concatenate(chunks[key], axis=0) for key in expected_keys}
    identity = np.stack(
        [merged["logical_episode"], merged["source_episode"], merged["source_frame"]],
        axis=1,
    )
    if len(np.unique(identity, axis=0)) != len(identity):
        raise ValueError("duplicate logical/source frame identities found across shards")
    for key in ("prefix_mean", "prefix_last", "action_hidden", "action_hidden_noise_std"):
        if not np.isfinite(merged[key]).all():
            raise ValueError(f"non-finite values found in merged {key}")

    feature_path = args.output_dir / "features.npz"
    _savez_compressed_mount_safe(feature_path, **merged)
    split_counts = np.bincount(merged["split"].astype(np.int64), minlength=3)
    label_counts = np.bincount(merged["completion"].astype(np.int64), minlength=2)
    manifest = {
        "feature_file": str(feature_path),
        "num_shards": args.num_shards,
        "shards": [str(path) for path in shard_paths],
        "shard_rows": shard_rows,
        "total_rows": len(identity),
        "split_counts": split_counts.tolist(),
        "label_counts": label_counts.tolist(),
    }
    (args.output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Merged {len(shard_paths)} shards ({len(identity)} rows) into {feature_path}")
    return feature_path


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    result = np.empty_like(logits)
    nonnegative = logits >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-logits[nonnegative]))
    exp_logits = np.exp(logits[~nonnegative])
    result[~nonnegative] = exp_logits / (1.0 + exp_logits)
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _roc_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    positive_count = int(np.sum(target))
    negative_count = len(target) - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = _average_ranks(np.asarray(score))
    return float(
        (np.sum(ranks[target]) - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)
    )


def _average_precision(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.int8)
    positive_count = int(np.sum(target))
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="mergesort")
    sorted_target = target[order]
    precision = np.cumsum(sorted_target) / np.arange(1, len(target) + 1)
    return float(np.sum(precision * sorted_target) / positive_count)


def _best_f1_threshold(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.int8)
    order = np.argsort(-score, kind="mergesort")
    sorted_target = target[order]
    tp = np.cumsum(sorted_target)
    fp = np.cumsum(1 - sorted_target)
    fn = int(np.sum(target)) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=np.float64), where=denominator > 0)
    best = int(np.argmax(f1))
    return float(score[order[best]])


def binary_metrics(target: np.ndarray, score: np.ndarray, *, threshold: float) -> dict[str, float]:
    target = np.asarray(target, dtype=np.int8)
    score = np.asarray(score, dtype=np.float64)
    prediction = score >= threshold
    positive = target == 1
    negative = ~positive
    tp = int(np.sum(prediction & positive))
    fp = int(np.sum(prediction & negative))
    fn = int(np.sum(~prediction & positive))
    tn = int(np.sum(~prediction & negative))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    clipped = np.clip(score, 1e-7, 1 - 1e-7)
    nll = float(-np.mean(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)))
    prevalence = float(np.mean(target))
    entropy = -prevalence * math.log(max(prevalence, 1e-12)) - (1 - prevalence) * math.log(max(1 - prevalence, 1e-12))
    return {
        "frame_count": float(len(target)),
        "positive_count": float(np.sum(positive)),
        "negative_count": float(np.sum(negative)),
        "prevalence": prevalence,
        "auroc": _roc_auc(target, score),
        "auprc": _average_precision(target, score),
        "nll": nll,
        "brier": float(np.mean((score - target) ** 2)),
        "usable_information_bits": float((entropy - nll) / math.log(2.0)),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_positive_rate": fp / max(fp + tn, 1),
    }


def progress_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    residual = prediction - target
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    pearson = float(np.corrcoef(target, prediction)[0, 1]) if np.std(prediction) > 0 else 0.0
    target_ranks = _average_ranks(target)
    prediction_ranks = _average_ranks(prediction)
    spearman = float(np.corrcoef(target_ranks, prediction_ranks)[0, 1]) if np.std(prediction_ranks) > 0 else 0.0
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else 0.0,
        "pearson": pearson,
        "spearman": spearman,
    }


def _standardize(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = np.mean(train, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(train, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-5, 1.0, std)
    return tuple(((values.astype(np.float32) - mean) / std).astype(np.float32) for values in (train, *others))


def _fit_logistic_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    l2_grid: tuple[float, ...],
    max_iter: int,
) -> tuple[np.ndarray, float, float]:
    from scipy import optimize  # noqa: PLC0415

    train_y = train_y.astype(np.float64)
    best: tuple[float, float, np.ndarray] | None = None
    initial = np.zeros(train_x.shape[1] + 1, dtype=np.float64)
    for l2 in l2_grid:

        def objective(parameters: np.ndarray, regularization: float = l2) -> tuple[float, np.ndarray]:
            weights = parameters[:-1]
            bias = parameters[-1]
            logits = train_x @ weights + bias
            loss = np.mean(np.logaddexp(0.0, logits) - train_y * logits) + 0.5 * regularization * np.sum(weights**2)
            probability = _sigmoid(logits)
            gradient_w = train_x.T @ (probability - train_y) / len(train_y) + regularization * weights
            gradient_b = float(np.mean(probability - train_y))
            return float(loss), np.concatenate([gradient_w, [gradient_b]])

        fitted = optimize.minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "ftol": 1e-9},
        )
        val_score = _sigmoid(val_x @ fitted.x[:-1] + fitted.x[-1])
        val_ap = _average_precision(val_y, val_score)
        candidate = (val_ap, -float(fitted.fun), fitted.x.copy())
        if best is None or candidate[:2] > best[:2]:
            best = candidate
        initial = fitted.x
    assert best is not None
    selected = best[2]
    return selected[:-1], float(selected[-1]), float(best[0])


def _fit_ridge_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    l2_grid: tuple[float, ...],
    max_iter: int,
) -> tuple[np.ndarray, float, float]:
    from scipy import optimize  # noqa: PLC0415

    train_y = train_y.astype(np.float64)
    best: tuple[float, np.ndarray] | None = None
    initial = np.zeros(train_x.shape[1] + 1, dtype=np.float64)
    for l2 in l2_grid:

        def objective(parameters: np.ndarray, regularization: float = l2) -> tuple[float, np.ndarray]:
            weights = parameters[:-1]
            bias = parameters[-1]
            residual = train_x @ weights + bias - train_y
            loss = 0.5 * np.mean(residual**2) + 0.5 * regularization * np.sum(weights**2)
            gradient_w = train_x.T @ residual / len(train_y) + regularization * weights
            gradient_b = float(np.mean(residual))
            return float(loss), np.concatenate([gradient_w, [gradient_b]])

        fitted = optimize.minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "ftol": 1e-10},
        )
        val_prediction = np.clip(val_x @ fitted.x[:-1] + fitted.x[-1], 0.0, 1.0)
        val_mae = float(np.mean(np.abs(val_prediction - val_y)))
        if best is None or val_mae < best[0]:
            best = (val_mae, fitted.x.copy())
        initial = fitted.x
    assert best is not None
    selected = best[1]
    return selected[:-1], float(selected[-1]), float(best[0])


def _fit_probe_jax(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    l2_grid: tuple[float, ...],
    max_iter: int,
    learning_rate: float,
    objective_name: str,
) -> tuple[np.ndarray, float, float]:
    """Fits a full-batch linear probe on one GPU with deterministic Adam."""

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import optax  # noqa: PLC0415

    device_train_x = jax.device_put(train_x.astype(np.float32, copy=False))
    device_train_y = jax.device_put(train_y.astype(np.float32, copy=False))
    parameters = {
        "weights": jnp.zeros(train_x.shape[1], dtype=jnp.float32),
        "bias": jnp.zeros((), dtype=jnp.float32),
    }
    optimizer = optax.adam(learning_rate)

    def loss_fn(params, regularization):
        prediction = device_train_x @ params["weights"] + params["bias"]
        if objective_name == "logistic":
            data_loss = jnp.mean(jnp.logaddexp(0.0, prediction) - device_train_y * prediction)
        elif objective_name == "ridge":
            data_loss = 0.5 * jnp.mean(jnp.square(prediction - device_train_y))
        else:
            raise ValueError(f"unknown objective: {objective_name}")
        return data_loss + 0.5 * regularization * jnp.sum(jnp.square(params["weights"]))

    @jax.jit
    def step(params, optimizer_state, regularization):
        loss, gradients = jax.value_and_grad(loss_fn)(params, regularization)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        return optax.apply_updates(params, updates), optimizer_state, loss

    best: tuple[float, float, np.ndarray, float] | None = None
    for l2 in l2_grid:
        optimizer_state = optimizer.init(parameters)
        loss = jnp.asarray(float("nan"), dtype=jnp.float32)
        for _ in range(max_iter):
            parameters, optimizer_state, loss = step(parameters, optimizer_state, np.float32(l2))
        weights = np.asarray(jax.device_get(parameters["weights"]), dtype=np.float64)
        bias = float(jax.device_get(parameters["bias"]))
        raw_val = val_x @ weights + bias
        if objective_name == "logistic":
            score = _sigmoid(raw_val)
            selection = _average_precision(val_y, score)
            candidate = (selection, -float(jax.device_get(loss)), weights.copy(), bias)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        else:
            prediction = np.clip(raw_val, 0.0, 1.0)
            selection = float(np.mean(np.abs(prediction - val_y)))
            candidate = (-selection, -float(jax.device_get(loss)), weights.copy(), bias)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    assert best is not None
    selected_metric = best[0] if objective_name == "logistic" else -best[0]
    return best[2], best[3], float(selected_metric)


def paired_prompt_accuracy(target: np.ndarray, score: np.ndarray, pair_id: np.ndarray) -> dict[str, float]:
    correct = 0.0
    count = 0
    margins: list[float] = []
    for raw_pair in np.unique(pair_id[pair_id >= 0]):
        rows = np.flatnonzero(pair_id == raw_pair)
        positive_rows = rows[target[rows] == 1]
        negative_rows = rows[target[rows] == 0]
        if len(positive_rows) != 1 or len(negative_rows) != 1:
            continue
        margin = float(score[positive_rows[0]] - score[negative_rows[0]])
        margins.append(margin)
        correct += 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
        count += 1
    return {
        "pair_count": float(count),
        "accuracy": correct / max(count, 1),
        "mean_positive_minus_negative_margin": float(np.mean(margins)) if margins else float("nan"),
    }


def _episode_event_metrics(
    target: np.ndarray,
    score: np.ndarray,
    logical_episode: np.ndarray,
    logical_frame: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    early = 0
    detected = 0
    episode_count = 0
    delays: list[int] = []
    for episode in np.unique(logical_episode):
        rows = np.flatnonzero(logical_episode == episode)
        rows = rows[np.argsort(logical_frame[rows])]
        positive_rows = rows[target[rows] == 1]
        if not len(positive_rows):
            continue
        onset_frame = int(np.min(logical_frame[positive_rows]))
        trigger_rows = rows[score[rows] >= threshold]
        episode_count += 1
        if len(trigger_rows) and int(logical_frame[trigger_rows[0]]) < onset_frame:
            early += 1
        valid_trigger = trigger_rows[logical_frame[trigger_rows] >= onset_frame]
        if len(valid_trigger):
            detected += 1
            delays.append(int(logical_frame[valid_trigger[0]]) - onset_frame)
    return {
        "episode_count": float(episode_count),
        "early_trigger_episode_rate": early / max(episode_count, 1),
        "detection_episode_rate": detected / max(episode_count, 1),
        "mean_detection_delay_logical_frames": float(np.mean(delays)) if delays else float("nan"),
    }


def _representation_arrays(values: Any) -> dict[str, np.ndarray]:
    prefix_mean = np.asarray(values["prefix_mean"], dtype=np.float32)
    prefix_last = np.asarray(values["prefix_last"], dtype=np.float32)
    action_hidden = np.asarray(values["action_hidden"], dtype=np.float32)
    return {
        "prefix_mean": prefix_mean,
        "prefix_last": prefix_last,
        "prefix_mean_last": np.concatenate([prefix_mean, prefix_last], axis=1),
        "action_hidden": action_hidden,
        "prefix_action": np.concatenate([prefix_last, action_hidden], axis=1),
    }


def analyze_features(args: argparse.Namespace, feature_path: Path | None = None) -> Path:
    """Fits validation-selected probes and writes an untouched-test summary."""

    feature_path = feature_path or args.feature_file
    if feature_path is None or not feature_path.is_file():
        raise FileNotFoundError(f"feature file not found: {feature_path}")
    with np.load(feature_path, allow_pickle=False) as values:
        representations = _representation_arrays(values)
        target = np.asarray(values["completion"], dtype=np.int8)
        progress = np.asarray(values["progress"], dtype=np.float32)
        split = np.asarray(values["split"], dtype=np.int8)
        kind = np.asarray(values["kind"], dtype=np.int8)
        pair_id = np.asarray(values["pair_id"], dtype=np.int64)
        logical_episode = np.asarray(values["logical_episode"], dtype=np.int32)
        logical_frame = np.asarray(values["logical_frame"], dtype=np.int32)
        group_index = np.asarray(values["group_index"], dtype=np.int32)
        noise_std = np.asarray(values["action_hidden_noise_std"], dtype=np.float32)

    train_mask = split == SPLIT_TRAIN
    val_mask = split == SPLIT_VAL
    test_mask = split == SPLIT_TEST
    if min(np.sum(train_mask), np.sum(val_mask), np.sum(test_mask)) == 0:
        raise ValueError("feature file must contain non-empty train/val/test splits")

    summary: dict[str, Any] = {
        "feature_file": str(feature_path.resolve()),
        "split_counts": {
            "train": int(np.sum(train_mask)),
            "val": int(np.sum(val_mask)),
            "test": int(np.sum(test_mask)),
        },
        "action_hidden_noise_std": {
            "mean": float(np.mean(noise_std)),
            "p95": float(np.quantile(noise_std, 0.95)),
            "max": float(np.max(noise_std)),
        },
        "representations": {},
    }
    test_scores: dict[str, np.ndarray] = {}

    for name, feature in representations.items():
        print(f"Fitting probes for {name} ({feature.shape[1]} dims)")
        train_x, val_x, test_x = _standardize(feature[train_mask], feature[val_mask], feature[test_mask])

        if getattr(args, "probe_backend", "scipy") == "jax":
            completion_w, completion_b, val_ap = _fit_probe_jax(
                train_x,
                target[train_mask],
                val_x,
                target[val_mask],
                l2_grid=tuple(args.l2_grid),
                max_iter=args.max_iter,
                learning_rate=args.probe_learning_rate,
                objective_name="logistic",
            )
        else:
            completion_w, completion_b, val_ap = _fit_logistic_probe(
                train_x,
                target[train_mask],
                val_x,
                target[val_mask],
                l2_grid=tuple(args.l2_grid),
                max_iter=args.max_iter,
            )
        val_completion_score = _sigmoid(val_x @ completion_w + completion_b)
        test_completion_score = _sigmoid(test_x @ completion_w + completion_b)
        completion_threshold = _best_f1_threshold(target[val_mask], val_completion_score)
        test_scores[name] = test_completion_score

        if getattr(args, "probe_backend", "scipy") == "jax":
            progress_w, progress_b, val_progress_mae = _fit_probe_jax(
                train_x,
                progress[train_mask],
                val_x,
                progress[val_mask],
                l2_grid=tuple(args.l2_grid),
                max_iter=args.max_iter,
                learning_rate=args.probe_learning_rate,
                objective_name="ridge",
            )
        else:
            progress_w, progress_b, val_progress_mae = _fit_ridge_probe(
                train_x,
                progress[train_mask],
                val_x,
                progress[val_mask],
                l2_grid=tuple(args.l2_grid),
                max_iter=args.max_iter,
            )
        val_progress_prediction = np.clip(val_x @ progress_w + progress_b, 0.0, 1.0)
        test_progress_prediction = np.clip(test_x @ progress_w + progress_b, 0.0, 1.0)
        progress_boundary_threshold = _best_f1_threshold(target[val_mask], val_progress_prediction)

        test_target = target[test_mask]
        test_kind = kind[test_mask]
        hard_mask = np.logical_or(test_target == 1, test_kind == KIND_HARD_NEGATIVE)
        easy_mask = np.logical_or(test_target == 1, test_kind == KIND_ORDINARY_NEGATIVE)
        row = {
            "completion_probe": {
                "selected_val_auprc": val_ap,
                "overall": binary_metrics(test_target, test_completion_score, threshold=completion_threshold),
                "hard_boundary": binary_metrics(
                    test_target[hard_mask], test_completion_score[hard_mask], threshold=completion_threshold
                ),
                "easy_negative": binary_metrics(
                    test_target[easy_mask], test_completion_score[easy_mask], threshold=completion_threshold
                ),
                "paired_prompt": paired_prompt_accuracy(test_target, test_completion_score, pair_id[test_mask]),
                "episode_events": _episode_event_metrics(
                    test_target,
                    test_completion_score,
                    logical_episode[test_mask],
                    logical_frame[test_mask],
                    threshold=completion_threshold,
                ),
            },
            "progress_probe": {
                "selected_val_mae": val_progress_mae,
                "regression": progress_metrics(progress[test_mask], test_progress_prediction),
                "as_boundary_overall": binary_metrics(
                    test_target, test_progress_prediction, threshold=progress_boundary_threshold
                ),
                "as_boundary_hard": binary_metrics(
                    test_target[hard_mask],
                    test_progress_prediction[hard_mask],
                    threshold=progress_boundary_threshold,
                ),
                "paired_prompt": paired_prompt_accuracy(test_target, test_progress_prediction, pair_id[test_mask]),
                "episode_events": _episode_event_metrics(
                    test_target,
                    test_progress_prediction,
                    logical_episode[test_mask],
                    logical_frame[test_mask],
                    threshold=progress_boundary_threshold,
                ),
            },
        }
        summary["representations"][name] = row

    # Group bootstrap of the most decision-relevant metric.  Mean pooling is
    # the primary prefix baseline; the reference can be changed explicitly
    # without refitting or inspecting test labels during model selection.
    reference_name = getattr(args, "bootstrap_reference", "prefix_mean")
    reference_score = test_scores[reference_name]
    test_groups = group_index[test_mask]
    test_target = target[test_mask]
    test_kind = kind[test_mask]
    rng = np.random.default_rng(args.bootstrap_seed)
    deltas: dict[str, Any] = {}
    for name, score in test_scores.items():
        if name == reference_name:
            continue
        samples: list[float] = []
        unique_groups = np.unique(test_groups)
        for _ in range(args.bootstrap_samples):
            selected_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            selected_rows = np.concatenate([np.flatnonzero(test_groups == group) for group in selected_groups])
            selected_hard = np.logical_or(
                test_target[selected_rows] == 1,
                test_kind[selected_rows] == KIND_HARD_NEGATIVE,
            )
            rows = selected_rows[selected_hard]
            samples.append(
                _average_precision(test_target[rows], score[rows])
                - _average_precision(test_target[rows], reference_score[rows])
            )
        observed_hard = np.logical_or(test_target == 1, test_kind == KIND_HARD_NEGATIVE)
        deltas[name] = {
            "metric": f"hard_boundary_auprc_delta_vs_{reference_name}",
            "observed": _average_precision(test_target[observed_hard], score[observed_hard])
            - _average_precision(test_target[observed_hard], reference_score[observed_hard]),
            "bootstrap_ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        }
    summary["group_bootstrap_deltas"] = deltas

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "probe_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved probe summary to {summary_path}")
    return summary_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("extract", "merge", "analyze", "all"))
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--hf-lerobot-home", type=Path, default=DEFAULT_HF_LEROBOT_HOME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--negative-stride", type=int, default=15)
    parser.add_argument("--copy-frames", type=int, default=5)
    parser.add_argument("--positive-tail", type=int, default=5)
    parser.add_argument("--subtask4-tail", type=int, default=10)
    parser.add_argument("--hard-negative-frames", type=int, default=60)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--action-timestep", type=float, default=1.0)
    parser.add_argument("--noise-samples", type=int, default=4)
    parser.add_argument("--noise-seed", type=int, default=20260816)
    parser.add_argument("--l2-grid", type=float, nargs="+", default=(1e-4, 1e-3, 1e-2))
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    parser.add_argument(
        "--bootstrap-reference",
        choices=("prefix_mean", "prefix_last", "prefix_mean_last", "action_hidden", "prefix_action"),
        default="prefix_mean",
    )
    parser.add_argument("--probe-backend", choices=("scipy", "jax"), default="scipy")
    parser.add_argument("--probe-learning-rate", type=float, default=0.03)
    parser.add_argument(
        "--max-groups-per-split",
        type=int,
        default=0,
        help="Limit whole groups per split for extraction smoke tests; 0 keeps the full dataset.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.noise_samples <= 0 or args.max_iter <= 0:
        parser.error("batch-size, noise-samples and max-iter must be positive")
    if not 0.0 <= args.action_timestep <= 1.0:
        parser.error("action-timestep must lie in [0, 1]")
    if not args.l2_grid or any(value < 0 for value in args.l2_grid):
        parser.error("l2-grid must contain non-negative values")
    if args.bootstrap_samples <= 0:
        parser.error("bootstrap-samples must be positive")
    if args.probe_learning_rate <= 0:
        parser.error("probe-learning-rate must be positive")
    if args.max_groups_per_split < 0:
        parser.error("max-groups-per-split must be non-negative")
    if args.num_shards <= 0:
        parser.error("num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must lie in [0, num-shards)")
    if args.command == "all" and args.num_shards != 1:
        parser.error("all cannot run individual shards; use extract, then merge and analyze")
    return args


def main() -> int:
    args = _parse_args()
    try:
        feature_path: Path | None = None
        if args.command in ("extract", "all"):
            feature_path = extract_features(args)
        elif args.command == "merge":
            feature_path = merge_feature_shards(args)
        if args.command in ("analyze", "all"):
            analyze_features(args, feature_path)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
