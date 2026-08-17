"""Extract the sealed three-prefix cache for temporal completion training.

This command is intentionally read-only with respect to both source datasets
and the clean checkpoint.  It expands the canonical rows from a sealed
temporal manifest, de-duplicates exact ``(episode, frame, logical prompt)``
requests, evaluates the clean pi0.5 prefix in deterministic inference mode,
then reassembles the features in canonical ``[row, oldest..current, dim]``
order.

The supervision fields on a row are never consulted while selecting or
extracting features.  In particular, a post-boundary source frame belonging
to the next raw subtask is still evaluated with the previous logical task's
prompt when the manifest requests it.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from openpi.training import temporal_completion_data as temporal_data
from openpi.training import temporal_completion_features as temporal_features
from openpi.training import temporal_completion_preprocess as temporal_preprocess

DEFAULT_CONFIG_NAME = "pi05_730_breakfast_subtasks"
DEFAULT_CHECKPOINT = Path(
    "/mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999"
)
DEFAULT_DATASET_ROOT = Path("/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730")
DEFAULT_HF_LEROBOT_HOME = Path("/mnt/data/dataset/ei/huggingface")
DEFAULT_MANIFEST = Path("/mnt/data/models/wyt/split_manifests/agilex_make_breakfast_temporal_completion_v1.json")
DEFAULT_OUTPUT = Path("/mnt/data/models/wyt/evaluations/temporal_completion_prefix_features_v2/features.npz")


@dataclasses.dataclass(frozen=True, order=True)
class PrefixFeatureKey:
    """One exact model input identity, including the logical prompt text."""

    source_episode_id: int
    source_frame_index: int
    prompt: str
    # The model input identity is the exact prompt text.  Keep the logical
    # index for audits without letting two identical prompt strings defeat
    # de-duplication of the required (episode, frame, prompt) tuple.
    prompt_index: int = dataclasses.field(compare=False)

    def __post_init__(self) -> None:
        if self.source_episode_id < 0 or self.source_frame_index < 0:
            raise ValueError("prefix feature source coordinates must be non-negative")
        if self.prompt_index not in range(temporal_data.TASKS_PER_TRAJECTORY):
            raise ValueError(f"prefix feature prompt_index must be in [0, 3], got {self.prompt_index}")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prefix feature prompt must be a non-empty string")


@dataclasses.dataclass(frozen=True)
class PrefixFeaturePlan:
    """Unique requests plus the canonical row/history reassembly indices."""

    keys: tuple[PrefixFeatureKey, ...]
    row_key_indices: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.row_key_indices)
        if indices.ndim != 2 or indices.shape[1] != temporal_data.TEMPORAL_HISTORY_STEPS:
            raise ValueError(f"row_key_indices must have shape [N, 3], got {indices.shape}")
        if indices.dtype.kind not in "iu":
            raise ValueError("row_key_indices must contain integer indices")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("prefix feature plan contains duplicate keys")
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= len(self.keys)):
            raise ValueError("row_key_indices refers outside the unique key table")


def metadata_files(dataset_root: str | os.PathLike[str]) -> tuple[Path, ...]:
    """Returns every regular file below ``meta/`` in deterministic order."""

    root = Path(dataset_root).resolve()
    meta = root / "meta"
    if not meta.is_dir():
        raise FileNotFoundError(f"dataset metadata directory not found: {meta}")
    files = tuple(sorted((path.resolve() for path in meta.rglob("*") if path.is_file()), key=Path.as_posix))
    if not files:
        raise ValueError(f"dataset metadata directory contains no regular files: {meta}")
    return files


def regular_files(root: str | os.PathLike[str]) -> tuple[Path, ...]:
    """Returns all regular files under a sealed directory in stable order."""

    resolved = Path(root).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"sealed directory not found: {resolved}")
    files = tuple(sorted((path.resolve() for path in resolved.rglob("*") if path.is_file()), key=Path.as_posix))
    if not files:
        raise ValueError(f"sealed directory contains no regular files: {resolved}")
    return files


def tree_fingerprint(root: str | os.PathLike[str]) -> str:
    """Content-fingerprints a directory using the shared fail-closed helper."""

    return temporal_data.fingerprint_files(regular_files(root))


def resolve_logical_prompts(tasks: Mapping[int, str]) -> dict[int, str]:
    """Selects the exact four logical task prompts from LeRobot metadata."""

    prompts: dict[int, str] = {}
    for prompt_index in range(temporal_data.TASKS_PER_TRAJECTORY):
        if prompt_index not in tasks:
            raise ValueError(f"dataset metadata has no prompt for logical task {prompt_index}")
        prompt = tasks[prompt_index]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"dataset metadata prompt {prompt_index} is not a non-empty string")
        prompts[prompt_index] = prompt
    return prompts


def build_prefix_feature_plan(
    rows: Sequence[temporal_data.TemporalSampleRow],
    prompts: Mapping[int, str],
) -> PrefixFeaturePlan:
    """Builds a label-independent, globally de-duplicated extraction plan."""

    if not rows:
        raise ValueError("cannot extract temporal features for an empty canonical row set")
    logical_prompts = resolve_logical_prompts(prompts)
    row_keys: list[tuple[PrefixFeatureKey, PrefixFeatureKey, PrefixFeatureKey]] = []
    unique_keys: set[PrefixFeatureKey] = set()
    for row in rows:
        # Deliberately use only immutable source references and prompt identity.
        # No label/sample_kind/boundary field participates in feature selection.
        prompt = logical_prompts[row.prompt_index]
        keys = tuple(
            PrefixFeatureKey(
                source_episode_id=int(episode_id),
                source_frame_index=int(frame_index),
                prompt_index=int(row.prompt_index),
                prompt=prompt,
            )
            for episode_id, frame_index in zip(
                row.source_episode_ids,
                row.source_frame_indices,
                strict=True,
            )
        )
        if len(keys) != temporal_data.TEMPORAL_HISTORY_STEPS:
            raise ValueError("each temporal sample must refer to exactly three prefix observations")
        typed_keys = (keys[0], keys[1], keys[2])
        row_keys.append(typed_keys)
        unique_keys.update(typed_keys)

    # Sorting makes extraction order independent of row labels and improves
    # locality when the source dataset is read from videos.
    ordered_keys = tuple(sorted(unique_keys))
    index_by_key = {key: index for index, key in enumerate(ordered_keys)}
    row_key_indices = np.asarray(
        [[index_by_key[key] for key in keys] for keys in row_keys],
        dtype=np.int64,
    )
    return PrefixFeaturePlan(keys=ordered_keys, row_key_indices=row_key_indices)


def assemble_prefix_history(
    plan: PrefixFeaturePlan,
    unique_features: np.ndarray,
    *,
    storage_dtype: np.dtype[Any] | type[np.floating[Any]] = np.float16,
) -> np.ndarray:
    """Reassembles unique FP32 features into canonical oldest-to-current rows."""

    features = np.asarray(unique_features)
    if features.ndim != 2 or features.shape[0] != len(plan.keys) or features.shape[1] <= 0:
        raise ValueError(f"unique_features must have shape [{len(plan.keys)}, D], got {features.shape}")
    if features.dtype != np.float32:
        raise ValueError(f"model prefix features must be FP32 before cache storage, got {features.dtype}")
    if not np.isfinite(features).all():
        raise ValueError("model prefix features contain non-finite values")
    dtype = np.dtype(storage_dtype)
    if dtype not in temporal_features.SUPPORTED_FEATURE_DTYPES:
        raise ValueError(f"cache storage dtype must be float16 or float32, got {dtype}")
    history = features[np.asarray(plan.row_key_indices, dtype=np.int64)]
    expected_shape = (len(plan.row_key_indices), temporal_data.TEMPORAL_HISTORY_STEPS, features.shape[1])
    if history.shape != expected_shape:
        raise AssertionError(
            f"internal temporal history shape mismatch: expected {expected_shape}, got {history.shape}"
        )
    return history.astype(dtype, copy=False)


# Compatibility aliases keep the pure extractor tests and any existing local
# tooling on the one shared production implementation.
canonical_runtime_value = temporal_preprocess.canonical_runtime_value
implementation_fingerprint = temporal_preprocess.implementation_fingerprint
make_preprocess_fingerprint = temporal_preprocess.make_preprocess_fingerprint


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def assert_safe_output(output: Path, protected_roots: Sequence[Path]) -> None:
    resolved = output.resolve()
    for root in protected_roots:
        if _is_within(resolved, root):
            raise ValueError(f"refusing to write feature cache inside protected source/checkpoint root: {root}")


def _evaluation_repack() -> Any:
    # Keep heavy OpenPI/JAX imports out of module import so pure planning and
    # fingerprint tests run without accelerator dependencies.
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


def _read_manifest_fail_closed(path: Path) -> temporal_data.TemporalCompletionManifest:
    if not path.is_file():
        raise FileNotFoundError(f"sealed temporal manifest not found: {path}")
    raw = temporal_data.TemporalCompletionManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    subtask_fingerprint = temporal_data.fingerprint_files(metadata_files(raw.source_subtask_root))
    full_fingerprint = temporal_data.fingerprint_files(metadata_files(raw.source_full_root))
    return temporal_data.load_temporal_manifest(
        path,
        expected_subtask_metadata_fingerprint=subtask_fingerprint,
        expected_full_metadata_fingerprint=full_fingerprint,
    )


def _scalar_int(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def _validate_clean_runtime(config: Any, *, config_name: str, checkpoint: Path, manifest: Any) -> Any:
    if config.name != config_name or config_name != DEFAULT_CONFIG_NAME:
        raise ValueError(f"temporal v1 extraction is locked to clean config {DEFAULT_CONFIG_NAME!r}")
    if bool(getattr(getattr(config, "training_time_rtc", None), "enabled", False)):
        raise ValueError("clean temporal prefix extraction forbids TTRTC")
    completion_head = getattr(config.model, "completion_head", None)
    if completion_head is not None and bool(getattr(completion_head, "enabled", False)):
        raise ValueError("clean temporal prefix extraction forbids a completion head")
    if getattr(getattr(config, "completion", None), "stage", "disabled") != "disabled":
        raise ValueError("clean temporal prefix extraction requires completion training to be disabled")
    if not bool(getattr(config.model, "pi05", False)):
        raise ValueError("clean temporal prefix extraction requires pi0.5")

    params = (checkpoint / "params").resolve()
    configured_params = getattr(getattr(config, "weight_loader", None), "params_path", None)
    if configured_params is None or Path(configured_params).resolve() != params:
        raise ValueError(
            f"clean config weight_loader path {configured_params!r} does not match requested checkpoint {params}"
        )
    if not params.is_dir():
        raise FileNotFoundError(f"clean checkpoint params not found: {params}")
    if not (checkpoint / "assets").is_dir():
        raise FileNotFoundError(f"clean checkpoint assets not found: {checkpoint / 'assets'}")

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id != manifest.source_subtask_repo_id:
        raise ValueError(
            "clean config repo_id does not match sealed temporal manifest: "
            f"{data_config.repo_id!r} != {manifest.source_subtask_repo_id!r}"
        )
    if not bool(data_config.prompt_from_task):
        raise ValueError("clean source config must declare prompt_from_task=True")
    return data_config


def extract_temporal_features(args: argparse.Namespace) -> Path:
    """Restores the clean model and writes one immutable temporal cache."""

    # These imports initialize JAX/LeRobot and therefore belong only in the
    # real extraction path, after CLI validation.
    import flax.nnx as nnx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: PLC0415

    import openpi.models.model as model_api  # noqa: PLC0415
    from openpi.policies import policy_config  # noqa: PLC0415
    from openpi.training import config as training_config  # noqa: PLC0415

    manifest_path = args.manifest.resolve()
    dataset_root = args.dataset_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite sealed temporal feature cache: {output}")

    manifest = _read_manifest_fail_closed(manifest_path)
    if dataset_root != Path(manifest.source_subtask_root).resolve():
        raise ValueError(
            f"dataset_root {dataset_root} does not match sealed manifest root {manifest.source_subtask_root}"
        )
    assert_safe_output(
        output,
        [dataset_root, Path(manifest.source_full_root).resolve(), checkpoint, manifest_path.parent],
    )

    os.environ["HF_LEROBOT_HOME"] = str(args.hf_lerobot_home.resolve())
    config = training_config.get_config(args.config_name)
    data_config = _validate_clean_runtime(
        config,
        config_name=args.config_name,
        checkpoint=checkpoint,
        manifest=manifest,
    )
    configured_root = (
        Path(data_config.lerobot_home).resolve() / str(data_config.repo_id)
        if data_config.lerobot_home is not None
        else args.hf_lerobot_home.resolve() / str(data_config.repo_id)
    )
    if configured_root != dataset_root:
        raise ValueError(f"clean config resolves dataset to {configured_root}, not sealed root {dataset_root}")

    dataset_metadata = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=dataset_root)
    prompts = resolve_logical_prompts(dataset_metadata.tasks)
    ordered_prompts = tuple(prompts[index] for index in range(temporal_data.TASKS_PER_TRAJECTORY))
    if ordered_prompts != manifest.task_prompts:
        raise ValueError(
            "runtime task prompts differ from the sealed temporal manifest: "
            f"runtime={ordered_prompts!r}, manifest={manifest.task_prompts!r}"
        )
    rows = temporal_features.manifest_rows(manifest)
    plan = build_prefix_feature_plan(rows, prompts)
    if not plan.keys:
        raise ValueError("sealed temporal manifest produced no prefix feature requests")

    checkpoint_fingerprint = tree_fingerprint(checkpoint / "params")
    preprocess_fingerprint = temporal_preprocess.expected_preprocess_fingerprint(
        source_train_config=config,
        manifest=manifest,
        checkpoint_path=checkpoint,
    )

    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        repack_transforms=_evaluation_repack(),
        sample_kwargs={},
    )
    if bool(getattr(policy, "_is_pytorch_model", False)):
        raise ValueError("temporal v1 prefix extraction requires the JAX clean checkpoint")
    model = policy._model  # noqa: SLF001
    if not hasattr(model, "compute_prefix_feature"):
        raise ValueError("loaded clean pi0.5 model lacks compute_prefix_feature")
    model.eval()
    graphdef, state = nnx.split(model)

    def compute_prefix(state: Any, observation: model_api.Observation) -> Any:
        module = nnx.merge(graphdef, state)
        return module.compute_prefix_feature(jax.random.key(0), observation, train=False)

    compute_fn = jax.jit(compute_prefix)
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, root=dataset_root)
    episode_from = dataset.episode_data_index["from"]
    episode_to = dataset.episode_data_index["to"]
    feature_chunks: list[np.ndarray] = []

    for batch_start in range(0, len(plan.keys), args.batch_size):
        batch_keys = plan.keys[batch_start : batch_start + args.batch_size]
        if len(set(batch_keys)) != len(batch_keys):
            raise AssertionError("internal extraction batch contains duplicate source/frame/prompt keys")
        transformed: list[dict[str, Any]] = []
        for key in batch_keys:
            start = _scalar_int(episode_from[key.source_episode_id])
            stop = _scalar_int(episode_to[key.source_episode_id])
            dataset_index = start + key.source_frame_index
            if dataset_index < start or dataset_index >= stop:
                raise ValueError(
                    f"manifest source frame {key.source_episode_id}:{key.source_frame_index} is outside [{start}, {stop})"
                )
            sample = dict(dataset[dataset_index])
            if _scalar_int(sample["episode_index"]) != key.source_episode_id:
                raise ValueError("source dataset episode_index disagrees with manifest reference")
            if _scalar_int(sample["frame_index"]) != key.source_frame_index:
                raise ValueError("source dataset frame_index disagrees with manifest reference")

            # This assignment is deliberately before repack, AgileX, normalize,
            # resize, and tokenizer transforms.  It overrides the raw episode's
            # task even when a boundary row reads pixels from the next subtask.
            sample["prompt"] = key.prompt
            transformed.append(policy._input_transform(sample))  # noqa: SLF001

        batched_inputs = jax.tree.map(
            lambda *values: jnp.asarray(np.stack([np.asarray(value) for value in values], axis=0)),
            *transformed,
        )
        observation = model_api.Observation.from_dict(batched_inputs)
        batch_features = np.asarray(jax.block_until_ready(compute_fn(state, observation)))
        expected_shape = (len(batch_keys), int(model.prefix_feature_dim))
        if batch_features.shape != expected_shape:
            raise ValueError(f"compute_prefix_feature returned {batch_features.shape}, expected {expected_shape}")
        if batch_features.dtype != np.float32:
            raise ValueError(f"compute_prefix_feature must return FP32, got {batch_features.dtype}")
        if not np.isfinite(batch_features).all():
            raise ValueError("compute_prefix_feature returned non-finite values")
        feature_chunks.append(batch_features)

        completed = batch_start + len(batch_keys)
        if batch_start == 0 or completed % (args.batch_size * 25) == 0 or completed == len(plan.keys):
            print(f"Extracted {completed}/{len(plan.keys)} unique prompt-conditioned prefix features")

    unique_features = np.concatenate(feature_chunks, axis=0)
    storage_dtype = np.float16 if args.storage_dtype == "float16" else np.float32
    prefix_history = assemble_prefix_history(plan, unique_features, storage_dtype=storage_dtype)
    metadata = temporal_features.save_temporal_feature_cache(
        output,
        manifest=manifest,
        prefix_history=prefix_history,
        checkpoint_fingerprint=checkpoint_fingerprint,
        preprocess_fingerprint=preprocess_fingerprint,
        model_config_name=args.config_name,
        checkpoint_path=str(checkpoint),
    )
    print(
        "Saved immutable temporal feature cache "
        f"to {output} (rows={metadata.row_count}, unique={len(plan.keys)}, dim={metadata.feature_dim}, "
        f"dtype={prefix_history.dtype})"
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--hf-lerobot-home", type=Path, default=DEFAULT_HF_LEROBOT_HOME)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    return parser


def main() -> None:
    extract_temporal_features(_parser().parse_args())


if __name__ == "__main__":
    main()
