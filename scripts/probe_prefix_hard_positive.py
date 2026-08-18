"""Minimal hard-negative versus positive probe for frozen clean pi0.5 prefixes.

For every selected subtask episode with inclusive completion frame ``E``, the
script extracts only ``E-45, E-30, E-15, E`` under that episode's own prompt.
It compares a current-only logistic probe with a three-prefix temporal probe:

    current:  z_(E-15) -> 0, z_E -> 1
    temporal: [z_(E-45), z_(E-30), z_(E-15)] -> 0
              [z_(E-30), z_(E-15), z_E]     -> 1

The source dataset and checkpoint are read-only.  This deliberately has no
manifest, checkpoint hashing, action sampling, MLP, threshold search, or
oversampling; it is only a small representation-separability check.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import dataclasses
import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np

DEFAULT_DATASET_ROOT = Path("/mnt/data/dataset/ei/huggingface/modanqing/agilex_make_breakfast_subtask_730")
DEFAULT_HF_LEROBOT_HOME = Path("/mnt/data/dataset/ei/huggingface")
DEFAULT_CHECKPOINT = Path(
    "/mnt/data/models/wyt/checkpoints/pi05_730_breakfast_subtasks/breakfast_subtasks_bs64_50k/49999"
)
DEFAULT_OUTPUT = Path("/mnt/data/models/wyt/evaluations/hard_positive_prefix_probe/metrics.json")
DEFAULT_CONFIG_NAME = "pi05_730_breakfast_subtasks"
EXPECTED_REPO_ID = "modanqing/agilex_make_breakfast_subtask_730"
TASKS_PER_GROUP = 4
FRAME_OFFSETS = (-45, -30, -15, 0)
KNOWN_INCLUSIVE_END_FIELDS = (
    "completion_frame_index",
    "completion_frame",
    "terminal_frame_index",
    "terminal_frame",
    "end_frame_index",
    "end_frame",
)


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    episode_index: int
    group_index: int
    group_position: int
    task_index: int
    prompt: str
    length: int
    end_frame: int

    @property
    def requested_frames(self) -> tuple[int, int, int, int]:
        return tuple(self.end_frame + offset for offset in FRAME_OFFSETS)  # type: ignore[return-value]


@dataclasses.dataclass(frozen=True)
class DatasetAudit:
    episodes: tuple[EpisodeSpec, ...]
    eligible_group_ids: tuple[int, ...]
    excluded_short_by_task: Mapping[int, int]
    end_frame_source: str
    task_by_group_position: tuple[int, int, int, int]


@dataclasses.dataclass(frozen=True)
class _LogisticProbe:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(values, dtype=np.float32) - self.mean) / self.scale
        logits = standardized @ self.weight + self.bias
        logits = np.clip(logits, -80.0, 80.0)
        positive = (1.0 / (1.0 + np.exp(-logits))).astype(np.float64)
        return np.stack([1.0 - positive, positive], axis=1)


def _integer(value: Any, *, context: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{context} must be scalar, got shape {array.shape}")
    scalar = array.reshape(-1)[0]
    try:
        integer = int(scalar)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{context} must be an integer, got {scalar!r}") from error
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{context} must be numeric, got {scalar!r}") from error
    if not np.isfinite(numeric) or numeric != integer:
        raise ValueError(f"{context} must be an exact finite integer, got {scalar!r}")
    return integer


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain one JSON object")
            records.append(value)
    if not records:
        raise ValueError(f"{path} is empty")
    return records


def _task_prompts(dataset_root: Path) -> tuple[dict[int, str], dict[str, int]]:
    records = _read_jsonl(dataset_root / "meta" / "tasks.jsonl")
    by_index: dict[int, str] = {}
    by_prompt: dict[str, int] = {}
    for row_number, record in enumerate(records, start=1):
        if "task_index" not in record or "task" not in record:
            raise ValueError(f"tasks.jsonl row {row_number} lacks task_index or task")
        task_index = _integer(record["task_index"], context=f"tasks.jsonl row {row_number} task_index")
        prompt = str(record["task"])
        if not prompt.strip():
            raise ValueError(f"tasks.jsonl row {row_number} has an empty task prompt")
        if task_index in by_index or prompt in by_prompt:
            raise ValueError("tasks.jsonl contains duplicate task indices or prompt strings")
        by_index[task_index] = prompt
        by_prompt[prompt] = task_index
    return by_index, by_prompt


def _parquet_path(dataset_root: Path, info: Mapping[str, Any], episode_index: int) -> Path:
    data_path = info.get("data_path")
    chunks_size = info.get("chunks_size")
    if not isinstance(data_path, str) or not data_path:
        raise ValueError("meta/info.json must provide a non-empty data_path template")
    chunk_size = _integer(chunks_size, context="meta/info.json chunks_size")
    if chunk_size <= 0:
        raise ValueError("meta/info.json chunks_size must be positive")
    return dataset_root / data_path.format(
        episode_chunk=episode_index // chunk_size,
        episode_index=episode_index,
    )


def _task_index_from_episode_row(
    record: Mapping[str, Any],
    *,
    prompt_to_index: Mapping[str, int],
    context: str,
) -> int | None:
    if "task_index" in record:
        return _integer(record["task_index"], context=f"{context} task_index")
    tasks = record.get("tasks")
    if tasks is None:
        return None
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], str):
        raise ValueError(f"{context} tasks must contain exactly one prompt string")
    if tasks[0] not in prompt_to_index:
        raise ValueError(f"{context} prompt {tasks[0]!r} is absent from tasks.jsonl")
    return int(prompt_to_index[tasks[0]])


def _parquet_task_index(path: Path, *, episode_index: int) -> int:
    import pyarrow.parquet as pq  # noqa: PLC0415

    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    required = {"episode_index", "frame_index", "task_index"}
    missing = required.difference(parquet.schema.names)
    if missing:
        raise ValueError(f"{path} lacks required columns {sorted(missing)}")
    table = parquet.read(columns=["task_index"])
    values = np.asarray(table["task_index"].to_numpy(zero_copy_only=False))
    unique = np.unique(values)
    if len(unique) != 1:
        raise ValueError(f"episode {episode_index} must contain one task_index, got {unique.tolist()}")
    return _integer(unique[0], context=f"episode {episode_index} parquet task_index")


def _resolve_end_frame_field(
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    requested_field: str | None,
) -> tuple[str | None, str]:
    if requested_field is not None:
        absent = [index for index, row in enumerate(episode_rows) if requested_field not in row]
        if absent:
            raise ValueError(f"--end-frame-field {requested_field!r} is absent from episode rows {absent[:5]}")
        return requested_field, f"episodes.jsonl:{requested_field} (inclusive)"

    present = [field for field in KNOWN_INCLUSIVE_END_FIELDS if all(field in row for row in episode_rows)]
    if len(present) > 1:
        raise ValueError(f"multiple candidate inclusive end-frame fields found: {present}; choose --end-frame-field")
    if len(present) == 1:
        return present[0], f"episodes.jsonl:{present[0]} (inclusive)"
    return None, "episode last frame (dataset-confirmed manual inclusive endpoint)"


def _validate_selected_parquet(
    dataset_root: Path,
    info: Mapping[str, Any],
    spec: EpisodeSpec,
) -> None:
    import pyarrow.parquet as pq  # noqa: PLC0415

    path = _parquet_path(dataset_root, info, spec.episode_index)
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    required = {"episode_index", "frame_index", "task_index"}
    missing = required.difference(parquet.schema.names)
    if missing:
        raise ValueError(f"{path} lacks required columns {sorted(missing)}")
    table = parquet.read(columns=["episode_index", "frame_index", "task_index"])
    episode_values = np.asarray(table["episode_index"].to_numpy(zero_copy_only=False))
    frame_values = np.asarray(table["frame_index"].to_numpy(zero_copy_only=False))
    task_values = np.asarray(table["task_index"].to_numpy(zero_copy_only=False))
    if len(frame_values) != spec.length:
        raise ValueError(
            f"episode {spec.episode_index} metadata length {spec.length} != parquet rows {len(frame_values)}"
        )
    if not np.all(episode_values == spec.episode_index):
        raise ValueError(f"episode {spec.episode_index} parquet contains another episode_index")
    expected_frames = np.arange(spec.length, dtype=np.int64)
    try:
        integer_frames = frame_values.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"episode {spec.episode_index} has non-integer frame_index values") from error
    if not np.array_equal(frame_values, integer_frames) or not np.array_equal(integer_frames, expected_frames):
        raise ValueError(f"episode {spec.episode_index} frame_index is not exactly 0..{spec.length - 1}")
    unique_tasks = np.unique(task_values)
    if (
        len(unique_tasks) != 1
        or _integer(unique_tasks[0], context=f"episode {spec.episode_index} task_index") != spec.task_index
    ):
        raise ValueError(
            f"episode {spec.episode_index} parquet task_index {unique_tasks.tolist()} != metadata {spec.task_index}"
        )


def audit_dataset(args: argparse.Namespace) -> DatasetAudit:
    dataset_root = args.dataset_root.resolve()
    info = _read_json(dataset_root / "meta" / "info.json")
    episode_rows = _read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    prompts, prompt_to_index = _task_prompts(dataset_root)
    end_field, end_source = _resolve_end_frame_field(
        episode_rows,
        requested_field=args.end_frame_field,
    )

    records: dict[int, EpisodeSpec] = {}
    for row_number, record in enumerate(episode_rows, start=1):
        if "episode_index" not in record or "length" not in record:
            raise ValueError(f"episodes.jsonl row {row_number} lacks episode_index or length")
        episode_index = _integer(record["episode_index"], context=f"episodes.jsonl row {row_number} episode_index")
        length = _integer(record["length"], context=f"episode {episode_index} length")
        if episode_index < 0 or length <= 0 or episode_index in records:
            raise ValueError(f"invalid or duplicate episode metadata for episode {episode_index}")
        task_index = _task_index_from_episode_row(
            record,
            prompt_to_index=prompt_to_index,
            context=f"episode {episode_index}",
        )
        if task_index is None:
            task_index = _parquet_task_index(
                _parquet_path(dataset_root, info, episode_index),
                episode_index=episode_index,
            )
        if task_index not in prompts:
            raise ValueError(f"episode {episode_index} task_index {task_index} is absent from tasks.jsonl")
        end_frame = (
            length - 1
            if end_field is None
            else _integer(record[end_field], context=f"episode {episode_index} {end_field}")
        )
        if end_frame < 0 or end_frame >= length:
            raise ValueError(f"episode {episode_index} inclusive end frame {end_frame} is outside [0, {length - 1}]")
        group_index, group_position = divmod(episode_index, TASKS_PER_GROUP)
        records[episode_index] = EpisodeSpec(
            episode_index=episode_index,
            group_index=group_index,
            group_position=group_position,
            task_index=task_index,
            prompt=prompts[task_index],
            length=length,
            end_frame=end_frame,
        )

    all_ids = sorted(records)
    if all_ids != list(range(len(all_ids))) or len(all_ids) % TASKS_PER_GROUP:
        raise ValueError(
            "episode_id//4 is not a reliable complete-group mapping: episode ids must be contiguous from zero "
            "and the count must be divisible by four"
        )
    groups: dict[int, tuple[EpisodeSpec, ...]] = {}
    for group_index in range(len(all_ids) // TASKS_PER_GROUP):
        group = tuple(records[group_index * TASKS_PER_GROUP + position] for position in range(TASKS_PER_GROUP))
        if tuple(spec.group_position for spec in group) != tuple(range(TASKS_PER_GROUP)):
            raise AssertionError("internal group-position mismatch")
        groups[group_index] = group
    task_by_position = tuple(groups[0][position].task_index for position in range(TASKS_PER_GROUP))
    if len(set(task_by_position)) != TASKS_PER_GROUP:
        raise ValueError(f"first group does not contain four distinct tasks: {task_by_position}")
    for group_index, group in groups.items():
        actual = tuple(spec.task_index for spec in group)
        if actual != task_by_position:
            raise ValueError(
                "episode_id//4 is not a reliable task-ordered mapping: "
                f"group {group_index} tasks {actual} != canonical {task_by_position}"
            )

    excluded_short = Counter(spec.task_index for spec in records.values() if spec.end_frame < 45)
    eligible_groups = tuple(
        group_index for group_index, group in groups.items() if all(spec.end_frame >= 45 for spec in group)
    )
    if not eligible_groups:
        raise ValueError("no complete four-task group has E>=45 for every subtask")
    return DatasetAudit(
        episodes=tuple(records[index] for index in all_ids),
        eligible_group_ids=eligible_groups,
        excluded_short_by_task=dict(excluded_short),
        end_frame_source=end_source,
        task_by_group_position=task_by_position,  # type: ignore[arg-type]
    )


def select_and_split_groups(
    eligible_group_ids: Sequence[int],
    *,
    max_groups: int,
    seed: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if max_groups <= 1:
        raise ValueError("--max-groups must be at least 2")
    rng = random.Random(seed)
    selected = rng.sample(list(eligible_group_ids), min(max_groups, len(eligible_group_ids)))
    rng.shuffle(selected)
    test_count = max(1, round(0.20 * len(selected)))
    if test_count >= len(selected):
        test_count = 1
    test = tuple(sorted(selected[:test_count]))
    train = tuple(sorted(selected[test_count:]))
    if not train or not test or set(train).intersection(test):
        raise AssertionError("invalid group train/test split")
    return train, test


def _evaluation_repack() -> Any:
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


def _validate_clean_config(config: Any, args: argparse.Namespace) -> None:
    if config.name != DEFAULT_CONFIG_NAME or args.config_name != DEFAULT_CONFIG_NAME:
        raise ValueError(f"probe is locked to clean config {DEFAULT_CONFIG_NAME!r}")
    if not bool(getattr(config.model, "pi05", False)):
        raise ValueError("probe requires pi0.5")
    if bool(getattr(getattr(config, "training_time_rtc", None), "enabled", False)):
        raise ValueError("probe forbids TTRTC")
    completion_head = getattr(config.model, "completion_head", None)
    if completion_head is not None and bool(getattr(completion_head, "enabled", False)):
        raise ValueError("probe requires the clean model without a completion head")
    if getattr(getattr(config, "completion", None), "stage", "disabled") != "disabled":
        raise ValueError("probe requires completion training to be disabled")
    if not (args.checkpoint / "params").is_dir():
        raise FileNotFoundError(args.checkpoint / "params")


def extract_prefix_features(
    args: argparse.Namespace,
    specs: Sequence[EpisodeSpec],
) -> tuple[dict[tuple[int, int], np.ndarray], int]:
    os.environ["HF_LEROBOT_HOME"] = str(args.hf_lerobot_home.resolve())

    import flax.nnx as nnx  # noqa: PLC0415
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # noqa: PLC0415

    import openpi.models.model as model_api  # noqa: PLC0415
    from openpi.policies import policy_config  # noqa: PLC0415
    from openpi.training import config as training_config  # noqa: PLC0415

    config = training_config.get_config(args.config_name)
    _validate_clean_config(config, args)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id != EXPECTED_REPO_ID:
        raise ValueError(f"clean config repo_id {data_config.repo_id!r} != {EXPECTED_REPO_ID!r}")
    if not bool(data_config.prompt_from_task):
        raise ValueError("clean data config must declare prompt_from_task=True")

    policy = policy_config.create_trained_policy(
        config,
        args.checkpoint,
        repack_transforms=_evaluation_repack(),
        sample_kwargs={},
    )
    if bool(getattr(policy, "_is_pytorch_model", False)):
        raise ValueError("probe requires the JAX clean checkpoint")
    model = policy._model  # noqa: SLF001
    if not hasattr(model, "compute_prefix_feature"):
        raise ValueError("clean model lacks compute_prefix_feature")
    model.eval()
    graphdef, state = nnx.split(model)

    def compute_prefix(state: Any, observation: model_api.Observation) -> Any:
        module = nnx.merge(graphdef, state)
        return module.compute_prefix_feature(jax.random.key(0), observation, train=False)

    compute_fn = jax.jit(compute_prefix)
    dataset = lerobot_dataset.LeRobotDataset(EXPECTED_REPO_ID, root=args.dataset_root)
    episode_from = dataset.episode_data_index["from"]
    episode_to = dataset.episode_data_index["to"]
    requests = [
        (spec, frame_index)
        for spec in sorted(specs, key=lambda item: item.episode_index)
        for frame_index in spec.requested_frames
    ]
    features: dict[tuple[int, int], np.ndarray] = {}

    for batch_start in range(0, len(requests), args.batch_size):
        batch = requests[batch_start : batch_start + args.batch_size]
        transformed: list[dict[str, Any]] = []
        for spec, frame_index in batch:
            start = _integer(episode_from[spec.episode_index], context="episode_data_index from")
            stop = _integer(episode_to[spec.episode_index], context="episode_data_index to")
            if stop - start != spec.length:
                raise ValueError(
                    f"episode {spec.episode_index} LeRobot length {stop - start} != metadata {spec.length}"
                )
            sample = dict(dataset[start + frame_index])
            if _integer(sample["episode_index"], context="sample episode_index") != spec.episode_index:
                raise ValueError("LeRobot sample episode_index mismatch")
            if _integer(sample["frame_index"], context="sample frame_index") != frame_index:
                raise ValueError("LeRobot sample frame_index mismatch")
            if _integer(sample["task_index"], context="sample task_index") != spec.task_index:
                raise ValueError("LeRobot sample task_index mismatch")
            sample["prompt"] = spec.prompt
            transformed.append(policy._input_transform(sample))  # noqa: SLF001

        valid_count = len(transformed)
        while len(transformed) < args.batch_size:
            transformed.append(jax.tree.map(lambda value: np.array(value, copy=True), transformed[-1]))
        batched = jax.tree.map(
            lambda *values: jnp.asarray(np.stack([np.asarray(value) for value in values], axis=0)),
            *transformed,
        )
        observation = model_api.Observation.from_dict(batched)
        batch_features = np.asarray(jax.block_until_ready(compute_fn(state, observation)))[:valid_count]
        expected = (valid_count, int(model.prefix_feature_dim))
        if batch_features.shape != expected or batch_features.dtype != np.float32:
            raise ValueError(
                f"compute_prefix_feature returned {batch_features.shape}/{batch_features.dtype}, "
                f"expected {expected}/float32"
            )
        if not np.isfinite(batch_features).all():
            raise ValueError("compute_prefix_feature returned non-finite values")
        for (spec, frame_index), feature in zip(batch, batch_features, strict=True):
            key = (spec.episode_index, frame_index)
            if key in features:
                raise AssertionError(f"duplicate feature request {key}")
            features[key] = feature
        print(f"Extracted {batch_start + valid_count}/{len(requests)} prefix features")

    return features, int(model.prefix_feature_dim)


def save_feature_cache(
    path: Path,
    features: Mapping[tuple[int, int], np.ndarray],
    *,
    feature_dim: int,
) -> None:
    """Save the extracted frame features before any classifier is fitted."""

    if path.exists() and not path.is_dir():
        raise ValueError(f"feature cache path exists but is not a directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    keys = sorted(features)
    if not keys:
        raise ValueError("cannot save an empty feature cache")
    matrix = np.asarray([features[key] for key in keys], dtype=np.float32)
    if matrix.shape != (len(keys), feature_dim) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid feature matrix for cache: {matrix.shape}")
    # Keep each member as a plain NPY file.  Some mounted /mnt filesystems do
    # not support the seek pattern used by ZIP-based np.savez_compressed.
    np.save(path / "feature.npy", matrix)
    np.save(path / "episode_index.npy", np.asarray([key[0] for key in keys], dtype=np.int32))
    np.save(path / "frame_index.npy", np.asarray([key[1] for key in keys], dtype=np.int32))
    print(f"Saved prefix feature cache directory to {path} ({len(keys)} frames)")


def load_feature_cache(
    path: Path,
    expected_keys: Sequence[tuple[int, int]],
    *,
    feature_dim: int | None = None,
) -> tuple[dict[tuple[int, int], np.ndarray], int]:
    """Load a simple cache and fail if it does not match this exact request set."""

    if not path.is_dir():
        raise FileNotFoundError(f"feature cache directory not found: {path}")
    members = {name: path / f"{name}.npy" for name in ("episode_index", "frame_index", "feature")}
    missing = [name for name, member in members.items() if not member.is_file()]
    if missing:
        raise ValueError(f"feature cache {path} is incomplete; missing {missing}")
    episode_indices = np.asarray(np.load(members["episode_index"], allow_pickle=False), dtype=np.int64)
    frame_indices = np.asarray(np.load(members["frame_index"], allow_pickle=False), dtype=np.int64)
    matrix = np.asarray(np.load(members["feature"], allow_pickle=False), dtype=np.float32)
    if episode_indices.ndim != 1 or frame_indices.shape != episode_indices.shape:
        raise ValueError(f"feature cache {path} has malformed frame keys")
    if matrix.ndim != 2 or matrix.shape[0] != len(episode_indices):
        raise ValueError(f"feature cache {path} has malformed feature shape {matrix.shape}")
    if feature_dim is not None and matrix.shape[1] != feature_dim:
        raise ValueError(f"feature cache feature_dim {matrix.shape[1]} != requested {feature_dim}")
    keys = list(zip(episode_indices.tolist(), frame_indices.tolist(), strict=True))
    if len(set(keys)) != len(keys):
        raise ValueError(f"feature cache {path} contains duplicate frame keys")
    expected = sorted(set(expected_keys))
    if sorted(keys) != expected:
        raise ValueError(
            f"feature cache {path} keys do not match this run: cached={len(keys)}, expected={len(expected)}"
        )
    features = {key: matrix[index] for index, key in enumerate(keys)}
    print(f"Loaded prefix feature cache directory from {path} ({len(features)} frames)")
    return features, int(matrix.shape[1])


def _episode_arrays(
    specs: Sequence[EpisodeSpec],
    features: Mapping[tuple[int, int], np.ndarray],
) -> dict[str, np.ndarray]:
    current_positive: list[np.ndarray] = []
    current_hard: list[np.ndarray] = []
    temporal_positive: list[np.ndarray] = []
    temporal_hard: list[np.ndarray] = []
    for spec in specs:
        em45, em30, em15, end = spec.requested_frames
        z_em45 = features[(spec.episode_index, em45)]
        z_em30 = features[(spec.episode_index, em30)]
        z_em15 = features[(spec.episode_index, em15)]
        z_end = features[(spec.episode_index, end)]
        current_positive.append(z_end)
        current_hard.append(z_em15)
        temporal_positive.append(np.concatenate([z_em30, z_em15, z_end]))
        temporal_hard.append(np.concatenate([z_em45, z_em30, z_em15]))
    return {
        "current_positive": np.asarray(current_positive, dtype=np.float32),
        "current_hard": np.asarray(current_hard, dtype=np.float32),
        "temporal_positive": np.asarray(temporal_positive, dtype=np.float32),
        "temporal_hard": np.asarray(temporal_hard, dtype=np.float32),
    }


def _ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    ordered_labels = labels[order]
    positive_count = int(np.sum(ordered_labels))
    if positive_count == 0 or positive_count == len(ordered_labels):
        raise ValueError("probe metric inputs must contain both classes")
    precision = np.cumsum(ordered_labels) / np.arange(1, len(ordered_labels) + 1)
    auprc = float(np.sum(precision[ordered_labels == 1]) / positive_count)
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    comparisons = positive_scores[:, None] - negative_scores[None, :]
    roc_auc = float(np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0))
    return {
        "auprc": auprc,
        "roc_auc": roc_auc,
        "accuracy_at_0_5": float(np.mean((scores >= 0.5) == labels)),
    }


def _fit_logistic(values: np.ndarray, labels: np.ndarray, *, seed: int) -> _LogisticProbe:
    """Fits a small L2 logistic probe without requiring scikit-learn.

    The clean environment used for prefix extraction does not necessarily have
    sklearn installed.  JAX is already required by the frozen pi0.5 forward,
    so a fixed-seed, full-batch Adam fit keeps this diagnostic self-contained
    while preserving the requested StandardScaler + C=1 logistic objective.
    """

    del seed  # zero initialization and full-batch updates are deterministic
    values = np.asarray(values, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    standardized = ((values - mean) / scale).astype(np.float32)

    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    x = jnp.asarray(standardized)
    y = jnp.asarray(labels)
    sample_count = float(len(labels))
    regularization = 1.0 / sample_count  # C=1 in sklearn's sum-loss convention
    weight = jnp.zeros((standardized.shape[1],), dtype=jnp.float32)
    bias = jnp.asarray(0.0, dtype=jnp.float32)
    first_m = jnp.zeros_like(weight)
    first_mb = jnp.asarray(0.0, dtype=jnp.float32)
    second_m = jnp.zeros_like(weight)
    second_mb = jnp.asarray(0.0, dtype=jnp.float32)

    def loss_fn(current_weight: jax.Array, current_bias: jax.Array) -> jax.Array:
        logits = x @ current_weight + current_bias
        data_loss = jnp.mean(jnp.maximum(logits, 0.0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits))))
        return data_loss + 0.5 * regularization * jnp.sum(current_weight * current_weight)

    @jax.jit
    def update(
        current_weight: jax.Array,
        current_bias: jax.Array,
        current_first_m: jax.Array,
        current_first_mb: jax.Array,
        current_second_m: jax.Array,
        current_second_mb: jax.Array,
        step: jax.Array,
    ) -> tuple[jax.Array, ...]:
        _, (gradient, gradient_b) = jax.value_and_grad(loss_fn, argnums=(0, 1))(current_weight, current_bias)
        beta1 = 0.9
        beta2 = 0.999
        first_m = beta1 * current_first_m + (1.0 - beta1) * gradient
        first_mb = beta1 * current_first_mb + (1.0 - beta1) * gradient_b
        second_m = beta2 * current_second_m + (1.0 - beta2) * gradient * gradient
        second_mb = beta2 * current_second_mb + (1.0 - beta2) * gradient_b * gradient_b
        step_float = step.astype(jnp.float32)
        first_hat = first_m / (1.0 - beta1**step_float)
        first_hat_b = first_mb / (1.0 - beta1**step_float)
        second_hat = second_m / (1.0 - beta2**step_float)
        second_hat_b = second_mb / (1.0 - beta2**step_float)
        learning_rate = 0.03
        current_weight = current_weight - learning_rate * first_hat / (jnp.sqrt(second_hat) + 1e-8)
        current_bias = current_bias - learning_rate * first_hat_b / (jnp.sqrt(second_hat_b) + 1e-8)
        return current_weight, current_bias, first_m, first_mb, second_m, second_mb

    for step in range(1, 1001):
        weight, bias, first_m, first_mb, second_m, second_mb = update(
            weight,
            bias,
            first_m,
            first_mb,
            second_m,
            second_mb,
            jnp.asarray(step, dtype=jnp.float32),
        )
    weight = np.asarray(jax.block_until_ready(weight), dtype=np.float32)
    bias = float(np.asarray(jax.block_until_ready(bias)))
    if not np.isfinite(weight).all() or not np.isfinite(bias):
        raise ValueError("logistic probe produced non-finite parameters")
    return _LogisticProbe(mean=mean, scale=scale, weight=weight, bias=bias)


def _evaluate_probe(
    train_specs: Sequence[EpisodeSpec],
    test_specs: Sequence[EpisodeSpec],
    features: Mapping[tuple[int, int], np.ndarray],
    *,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    train = _episode_arrays(train_specs, features)
    test = _episode_arrays(test_specs, features)
    train_positive = train[f"{mode}_positive"]
    train_hard = train[f"{mode}_hard"]
    test_positive = test[f"{mode}_positive"]
    test_hard = test[f"{mode}_hard"]
    x_train = np.concatenate([train_hard, train_positive], axis=0)
    y_train = np.concatenate([np.zeros(len(train_hard), dtype=np.int8), np.ones(len(train_positive), dtype=np.int8)])
    classifier = _fit_logistic(x_train, y_train, seed=seed)

    def evaluate_subset(indices: np.ndarray) -> dict[str, Any]:
        hard_scores = classifier.predict_proba(test_hard[indices])[:, 1]
        positive_scores = classifier.predict_proba(test_positive[indices])[:, 1]
        labels = np.concatenate([np.zeros(len(indices), dtype=np.int8), np.ones(len(indices), dtype=np.int8)])
        scores = np.concatenate([hard_scores, positive_scores])
        margin = positive_scores - hard_scores
        metrics: dict[str, Any] = _ranking_metrics(labels, scores)
        metrics.update(
            {
                "episode_pairs": len(indices),
                "paired_ordering_accuracy": float(np.mean(margin > 0.0)),
                "paired_margin": {
                    "mean": float(np.mean(margin)),
                    "median": float(np.median(margin)),
                    "p25": float(np.percentile(margin, 25)),
                    "p75": float(np.percentile(margin, 75)),
                },
            }
        )
        return metrics

    overall = evaluate_subset(np.arange(len(test_specs), dtype=np.int64))
    per_task: dict[str, Any] = {}
    task_values = np.asarray([spec.task_index for spec in test_specs], dtype=np.int64)
    for task_index in sorted(set(task_values.tolist())):
        indices = np.flatnonzero(task_values == task_index)
        per_task[str(task_index)] = {
            "prompt": test_specs[int(indices[0])].prompt,
            **evaluate_subset(indices),
        }
    return {"overall": overall, "per_task": per_task}


def _split_counts(specs: Sequence[EpisodeSpec]) -> dict[str, Any]:
    counts = Counter(spec.task_index for spec in specs)
    return {
        "trajectory_groups": len({spec.group_index for spec in specs}),
        "episodes": len(specs),
        "episodes_by_task": {str(task): int(count) for task, count in sorted(counts.items())},
    }


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.seed != 42:
        raise ValueError("minimal probe is locked to seed=42")
    args.dataset_root = args.dataset_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.hf_lerobot_home = args.hf_lerobot_home.resolve()
    audit = audit_dataset(args)
    max_groups = 2 if args.dry_run else args.max_groups
    train_groups, test_groups = select_and_split_groups(
        audit.eligible_group_ids,
        max_groups=max_groups,
        seed=args.seed,
    )
    selected_groups = set(train_groups).union(test_groups)
    specs = tuple(spec for spec in audit.episodes if spec.group_index in selected_groups)
    info = _read_json(args.dataset_root / "meta" / "info.json")
    for spec in specs:
        _validate_selected_parquet(args.dataset_root, info, spec)
    requested_keys = [(spec.episode_index, frame_index) for spec in specs for frame_index in spec.requested_frames]
    feature_cache = args.features_cache.resolve()
    if feature_cache.is_dir() and not args.refresh_features and not args.dry_run:
        features, feature_dim = load_feature_cache(feature_cache, requested_keys)
    else:
        features, feature_dim = extract_prefix_features(args, specs)
        if not args.dry_run:
            save_feature_cache(feature_cache, features, feature_dim=feature_dim)

    if args.dry_run:
        sample = specs[0]
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "end_frame_source": audit.end_frame_source,
                    "groups": sorted(selected_groups),
                    "episodes": len(specs),
                    "feature_requests": len(features),
                    "feature_dim": feature_dim,
                    "example": {
                        "episode_index": sample.episode_index,
                        "task_index": sample.task_index,
                        "prompt": sample.prompt,
                        "E": sample.end_frame,
                        "frames": sample.requested_frames,
                        "hard_label": 0,
                        "positive_label": 1,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return None

    train_group_set = set(train_groups)
    test_group_set = set(test_groups)
    train_specs = tuple(spec for spec in specs if spec.group_index in train_group_set)
    test_specs = tuple(spec for spec in specs if spec.group_index in test_group_set)
    metrics: dict[str, Any] = {
        "protocol": "hard_positive_prefix_probe_v1",
        "classifier": "L2 logistic probe, StandardScaler, C=1, JAX Adam optimizer",
        "dataset_root": str(args.dataset_root),
        "config_name": args.config_name,
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "end_frame_source": audit.end_frame_source,
        "frame_offsets": list(FRAME_OFFSETS),
        "feature_dim": feature_dim,
        "selected_groups": len(selected_groups),
        "excluded_E_lt_45_by_task": {
            str(task): int(count) for task, count in sorted(audit.excluded_short_by_task.items())
        },
        "split": {
            "train": _split_counts(train_specs),
            "test": _split_counts(test_specs),
        },
        "current_only": _evaluate_probe(
            train_specs,
            test_specs,
            features,
            mode="current",
            seed=args.seed,
        ),
        "temporal": _evaluate_probe(
            train_specs,
            test_specs,
            features,
            mode="temporal",
            seed=args.seed,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Saved metrics to {args.output}")
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--hf-lerobot-home", type=Path, default=DEFAULT_HF_LEROBOT_HOME)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--features-cache",
        type=Path,
        default=None,
        help="Directory cache for extracted prefix features (defaults beside --output).",
    )
    parser.add_argument(
        "--refresh-features",
        action="store_true",
        help="Ignore an existing --features-cache and extract prefix features again.",
    )
    parser.add_argument("--max-groups", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--end-frame-field",
        help="Explicit inclusive manual completion-frame field in meta/episodes.jsonl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract exactly two complete groups, print shape/prompt/frame audits, and skip logistic fitting.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.features_cache is None:
        args.features_cache = args.output.with_name("prefix_features")
    run(args)


if __name__ == "__main__":
    main()
