from collections.abc import Iterator, Sequence
import dataclasses
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.completion as _completion
import openpi.training.completion_data as _completion_data
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)
COMPLETION_TARGET_KEY = "completion_target"


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        preserve_keys: dict[str, str] | None = None,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._preserve_keys = preserve_keys or {}

    def __getitem__(self, index: SupportsIndex) -> T_co:
        sample = self._dataset[index]
        preserved = {output_key: sample[input_key] for output_key, input_key in self._preserve_keys.items()}
        transformed = self._transform(sample)
        transformed.update(preserved)
        return transformed

    def __len__(self) -> int:
        return len(self._dataset)


class EpisodeSubsetDataset(Dataset[T_co]):
    """Exposes selected episode frames while preserving LeRobot's global indices."""

    def __init__(self, dataset: Dataset[T_co], episode_ids: Sequence[int]):
        self._dataset = dataset
        self._episode_ids = tuple(int(episode_id) for episode_id in episode_ids)
        if not self._episode_ids:
            raise ValueError("episode subset must not be empty")
        episode_data_index = getattr(dataset, "episode_data_index", None)
        if episode_data_index is None:
            raise ValueError("episode subset requires a LeRobotDataset with episode_data_index")
        starts = episode_data_index["from"]
        ends = episode_data_index["to"]
        max_episode_id = len(starts) - 1
        invalid_ids = [episode_id for episode_id in self._episode_ids if episode_id < 0 or episode_id > max_episode_id]
        if invalid_ids:
            raise ValueError(f"episode subset contains out-of-range episode IDs: {invalid_ids}")
        self._indices = np.concatenate(
            [
                np.arange(int(starts[episode_id]), int(ends[episode_id]), dtype=np.int64)
                for episode_id in self._episode_ids
            ]
        )

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._dataset[int(self._indices[int(index)])]

    def __len__(self) -> int:
        return int(self._indices.shape[0])


class BalancedCompletionSampler(torch.utils.data.Sampler[int]):
    """Yields fixed-composition batches over an episode-subset dataset.

    Dataset-local indices are reconstructed from the audited episode lengths;
    ``EpisodeSubsetDataset`` concatenates episodes in the same order. Sampling
    is with replacement across batches so the two positive frames per episode
    cannot be exhausted after only a handful of optimizer steps.
    """

    def __init__(
        self,
        episode_ids: Sequence[int],
        episode_audits: typing.Mapping[int, _completion_data.EpisodeAudit],
        *,
        batch_size: int,
        positive_fraction: float,
        hard_negative_fraction: float,
        hard_negative_window: int,
        seed: int,
    ):
        if batch_size < 3:
            raise ValueError("balanced completion sampling requires batch_size >= 3")
        positive_per_batch = max(1, round(batch_size * positive_fraction))
        hard_negative_per_batch = max(1, round(batch_size * hard_negative_fraction))
        ordinary_negative_per_batch = batch_size - positive_per_batch - hard_negative_per_batch
        if ordinary_negative_per_batch <= 0:
            raise ValueError("balanced completion fractions leave no ordinary negatives in a batch")

        positives: list[int] = []
        hard_negatives: list[int] = []
        ordinary_negatives: list[int] = []
        offset = 0
        for episode_id in episode_ids:
            audit = episode_audits[int(episode_id)]
            positive_start = audit.frame_count - audit.positive_count
            positives.extend(range(offset + positive_start, offset + audit.frame_count))
            hard_start = max(0, positive_start - hard_negative_window)
            hard_negatives.extend(range(offset + hard_start, offset + positive_start))
            ordinary_negatives.extend(range(offset, offset + hard_start))
            offset += audit.frame_count

        if not positives:
            raise ValueError("balanced completion sampling found no positive frames")
        if not hard_negatives:
            raise ValueError("balanced completion sampling found no hard-negative frames")
        if not ordinary_negatives:
            # Very short diagnostic episodes may contain only terminal-near
            # negatives. Keep the sampler usable while making the fallback
            # explicit through the public pool sizes below.
            ordinary_negatives = list(hard_negatives)

        self._positive_indices = np.asarray(positives, dtype=np.int64)
        self._hard_negative_indices = np.asarray(hard_negatives, dtype=np.int64)
        self._ordinary_negative_indices = np.asarray(ordinary_negatives, dtype=np.int64)
        self._positive_per_batch = positive_per_batch
        self._hard_negative_per_batch = hard_negative_per_batch
        self._ordinary_negative_per_batch = ordinary_negative_per_batch
        self._batch_size = batch_size
        self._batch_count = max(1, int(np.ceil(offset / batch_size)))
        self._seed = seed
        self._epoch = 0

    @property
    def batch_composition(self) -> dict[str, int]:
        return {
            "positive": self._positive_per_batch,
            "hard_negative": self._hard_negative_per_batch,
            "ordinary_negative": self._ordinary_negative_per_batch,
        }

    @property
    def pool_sizes(self) -> dict[str, int]:
        return {
            "positive": int(self._positive_indices.size),
            "hard_negative": int(self._hard_negative_indices.size),
            "ordinary_negative": int(self._ordinary_negative_indices.size),
        }

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self._seed + self._epoch)
        self._epoch += 1
        for _ in range(self._batch_count):
            batch = np.concatenate(
                [
                    rng.choice(self._positive_indices, self._positive_per_batch, replace=True),
                    rng.choice(self._hard_negative_indices, self._hard_negative_per_batch, replace=True),
                    rng.choice(self._ordinary_negative_indices, self._ordinary_negative_per_batch, replace=True),
                ]
            )
            rng.shuffle(batch)
            yield from (int(index) for index in batch)

    def __len__(self) -> int:
        return self._batch_count * self._batch_size


class ProgressStratifiedSampler(torch.utils.data.Sampler[int]):
    """Samples uniformly over ten within-subtask progress bins and episodes.

    Each draw first chooses an episode uniformly from the episodes represented
    in a bin, then chooses one of that episode's frames in the bin. That
    two-stage draw keeps long episodes from dominating merely because they
    contribute more frames. Draws use replacement so sparse bins stay present
    in every batch throughout a run.
    """

    def __init__(
        self,
        episode_ids: Sequence[int],
        episode_audits: typing.Mapping[int, _completion_data.EpisodeAudit],
        *,
        batch_size: int,
        seed: int,
        bin_count: int = _completion.PROGRESS_BIN_COUNT,
    ):
        if bin_count != _completion.PROGRESS_BIN_COUNT:
            raise ValueError(f"progress sampling requires exactly {_completion.PROGRESS_BIN_COUNT} bins")
        if batch_size < bin_count:
            raise ValueError(f"progress-stratified sampling requires batch_size >= {bin_count}")
        if not episode_ids:
            raise ValueError("progress-stratified sampling requires at least one episode")

        indices_by_bin_episode: list[dict[int, np.ndarray]] = [{} for _ in range(bin_count)]
        offset = 0
        for raw_episode_id in episode_ids:
            episode_id = int(raw_episode_id)
            audit = episode_audits[episode_id]
            targets = _completion_data.make_progress_targets(audit.frame_count)
            bin_indices = np.minimum((targets * bin_count).astype(np.int64), bin_count - 1)
            for bin_index in range(bin_count):
                local_indices = np.flatnonzero(bin_indices == bin_index)
                if local_indices.size:
                    indices_by_bin_episode[bin_index][episode_id] = (offset + local_indices).astype(np.int64)
            offset += audit.frame_count

        empty_bins = [bin_index for bin_index, pools in enumerate(indices_by_bin_episode) if not pools]
        if empty_bins:
            raise ValueError(
                "progress-stratified sampling found no frames in progress bin(s) "
                f"{empty_bins}; use episodes that cover all {_completion.PROGRESS_BIN_COUNT} bins"
            )

        self._indices_by_bin_episode = indices_by_bin_episode
        self._episode_ids_by_bin = [
            np.asarray(sorted(episode_pools), dtype=np.int64) for episode_pools in indices_by_bin_episode
        ]
        self._pool_sizes = {
            f"bin_{bin_index}": int(sum(len(indices) for indices in episode_pools.values()))
            for bin_index, episode_pools in enumerate(indices_by_bin_episode)
        }
        self._episode_pool_sizes = {
            f"bin_{bin_index}": len(episode_pools) for bin_index, episode_pools in enumerate(indices_by_bin_episode)
        }
        self._batch_size = batch_size
        self._bin_count = bin_count
        self._batch_count = max(1, int(np.ceil(offset / batch_size)))
        self._seed = seed
        self._epoch = 0

    def _batch_composition(self, batch_index: int, *, epoch: int) -> dict[str, int]:
        base_count, remainder = divmod(self._batch_size, self._bin_count)
        composition = {f"bin_{bin_index}": base_count for bin_index in range(self._bin_count)}
        # Rotate the bins that receive the remainder to avoid a systematic
        # advantage for early progress ranges when batch_size is not divisible.
        start_bin = (epoch * self._batch_count + batch_index) % self._bin_count
        for offset in range(remainder):
            composition[f"bin_{(start_bin + offset) % self._bin_count}"] += 1
        return composition

    @property
    def batch_composition(self) -> dict[str, int]:
        """Exact composition of the first batch; remainders rotate thereafter."""

        return self._batch_composition(0, epoch=self._epoch)

    @property
    def pool_sizes(self) -> dict[str, int]:
        return dict(self._pool_sizes)

    @property
    def episode_pool_sizes(self) -> dict[str, int]:
        return dict(self._episode_pool_sizes)

    def __iter__(self) -> Iterator[int]:
        epoch = self._epoch
        rng = np.random.default_rng(self._seed + epoch)
        self._epoch += 1
        for batch_index in range(self._batch_count):
            composition = self._batch_composition(batch_index, epoch=epoch)
            batch_parts: list[np.ndarray] = []
            for bin_index in range(self._bin_count):
                count = composition[f"bin_{bin_index}"]
                if count == 0:
                    continue
                episode_ids = self._episode_ids_by_bin[bin_index]
                sampled_episode_ids = rng.choice(episode_ids, count, replace=True)
                sampled_indices = np.asarray(
                    [
                        rng.choice(self._indices_by_bin_episode[bin_index][int(episode_id)])
                        for episode_id in sampled_episode_ids
                    ],
                    dtype=np.int64,
                )
                batch_parts.append(sampled_indices)
            batch = np.concatenate(batch_parts)
            rng.shuffle(batch)
            yield from (int(index) for index in batch)

    def __len__(self) -> int:
        return self._batch_count * self._batch_size


class BoundaryCompletionSampler(torch.utils.data.Sampler[int]):
    """Deterministic per-epoch sampler for the boundary completion scheme.

    Builds a fixed training sample set per episode (all positives + every
    ``negative_stride`` ordinary negative on the source-frame grid + forced
    first-frame negatives of subtasks 2/3/4), concatenates them across the
    selected train episodes into one global sample set ``S`` of dataset-local
    indices, and each epoch yields ``S`` fully shuffled and split into full
    batches. The final short batch is deterministically padded by repeating the
    first samples of the epoch so every original sample is visited at least once
    per epoch and every batch is full-sized (required for device sharding).

    Sampling is fully deterministic given ``(seed, epoch)``: resume reconstructs
    the same epoch sequence and fast-forwards past already-consumed batches.
    """

    def __init__(
        self,
        episode_ids: Sequence[int],
        boundary_audits: typing.Mapping[int, _completion_data.BoundaryEpisodeAudit],
        *,
        batch_size: int,
        seed: int,
        stride: int = 15,
        forced_first_n: int = 5,
    ):
        if batch_size <= 0:
            raise ValueError("boundary completion sampling requires batch_size > 0")
        if not episode_ids:
            raise ValueError("boundary completion sampling requires at least one episode")

        sample_sets: list[np.ndarray] = []
        offset = 0
        for raw_episode_id in episode_ids:
            episode_id = int(raw_episode_id)
            audit = boundary_audits[episode_id]
            local = _completion_data.build_boundary_train_sample_set(
                audit, stride=stride, forced_first_n=forced_first_n
            )
            sample_sets.append(local.astype(np.int64) + offset)
            offset += audit.frame_count

        sample_set = np.concatenate(sample_sets) if sample_sets else np.empty(0, dtype=np.int64)
        if sample_set.size == 0:
            raise ValueError("boundary completion sampling produced an empty sample set")
        if np.unique(sample_set).size != sample_set.size:
            raise ValueError("boundary completion sample set contains duplicate indices")

        self._sample_set = sample_set
        self._batch_size = batch_size
        self._steps_per_epoch = max(1, int(np.ceil(sample_set.size / batch_size)))
        self._seed = seed
        self._epoch = 0
        self._skip_batches = 0

    @property
    def num_samples(self) -> int:
        return int(self._sample_set.size)

    @property
    def steps_per_epoch(self) -> int:
        return self._steps_per_epoch

    @property
    def sample_set(self) -> np.ndarray:
        return self._sample_set

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def set_skip_batches(self, skip_batches: int) -> None:
        """Fast-forward past ``skip_batches`` batches on the next ``__iter__`` (resume)."""

        if skip_batches < 0:
            raise ValueError("skip_batches must be non-negative")
        self._skip_batches = int(skip_batches)

    def _epoch_indices(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng(self._seed + epoch)
        shuffled = rng.permutation(self._sample_set)
        total = self._steps_per_epoch * self._batch_size
        if shuffled.size < total:
            pad = total - shuffled.size
            shuffled = np.concatenate([shuffled, shuffled[:pad]])
        return shuffled

    def __iter__(self) -> Iterator[int]:
        shuffled = self._epoch_indices(self._epoch)
        self._epoch += 1
        start = self._skip_batches * self._batch_size
        self._skip_batches = 0
        if start >= shuffled.size:
            return
        for i in range(start, shuffled.size, self._batch_size):
            yield from (int(index) for index in shuffled[i : i + self._batch_size])

    def __len__(self) -> int:
        return (self._steps_per_epoch - self._skip_batches) * self._batch_size


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    if data_config.lerobot_home is not None:
        os.environ["HF_LEROBOT_HOME"] = data_config.lerobot_home
        dataset_root = os.path.join(data_config.lerobot_home, repo_id)
    else:
        dataset_root = None

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=dataset_root)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=dataset_root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )
    if data_config.episodes is not None:
        dataset = EpisodeSubsetDataset(dataset, data_config.episodes)

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    preserve_keys = None
    if data_config.completion_label_key is not None:
        preserve_keys = {COMPLETION_TARGET_KEY: data_config.completion_label_key}
    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        preserve_keys=preserve_keys,
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def prepare_completion_data(config: _config.TrainConfig) -> _completion_data.CompletionDataInfo:
    """Creates/reuses the split manifest and audits labels only for S2 head training."""

    if not config.completion.uses_completion_data:
        raise ValueError("completion data preparation requires completion stage 'action' or 'head'")
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id in (None, "fake"):
        raise ValueError("completion training requires an explicit non-fake LeRobot repo_id")
    if data_config.rlds_data_dir is not None:
        raise ValueError("completion training currently supports only LeRobot datasets, not RLDS")
    if data_config.lerobot_home is not None:
        os.environ["HF_LEROBOT_HOME"] = data_config.lerobot_home
        dataset_root: str | None = os.path.join(data_config.lerobot_home, data_config.repo_id)
    else:
        dataset_root = None
    metadata = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=dataset_root)
    manifest_path = config.completion.split_manifest_path
    if manifest_path is None:
        raise ValueError("completion.split_manifest_path must be set")
    info = _completion_data.prepare_completion_data(
        metadata,
        repo_id=data_config.repo_id,
        dataset_root=metadata.root,
        label_key=config.completion.label_key,
        manifest_path=manifest_path,
        manifest_repo_id=config.completion.split_manifest_repo_id,
        objective=config.completion.objective,
        audit_labels=config.completion.requires_completion_labels,
        seed=config.completion.split_seed,
        episodes_per_group=config.completion.episodes_per_group,
        val_groups=config.completion.val_groups,
        test_groups=config.completion.test_groups,
    )
    logging.info(
        "Completion split: train_episodes=%d val_episodes=%d test_episodes=%d audit_labels=%s",
        len(info.manifest.episode_ids("train")),
        len(info.manifest.episode_ids("val")),
        len(info.manifest.episode_ids("test")),
        config.completion.requires_completion_labels,
    )
    if config.completion.uses_progress_objective:
        logging.info(
            "Progress train labels audited: train_episodes=%d bins=%d",
            len(info.manifest.episode_ids("train")),
            _completion.PROGRESS_BIN_COUNT,
        )
    elif info.pos_weight is not None:
        logging.info(
            "Completion train labels: positive=%d negative=%d pos_weight=%.6f",
            info.train_positive_count,
            info.train_negative_count,
            info.pos_weight,
        )
    return info


def create_data_loader(
    config: _config.TrainConfig,
    *,
    split: _completion_data.SplitName = "train",
    completion_data_info: _completion_data.CompletionDataInfo | None = None,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    natural_train_eval: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions] | tuple[_model.Observation, _model.Actions, jax.Array]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    if natural_train_eval and split != "train":
        raise ValueError("natural_train_eval is only valid for the train split")
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    uses_completion_split = config.completion.uses_completion_data
    emits_completion_target = config.completion.trains_completion_head
    selected_episode_ids: tuple[int, ...] | None = None
    if uses_completion_split:
        if completion_data_info is None:
            completion_data_info = prepare_completion_data(config)
        selected_episode_ids = completion_data_info.manifest.episode_ids(split)
        if split == "train" and config.completion.train_episode_limit is not None:
            selected_episode_ids = selected_episode_ids[: config.completion.train_episode_limit]
            logging.info(
                "Completion diagnostic episode limit: using %d train episodes: %s",
                len(selected_episode_ids),
                selected_episode_ids,
            )
        data_config = dataclasses.replace(
            data_config,
            episodes=selected_episode_ids,
            completion_label_key=config.completion.label_key if emits_completion_target else None,
        )
    elif split != "train":
        raise ValueError(f"split={split!r} is only available for staged completion training")

    if data_config.rlds_data_dir is not None:
        if uses_completion_split:
            raise ValueError("completion training currently supports only LeRobot datasets, not RLDS")
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        repeat=(not uses_completion_split or split == "train") and not natural_train_eval,
        drop_last=(not uses_completion_split or split == "train") and not natural_train_eval,
        completion_sampling_config=(
            config.completion
            if split == "train" and config.completion.balanced_sampling and not natural_train_eval
            else None
        ),
        progress_sampling_config=(
            config.completion
            if split == "train" and config.completion.uses_progress_stratified_sampling and not natural_train_eval
            else None
        ),
        completion_episode_audits=(
            completion_data_info.episode_audits
            if split == "train"
            and (config.completion.balanced_sampling or config.completion.uses_progress_stratified_sampling)
            and not natural_train_eval
            and completion_data_info is not None
            else None
        ),
        boundary_sampling_config=(
            config.completion
            if split == "train" and config.completion.uses_boundary_sampling and not natural_train_eval
            else None
        ),
        boundary_episode_audits=(
            completion_data_info.boundary_episode_audits
            if split == "train"
            and config.completion.uses_boundary_sampling
            and not natural_train_eval
            and completion_data_info is not None
            else None
        ),
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    repeat: bool = True,
    drop_last: bool = True,
    completion_sampling_config: _completion.CompletionTrainingConfig | None = None,
    progress_sampling_config: _completion.CompletionTrainingConfig | None = None,
    completion_episode_audits: typing.Mapping[int, _completion_data.EpisodeAudit] | None = None,
    boundary_sampling_config: _completion.CompletionTrainingConfig | None = None,
    boundary_episode_audits: typing.Mapping[int, _completion_data.BoundaryEpisodeAudit] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    if completion_sampling_config is not None:
        if sampler is not None:
            raise ValueError("balanced completion sampling cannot be combined with a distributed sampler")
        if completion_episode_audits is None or data_config.episodes is None:
            raise ValueError("balanced completion sampling requires audited episode labels")
        sampler = BalancedCompletionSampler(
            data_config.episodes,
            completion_episode_audits,
            batch_size=local_batch_size,
            positive_fraction=completion_sampling_config.balanced_positive_fraction,
            hard_negative_fraction=completion_sampling_config.balanced_hard_negative_fraction,
            hard_negative_window=completion_sampling_config.hard_negative_window,
            seed=seed,
        )
        logging.info(
            "Balanced completion sampler: batch=%s pools=%s",
            sampler.batch_composition,
            sampler.pool_sizes,
        )

    if progress_sampling_config is not None:
        if sampler is not None:
            raise ValueError("progress-stratified sampling cannot be combined with another sampler")
        if completion_episode_audits is None or data_config.episodes is None:
            raise ValueError("progress-stratified sampling requires audited progress labels")
        sampler = ProgressStratifiedSampler(
            data_config.episodes,
            completion_episode_audits,
            batch_size=local_batch_size,
            seed=seed,
        )
        logging.info(
            "Progress-stratified sampler: batch=%s pools=%s episode_pools=%s "
            "(replacement=True; remainder bins rotate each batch)",
            sampler.batch_composition,
            sampler.pool_sizes,
            sampler.episode_pool_sizes,
        )

    boundary_sampler = None
    if boundary_sampling_config is not None:
        if sampler is not None:
            raise ValueError("boundary completion sampling cannot be combined with another sampler")
        if boundary_episode_audits is None or data_config.episodes is None:
            raise ValueError("boundary completion sampling requires boundary episode audits")
        sampler = BoundaryCompletionSampler(
            data_config.episodes,
            boundary_episode_audits,
            batch_size=local_batch_size,
            seed=seed,
            stride=boundary_sampling_config.negative_stride,
            forced_first_n=boundary_sampling_config.boundary_copy_frames,
        )
        boundary_sampler = sampler
        logging.info(
            "Boundary completion sampler: num_samples=%d steps_per_epoch=%d batch=%d stride=%d "
            "(deterministic shuffle per epoch; last batch padded)",
            sampler.num_samples,
            sampler.steps_per_epoch,
            local_batch_size,
            boundary_sampling_config.negative_stride,
        )

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
        repeat=repeat,
        drop_last=drop_last,
        boundary_sampler=boundary_sampler,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
        repeat: bool = True,
        drop_last: bool = True,
        boundary_sampler: BoundaryCompletionSampler | None = None,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches
        self._repeat = repeat
        self._boundary_sampler = boundary_sampler
        # Validation must keep every frame, including a final short batch. A
        # data-sharded JAX array still requires that batch to divide evenly
        # across devices, so pad it here and trim the repeated rows in the
        # completion evaluator.
        self._batch_size_multiple = (
            len(self._sharding.device_set) if not drop_last and self._sharding is not None else 1
        )

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=drop_last,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    @property
    def boundary_sampler(self) -> BoundaryCompletionSampler | None:
        return self._boundary_sampler

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    batch = _pad_batch_to_multiple(batch, self._batch_size_multiple)
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)
            if not self._repeat:
                return


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _pad_batch_to_multiple(batch, multiple: int):
    """Pads a final evaluation batch by repeating its last row."""

    if multiple <= 1:
        return batch
    leaves = jax.tree.leaves(batch)
    if not leaves:
        raise ValueError("cannot pad an empty batch tree")
    batch_size = leaves[0].shape[0]
    if any(leaf.shape[0] != batch_size for leaf in leaves):
        raise ValueError("all batch leaves must have the same leading dimension")
    padding = (-batch_size) % multiple
    if padding == 0:
        return batch
    return jax.tree.map(
        lambda x: np.concatenate([x, np.repeat(x[-1:], padding, axis=0)], axis=0),
        batch,
    )


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    @property
    def boundary_sampler(self) -> BoundaryCompletionSampler | None:
        """The boundary completion sampler, if this loader was built with one."""

        if isinstance(self._data_loader, TorchDataLoader):
            return self._data_loader.boundary_sampler
        return None

    def __iter__(self):
        for batch in self._data_loader:
            training_batch = (_model.Observation.from_dict(batch), batch["actions"])
            if self._data_config.completion_label_key is None:
                yield training_batch
            else:
                yield (*training_batch, batch[COMPLETION_TARGET_KEY])
