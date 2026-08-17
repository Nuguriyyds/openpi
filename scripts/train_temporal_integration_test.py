"""CPU integration test for the cached temporal-completion training path."""

from __future__ import annotations

import functools
import importlib.util
import os
import sys
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# ``scripts.train`` imports the raw LeRobot loader even though this focused test
# only exercises the already-extracted feature-cache path. Keep the test CPU-only
# and self-contained on developer machines that do not install that optional
# dataset package; normal training environments use the real module.
if importlib.util.find_spec("lerobot") is None:
    lerobot = types.ModuleType("lerobot")
    lerobot.__path__ = []
    lerobot_common = types.ModuleType("lerobot.common")
    lerobot_common.__path__ = []
    lerobot_datasets = types.ModuleType("lerobot.common.datasets")
    lerobot_datasets.__path__ = []
    lerobot_dataset = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
    lerobot.common = lerobot_common
    lerobot_common.datasets = lerobot_datasets
    lerobot_datasets.lerobot_dataset = lerobot_dataset
    sys.modules.update(
        {
            "lerobot": lerobot,
            "lerobot.common": lerobot_common,
            "lerobot.common.datasets": lerobot_datasets,
            "lerobot.common.datasets.lerobot_dataset": lerobot_dataset,
        }
    )

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import completion as completion_model
from openpi.models import model as model_lib
from openpi.training import completion as completion_training
from openpi.training import config as config_lib
from openpi.training import temporal_completion_data
from openpi.training import temporal_completion_features
from openpi.training import utils as training_utils
from scripts import train


class _TinyTemporalModel(model_lib.BaseModel):
    """Minimal BaseModel with one frozen action parameter and a real temporal head."""

    def __init__(self) -> None:
        super().__init__(action_dim=1, action_horizon=1, max_token_len=1)
        # This audited Pi0 path must remain unchanged by head-only training.
        self.action_in_proj = nnx.Param(jnp.asarray([2.0], dtype=jnp.float32))
        self.completion_head = completion_model.TemporalCompletionHead(
            input_dim=2,
            config=completion_model.CompletionHeadConfig(
                enabled=True,
                variant="temporal_mlp",
                temporal_steps=3,
                hidden_dim=4,
                dropout_rate=0.0,
            ),
            rngs=nnx.Rngs(7),
        )
        self.deterministic = True

    def compute_temporal_completion_logits(
        self,
        rng: jax.Array,
        prefix_history: jax.Array,
        *,
        train: bool = False,
    ) -> jax.Array:
        del rng
        return self.completion_head(prefix_history, train=train)

    def compute_loss(self, rng, observation, actions, *, train: bool = False, **kwargs):
        del rng, observation, actions, train, kwargs
        raise AssertionError("the temporal head path must not call the action loss")

    def sample_actions(self, rng, observation, **kwargs):
        del rng, observation, kwargs
        raise AssertionError("training and validation must not sample actions")


class _TinyCache:
    """Small in-memory cache implementing TemporalFeatureDataset's cache contract."""

    def __init__(self, rows: tuple[temporal_completion_data.TemporalSampleRow, ...], histories: np.ndarray):
        self.rows = rows
        self.prefix_history = histories

    def indices_for_split(self, split: temporal_completion_data.SplitName) -> np.ndarray:
        return np.asarray([index for index, row in enumerate(self.rows) if row.split == split], dtype=np.int64)


class _NaturalLoader:
    """One-batch natural loader with the dataset metadata required by validation."""

    def __init__(self, dataset: temporal_completion_features.TemporalFeatureDataset):
        self.dataset = dataset

    def __iter__(self):
        examples = [self.dataset[index] for index in range(len(self.dataset))]
        histories, targets = zip(*examples, strict=True)
        yield jnp.asarray(np.stack(histories)), jnp.asarray(np.stack(targets))


def _validation_rows() -> tuple[temporal_completion_data.TemporalSampleRow, ...]:
    rows = []
    for tick in (30, 45, 60):
        label = int(tick == 60)
        rows.append(
            temporal_completion_data.TemporalSampleRow(
                trajectory_id="tiny-val-trajectory",
                full_episode_id=0,
                task_index=0,
                split="val",
                logical_tick=tick,
                label=label,
                sample_kind="positive" if label else "hard_negative",
                boundary_tick=60,
                prompt_index=0,
                history_logical_ticks=(tick - 30, tick - 15, tick),
                source_episode_ids=(0, 0, 0),
                source_frame_indices=(tick - 30, tick - 15, tick),
                terminal_hold_flags=(False, False, False),
            )
        )
    return tuple(rows)


def _train_state(model: _TinyTemporalModel, config: config_lib.TrainConfig) -> training_utils.TrainState:
    model_def, params = nnx.split(model)
    trainable_params = params.filter(config.trainable_filter)
    optimizer = optax.sgd(learning_rate=0.05)
    return training_utils.TrainState(
        step=jnp.asarray(0, dtype=jnp.int32),
        params=params,
        model_def=model_def,
        opt_state=optimizer.init(trainable_params),
        tx=optimizer,
        ema_decay=None,
        ema_params=None,
    )


def _head_parameters(model: _TinyTemporalModel) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(variable.value).copy()
        for _path, variable in nnx.state(model.completion_head, nnx.Param).flat_state().items()
    )


def test_cached_temporal_batch_trains_only_head_and_flows_through_validation_rank() -> None:
    config = config_lib.get_config("pi05_agilex_breakfast_temporal_completion_head")
    assert config.completion.uses_temporal_completion
    assert config.completion.temporal_input_mode == "history"

    histories = np.asarray(
        [
            [[-1.0, 0.2], [-0.8, 0.3], [-0.6, 0.4]],
            [[-0.4, 0.5], [-0.2, 0.6], [0.0, 0.7]],
            [[0.2, 0.8], [0.5, 0.9], [0.9, 1.0]],
        ],
        dtype=np.float16,
    )
    dataset = temporal_completion_features.TemporalFeatureDataset(
        _TinyCache(_validation_rows(), histories),  # type: ignore[arg-type]
        "val",
    )
    loader = _NaturalLoader(dataset)
    batch = next(iter(loader))
    assert batch[0].shape == (3, 3, 2)
    assert batch[0].dtype == jnp.float32
    assert batch[1].tolist() == [0.0, 0.0, 1.0]

    state = _train_state(_TinyTemporalModel(), config)
    before_model = nnx.merge(state.model_def, state.params)
    before_frozen = np.asarray(before_model.action_in_proj.value).copy()
    before_head = _head_parameters(before_model)
    before_logits = before_model.compute_temporal_completion_logits(jax.random.key(1), batch[0], train=False)
    expected_unweighted_bce = jnp.mean(completion_training.bce_with_logits(before_logits, batch[1]))

    # A deliberately large pos_weight must have no effect in temporal mode.
    new_state, info = train.train_step(
        config,
        jax.random.key(2),
        state,
        batch,
        pos_weight=99.0,
    )

    np.testing.assert_allclose(np.asarray(info["loss"]), np.asarray(expected_unweighted_bce), rtol=1e-6, atol=1e-6)
    assert float(info["pos_weight"]) == 1.0
    assert int(new_state.step) == 1

    after_model = nnx.merge(new_state.model_def, new_state.params)
    np.testing.assert_array_equal(np.asarray(after_model.action_in_proj.value), before_frozen)
    after_head = _head_parameters(after_model)
    assert any(not np.array_equal(before, after) for before, after in zip(before_head, after_head, strict=True))

    metrics, selection = train.evaluate_temporal_completion_loader(
        functools.partial(train.temporal_completion_eval_step, config),
        jax.random.key(3),
        new_state,
        loader,  # type: ignore[arg-type]
    )
    expected_rank_keys = {
        "val/temporal/boundary_top1_rate",
        "val/temporal/hard_local/auprc",
        "val/temporal/margin/hard_local_median",
        "val/temporal/natural/auprc",
    }
    assert expected_rank_keys <= metrics.keys()
    assert selection.selected_on_split == "val"
    assert train.temporal_validation_rank(metrics) == tuple(
        float(metrics[key])
        for key in (
            "val/temporal/boundary_top1_rate",
            "val/temporal/hard_local/auprc",
            "val/temporal/margin/hard_local_median",
            "val/temporal/natural/auprc",
        )
    )
