import argparse
from pathlib import Path

import numpy as np
import pytest

from scripts import quantify_clean_completion_features as quantify


def test_virtual_boundary_set_has_expected_labels_and_prompt_pairs():
    lengths = dict.fromkeys(range(12), 80)
    specs, audit = quantify.build_virtual_sample_specs(
        lengths,
        negative_stride=15,
        copy_frames=5,
        positive_tail=5,
        subtask4_tail=10,
        hard_negative_frames=30,
        split_seed=3,
        val_fraction=0.2,
        test_fraction=0.2,
    )

    assert audit["valid_group_count"] == 3
    for episode in range(12):
        episode_specs = [spec for spec in specs if spec.logical_episode == episode]
        assert sum(spec.completion for spec in episode_specs) == 10

    pairs: dict[int, list[quantify.SampleSpec]] = {}
    for spec in specs:
        if spec.pair_id >= 0:
            pairs.setdefault(spec.pair_id, []).append(spec)
    assert len(pairs) == 3 * 3 * 5
    for rows in pairs.values():
        assert len(rows) == 2
        assert sorted(row.completion for row in rows) == [0, 1]
        assert len({(row.source_episode, row.source_frame) for row in rows}) == 1
        assert len({row.task_index for row in rows}) == 2
        assert len({row.group_index for row in rows}) == 1
        assert len({row.split for row in rows}) == 1


def test_virtual_boundary_set_skips_incomplete_or_short_groups():
    lengths = dict.fromkeys(range(8), 80)
    lengths[3] = 9
    del lengths[6]
    # No valid groups remain, so split construction should reject the input
    # instead of silently producing empty train/val/test subsets.
    with pytest.raises(ValueError, match="not enough groups"):
        quantify.build_virtual_sample_specs(
            lengths,
            split_seed=1,
            val_fraction=0.2,
            test_fraction=0.2,
        )


def test_metrics_reward_correct_ranking_and_prompt_contrast():
    target = np.array([0, 0, 1, 1], dtype=np.int8)
    score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = quantify.binary_metrics(target, score, threshold=0.5)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["usable_information_bits"] > 0

    pair_target = np.array([1, 0, 1, 0], dtype=np.int8)
    pair_score = np.array([0.9, 0.2, 0.4, 0.6])
    pair_id = np.array([10, 10, 11, 11], dtype=np.int64)
    paired = quantify.paired_prompt_accuracy(pair_target, pair_score, pair_id)
    assert paired["pair_count"] == 2
    assert paired["accuracy"] == 0.5


def test_merge_feature_shards_validates_and_combines_rows(tmp_path: Path):
    for shard_index in range(2):
        row = np.array([shard_index], dtype=np.int32)
        np.savez_compressed(
            tmp_path / f"features_shard_{shard_index:03d}_of_002.npz",
            prefix_mean=np.ones((1, 2), dtype=np.float16) * shard_index,
            prefix_last=np.ones((1, 2), dtype=np.float16),
            action_hidden=np.ones((1, 1), dtype=np.float16),
            action_hidden_noise_std=np.zeros(1, dtype=np.float32),
            logical_episode=row,
            source_episode=row,
            source_frame=np.zeros(1, dtype=np.int32),
            split=np.array([shard_index], dtype=np.int8),
            completion=np.array([shard_index], dtype=np.int8),
        )
    args = argparse.Namespace(output_dir=tmp_path, num_shards=2)
    merged_path = quantify.merge_feature_shards(args)
    with np.load(merged_path) as merged:
        assert merged["prefix_mean"].shape == (2, 2)
        assert merged["completion"].tolist() == [0, 1]


def test_analyze_selects_action_feature_on_group_disjoint_synthetic_data(tmp_path: Path):
    rng = np.random.default_rng(7)
    rows_per_split = 120
    split = np.repeat(
        np.array([quantify.SPLIT_TRAIN, quantify.SPLIT_VAL, quantify.SPLIT_TEST], dtype=np.int8),
        rows_per_split,
    )
    target = rng.integers(0, 2, size=len(split), dtype=np.int8)
    progress = np.clip(0.15 + 0.7 * target + rng.normal(0, 0.05, size=len(split)), 0, 1).astype(np.float32)
    prefix = rng.normal(size=(len(split), 6)).astype(np.float32)
    action = np.column_stack(
        [
            target * 4 - 2 + rng.normal(0, 0.2, size=len(split)),
            progress * 3 + rng.normal(0, 0.1, size=len(split)),
            rng.normal(size=(len(split), 2)),
        ]
    ).astype(np.float32)
    groups = np.concatenate([np.repeat(np.arange(offset, offset + 12), 10) for offset in (0, 20, 40)]).astype(np.int32)
    logical_episode = groups * 4
    logical_frame = np.tile(np.arange(10), len(split) // 10).astype(np.int32)
    kind = np.where(target == 1, quantify.KIND_TAIL_POSITIVE, quantify.KIND_HARD_NEGATIVE).astype(np.int8)
    feature_file = tmp_path / "features.npz"
    np.savez_compressed(
        feature_file,
        prefix_mean=prefix.astype(np.float16),
        prefix_last=prefix.astype(np.float16),
        action_hidden=action.astype(np.float16),
        action_hidden_noise_std=np.zeros(len(split), dtype=np.float32),
        completion=target,
        progress=progress,
        split=split,
        kind=kind,
        pair_id=np.full(len(split), -1, dtype=np.int64),
        logical_episode=logical_episode,
        logical_frame=logical_frame,
        group_index=groups,
    )
    args = argparse.Namespace(
        feature_file=feature_file,
        output_dir=tmp_path,
        l2_grid=(1e-3,),
        max_iter=80,
        bootstrap_samples=20,
        bootstrap_seed=11,
    )
    summary_path = quantify.analyze_features(args)
    summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
    assert summary["representations"]["action_hidden"]["completion_probe"]["overall"]["auprc"] > 0.98
    assert (
        summary["representations"]["action_hidden"]["completion_probe"]["overall"]["auprc"]
        > summary["representations"]["prefix_mean"]["completion_probe"]["overall"]["auprc"]
    )
