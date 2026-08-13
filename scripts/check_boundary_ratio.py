"""Project boundary-label class ratios before generating the derived dataset.

This script only reads the source dataset's ``meta/episodes.jsonl``.  It
reproduces the deterministic group split and computes the exact counts implied
by the boundary-label and sparse-negative-sampling rules, without creating a
dataset, manifest, dataloader, or training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random


def _load_lengths(source_root: Path) -> dict[int, int]:
    path = source_root / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing source metadata: {path}")
    lengths: dict[int, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        episode_id = int(record["episode_index"])
        length = int(record["length"])
        if episode_id in lengths:
            raise ValueError(f"Duplicate episode_index {episode_id} at {path}:{line_number}")
        if length <= 0:
            raise ValueError(f"Episode {episode_id} has non-positive length {length}")
        lengths[episode_id] = length
    if not lengths:
        raise ValueError(f"No episodes found in {path}")
    return lengths


def _build_split(
    episode_ids: list[int], *, episodes_per_group: int, test_groups: int, seed: int
) -> tuple[list[int], list[int], dict[int, int]]:
    if episode_ids != list(range(len(episode_ids))):
        raise ValueError("Boundary generation requires episode IDs to be contiguous and start at 0")
    if len(episode_ids) % episodes_per_group:
        raise ValueError(f"Episode count {len(episode_ids)} is not divisible by group size {episodes_per_group}")
    groups = [
        episode_ids[start : start + episodes_per_group] for start in range(0, len(episode_ids), episodes_per_group)
    ]
    if len(groups) <= test_groups:
        raise ValueError(f"Need at least one train group after reserving {test_groups} test groups")
    random.Random(seed).shuffle(groups)
    test = groups[:test_groups]
    train = groups[test_groups:]
    positions = {episode_id: position for group in groups for position, episode_id in enumerate(group)}
    return (
        [episode for group in train for episode in group],
        [episode for group in test for episode in group],
        positions,
    )


def _episode_counts(
    length: int,
    group_position: int,
    *,
    excluded: bool,
    has_copy_source: bool,
    stride: int,
    forced_first_n: int,
) -> dict[str, int]:
    # Subtasks 1-3: five original tail positives plus five copied positives.
    # Subtask 4: ten original tail positives and no copies.
    positive_original_tail = 5 if group_position < 3 else 10
    copy_count = 5 if group_position < 3 and not excluded and has_copy_source else 0
    positive_count = 0 if excluded else positive_original_tail + copy_count
    negative_source_indices = set(range(length if excluded else length - positive_original_tail))
    ordinary_negative = {index for index in negative_source_indices if index % stride == 0}
    forced_negative = (
        {index for index in range(forced_first_n) if index in negative_source_indices}
        if group_position in (1, 2, 3)
        else set()
    )
    sampled_negative = ordinary_negative | forced_negative

    return {
        "full_positive": positive_count,
        "full_negative": len(negative_source_indices),
        "sampled_positive": positive_count,
        "sampled_negative": len(sampled_negative),
        "ordinary_negative": len(ordinary_negative),
        "forced_negative_added": len(forced_negative - ordinary_negative),
    }


def _ratio(positive: int, negative: int) -> dict[str, int | float | str]:
    total = positive + negative
    return {
        "positive": positive,
        "negative": negative,
        "ratio": f"1:{negative / positive:.4f}" if positive else "N/A",
        "positive_percent": round(100.0 * positive / total, 4) if total else 0.0,
    }


def _summarize(episode_ids: list[int], counts: dict[int, dict[str, int]]) -> dict[str, object]:
    full_positive = sum(counts[episode]["full_positive"] for episode in episode_ids)
    full_negative = sum(counts[episode]["full_negative"] for episode in episode_ids)
    sampled_positive = sum(counts[episode]["sampled_positive"] for episode in episode_ids)
    sampled_negative = sum(counts[episode]["sampled_negative"] for episode in episode_ids)
    return {
        "episode_count": len(episode_ids),
        "full": _ratio(full_positive, full_negative),
        "sampled": _ratio(sampled_positive, sampled_negative),
        "sampling_detail": {
            "ordinary_negative": sum(counts[episode]["ordinary_negative"] for episode in episode_ids),
            "forced_negative_added": sum(counts[episode]["forced_negative_added"] for episode in episode_ids),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes-per-group", type=int, default=4)
    parser.add_argument("--test-groups", type=int, default=10)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--forced-first-n", type=int, default=5)
    args = parser.parse_args()
    if args.episodes_per_group != 4:
        parser.error("the boundary label scheme requires --episodes-per-group=4")
    if args.stride <= 0:
        parser.error("--stride must be positive")
    if args.forced_first_n < 0:
        parser.error("--forced-first-n must be non-negative")

    lengths = _load_lengths(args.source_root)
    episode_ids = sorted(lengths)
    train_ids, test_ids, positions = _build_split(
        episode_ids,
        episodes_per_group=args.episodes_per_group,
        test_groups=args.test_groups,
        seed=args.seed,
    )
    excluded_episode_ids = {
        episode_id for episode_id in episode_ids if lengths[episode_id] < (10 if positions[episode_id] == 3 else 5)
    }
    counts = {
        episode_id: _episode_counts(
            lengths[episode_id],
            positions[episode_id],
            excluded=episode_id in excluded_episode_ids,
            has_copy_source=any(
                candidate not in excluded_episode_ids
                for candidate in range(
                    episode_id + 1,
                    episode_id - positions[episode_id] + args.episodes_per_group,
                )
            ),
            stride=args.stride,
            forced_first_n=args.forced_first_n,
        )
        for episode_id in episode_ids
    }

    result = {
        "mode": "pre_generation_projection",
        "source_root": str(args.source_root),
        "parameters": {
            "seed": args.seed,
            "episodes_per_group": args.episodes_per_group,
            "test_groups": args.test_groups,
            "negative_stride": args.stride,
            "forced_first_n": args.forced_first_n,
        },
        "excluded_episode_indices": sorted(excluded_episode_ids),
        "train": _summarize(train_ids, counts),
        "test": _summarize(test_ids, counts),
        "all": _summarize(episode_ids, counts),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
