from __future__ import annotations

import dataclasses

import numpy as np

from openpi.training import temporal_completion_data as data
from openpi.training import temporal_completion_sampler as sampler


def _group(group_id: int, *, length: int = 92) -> data.SubtaskGroupRecord:
    episodes = tuple(
        data.SubtaskEpisodeRecord(group_id * 4 + task, task, length)
        for task in range(4)
    )
    return data.build_subtask_groups(episodes)[0]


def test_subtask_reverse_index_uses_one_episode_and_prompt() -> None:
    rows = data.build_temporal_sample_rows(_group(0), trajectory_id="g0", full_episode_id=0, split="train")
    task0 = [row for row in rows if row.task_index == 0]
    positive = next(row for row in task0 if row.sample_kind == "positive")
    hard = next(row for row in task0 if row.sample_kind == "hard_negative")
    ordinary = [row for row in task0 if row.sample_kind == "ordinary_negative"]

    assert positive.history_logical_ticks == (61, 76, 91)
    assert hard.history_logical_ticks == (46, 61, 76)
    assert [row.logical_tick for row in ordinary] == [61, 46, 31]
    for row in (positive, hard, *ordinary):
        assert len(set(row.source_episode_ids)) == 1
        assert row.prompt_index == row.task_index == 0
        assert np.diff(row.history_logical_ticks).tolist() == [15, 15]


@dataclasses.dataclass(frozen=True)
class _Row:
    trajectory_id: str
    task_index: int
    logical_tick: int
    label: int
    sample_kind: str
    boundary_tick: int
    split: str = "train"


def _sampler_rows() -> list[_Row]:
    rows: list[_Row] = []
    for group_id in range(12):
        for task in range(4):
            trajectory = f"g{group_id}"
            endpoint = 90 + (group_id % 3)
            rows.extend(
                (
                    _Row(trajectory, task, endpoint, 1, "positive", endpoint),
                    _Row(trajectory, task, endpoint - 15, 0, "hard_negative", endpoint),
                    _Row(trajectory, task, endpoint - 30, 0, "ordinary_negative", endpoint),
                )
            )
    return rows


def test_subtask_pair_sampler_has_32_16_16_and_eight_per_task() -> None:
    rows = _sampler_rows()
    batch_sampler = sampler.TemporalCompletionBatchSampler(rows, seed=42, batches_per_epoch=1)
    batch = next(iter(batch_sampler))
    selected = [rows[index] for index in batch]
    assert len(batch) == 64
    assert [sum(row.sample_kind == kind for row in selected) for kind in ("positive", "hard_negative", "ordinary_negative")] == [32, 16, 16]
    assert [sum(row.sample_kind == "positive" and row.task_index == task for row in selected) for task in range(4)] == [8] * 4
    assert [sum(row.sample_kind == "hard_negative" and row.task_index == task for row in selected) for task in range(4)] == [4] * 4
    assert [sum(row.sample_kind == "ordinary_negative" and row.task_index == task for row in selected) for task in range(4)] == [4] * 4
    assert len({(row.trajectory_id, row.task_index) for row in selected}) == 32
