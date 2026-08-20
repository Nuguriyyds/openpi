import dataclasses

import pytest

import openpi.training.temporal_completion_sampler as _sampler


@dataclasses.dataclass(frozen=True)
class _Row:
    trajectory_id: str
    task_index: int
    logical_tick: int
    label: int
    sample_kind: str
    boundary_tick: int
    split: str = "train"


def _make_rows(trajectory_count: int = 24) -> list[_Row]:
    rows: list[_Row] = []
    for trajectory in range(trajectory_count):
        for task_index in _sampler.TEMPORAL_TASK_INDICES:
            boundary = 1000 + 300 * task_index
            common = {"trajectory_id": f"trajectory-{trajectory}", "task_index": task_index}
            rows.append(
                _Row(
                    **common,
                    logical_tick=boundary,
                    label=1,
                    sample_kind="positive",
                    boundary_tick=boundary,
                )
            )
            rows.extend(
                _Row(
                    **common,
                    logical_tick=boundary - tick_offset * _sampler.DEFAULT_TICK_STRIDE_FRAMES,
                    label=0,
                    sample_kind="hard_negative",
                    boundary_tick=boundary,
                )
                for tick_offset in _sampler.HARD_NEGATIVE_OFFSETS
            )
            rows.extend(
                _Row(
                    **common,
                    logical_tick=boundary - tick_offset * _sampler.DEFAULT_TICK_STRIDE_FRAMES,
                    label=0,
                    sample_kind="ordinary_negative",
                    boundary_tick=boundary,
                )
                for tick_offset in (6, 7)
            )
    return rows


def _event(row: _Row) -> tuple[str, int, int]:
    return (row.trajectory_id, row.task_index, row.boundary_tick)


def _make_history_carry_rows(trajectory_count: int = 12) -> list[_Row]:
    rows: list[_Row] = []
    for trajectory in range(trajectory_count):
        for task_index in _sampler.TEMPORAL_TASK_INDICES:
            boundary = 1000 + 300 * task_index
            common = {"trajectory_id": f"trajectory-{trajectory}", "task_index": task_index}
            rows.extend(
                (
                    _Row(**common, logical_tick=boundary, label=1, sample_kind="positive", boundary_tick=boundary),
                    _Row(
                        **common,
                        logical_tick=boundary - 15,
                        label=0,
                        sample_kind="hard_negative",
                        boundary_tick=boundary,
                    ),
                    _Row(
                        **common,
                        logical_tick=boundary - 30,
                        label=0,
                        sample_kind="ordinary_negative",
                        boundary_tick=boundary,
                    ),
                )
            )
        for task_index in (1, 2, 3):
            boundary = 1000 + 300 * task_index
            common = {"trajectory_id": f"trajectory-{trajectory}", "task_index": task_index}
            rows.extend(
                (
                    _Row(**common, logical_tick=0, label=0, sample_kind="transition_negative", boundary_tick=boundary),
                    _Row(**common, logical_tick=15, label=0, sample_kind="transition_negative", boundary_tick=boundary),
                )
            )
    return rows


def test_train_batch_has_exact_composition_unique_positives_and_paired_hard_negatives():
    rows = _make_rows()
    sampler = _sampler.TemporalCompletionBatchSampler(rows, seed=42, batches_per_epoch=4)

    for batch_index, batch in enumerate(sampler):
        selected = [rows[index] for index in batch]
        positives = [row for row in selected if row.sample_kind == "positive"]
        hard = [row for row in selected if row.sample_kind == "hard_negative"]
        ordinary = [row for row in selected if row.sample_kind == "ordinary_negative"]

        assert len(batch) == 64
        assert (len(positives), len(hard), len(ordinary)) == (32, 16, 16)
        assert len({(row.trajectory_id, row.task_index) for row in selected}) == 32
        assert {
            task_index: sum(row.task_index == task_index for row in positives)
            for task_index in _sampler.TEMPORAL_TASK_INDICES
        } == dict.fromkeys(_sampler.TEMPORAL_TASK_INDICES, 8)
        assert all(
            (row.trajectory_id, row.task_index, row.boundary_tick) in {_event(positive) for positive in positives}
            for row in hard
        )
        del batch_index


def test_history_carry_batch_is_task_balanced_and_rotates_transition_steps():
    rows = _make_history_carry_rows()
    sampler = _sampler.TemporalCompletionBatchSampler(
        rows,
        seed=42,
        batches_per_epoch=3,
        positive_per_batch=16,
        hard_negative_per_batch=16,
        ordinary_negative_per_batch=28,
        transition_negative_per_batch=4,
    )
    batches = list(sampler)
    for batch in batches:
        selected = [rows[index] for index in batch]
        audit = sampler.audit_batch(batch)
        assert audit.positive_task_counts == (4, 4, 4, 4)
        assert audit.hard_task_counts == (4, 4, 4, 4)
        assert audit.ordinary_task_counts == (7, 7, 7, 7)
        assert audit.transition_negative_count == 4
        assert [
            sum(row.sample_kind == kind for row in selected)
            for kind in (
                "positive",
                "hard_negative",
                "ordinary_negative",
                "transition_negative",
            )
        ] == [16, 16, 28, 4]
        assert [
            sum(row.sample_kind == "positive" and row.task_index == task for row in selected)
            for task in _sampler.TEMPORAL_TASK_INDICES
        ] == [4] * 4
        assert [
            sum(row.sample_kind == "ordinary_negative" and row.task_index == task for row in selected)
            for task in _sampler.TEMPORAL_TASK_INDICES
        ] == [7] * 4
        assert all(
            _event(row) in {_event(positive) for positive in selected if positive.sample_kind == "positive"}
            for row in selected
            if row.sample_kind == "hard_negative"
        )
    transition_counts = {(task, step): 0 for task in (1, 2, 3) for step in (0, 15)}
    for batch in batches:
        for index in batch:
            row = rows[index]
            if row.sample_kind == "transition_negative":
                transition_counts[(row.task_index, row.logical_tick)] += 1
    assert transition_counts == dict.fromkeys(transition_counts, 2)


def test_history_carry_32_16_12_4_keeps_positive_count_and_pairs_hard_subset():
    rows = _make_history_carry_rows()
    sampler = _sampler.TemporalCompletionBatchSampler(
        rows,
        seed=42,
        positive_per_batch=32,
        hard_negative_per_batch=16,
        ordinary_negative_per_batch=12,
        transition_negative_per_batch=4,
    )
    batch = next(iter(sampler))
    selected = [rows[index] for index in batch]
    audit = sampler.audit_batch(batch)

    assert audit.positive_count == 32
    assert audit.hard_negative_count == 16
    assert audit.ordinary_negative_count == 12
    assert audit.transition_negative_count == 4
    assert audit.positive_task_counts == (8, 8, 8, 8)
    assert audit.hard_task_counts == (4, 4, 4, 4)
    assert audit.ordinary_task_counts == (3, 3, 3, 3)
    positive_events = {_event(row) for row in selected if row.sample_kind == "positive"}
    assert all(_event(row) in positive_events for row in selected if row.sample_kind == "hard_negative")
    assert sampler.steps_per_epoch == 2


def test_task_remainder_rotates_evenly_across_batches_and_epochs():
    rows = _make_rows()
    sampler = _sampler.TemporalCompletionBatchSampler(rows, seed=3, batches_per_epoch=5)
    task_totals = dict.fromkeys(_sampler.TEMPORAL_TASK_INDICES, 0)

    for _ in range(4):
        for batch in sampler:
            for index in batch:
                row = rows[index]
                if row.sample_kind == "positive":
                    task_totals[row.task_index] += 1

    assert task_totals == {0: 160, 1: 160, 2: 160, 3: 160}


def test_ordinary_negatives_maximise_distinct_trajectories():
    rows = _make_rows(trajectory_count=24)
    sampler = _sampler.TemporalCompletionBatchSampler(rows, seed=8, batches_per_epoch=1)
    batch = next(iter(sampler))
    ordinary = [rows[index] for index in batch if rows[index].sample_kind == "ordinary_negative"]

    assert len({(row.trajectory_id, row.task_index) for row in ordinary}) == 16


def test_seed_epoch_and_resume_are_reproducible():
    rows = _make_rows()
    full = _sampler.TemporalCompletionBatchSampler(rows, seed=91, batches_per_epoch=6)
    full.set_epoch(7)
    expected = list(full)

    same = _sampler.TemporalCompletionBatchSampler(rows, seed=91, batches_per_epoch=6)
    same.set_epoch(7)
    assert list(same) == expected

    resumed = _sampler.TemporalCompletionBatchSampler(rows, seed=91, batches_per_epoch=6)
    resumed.set_epoch(7)
    resumed.set_skip_batches(2)
    state = resumed.state_dict()
    restored = _sampler.TemporalCompletionBatchSampler(rows, seed=91, batches_per_epoch=6)
    restored.load_state_dict(state)
    assert len(restored) == 4
    assert list(restored) == expected[2:]

    different_epoch = _sampler.TemporalCompletionBatchSampler(rows, seed=91, batches_per_epoch=6)
    different_epoch.set_epoch(8)
    assert list(different_epoch) != expected


def test_mapping_rows_are_supported():
    rows = [dataclasses.asdict(row) for row in _make_rows()]
    sampler = _sampler.TemporalCompletionBatchSampler(rows, seed=1, batches_per_epoch=1)
    assert len(next(iter(sampler))) == 64


def test_train_sampler_rejects_non_train_rows():
    rows = _make_rows()
    rows[0] = dataclasses.replace(rows[0], split="val")
    with pytest.raises(ValueError, match="only split='train'"):
        _sampler.TemporalCompletionBatchSampler(rows, seed=1)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda rows: rows.__setitem__(1, dataclasses.replace(rows[1], logical_tick=rows[1].boundary_tick - 75)),
            "exactly E-15",
        ),
        (
            lambda rows: rows.__setitem__(4, dataclasses.replace(rows[4], logical_tick=rows[4].boundary_tick - 1)),
            "positive endpoint",
        ),
        (
            lambda rows: rows.__setitem__(0, dataclasses.replace(rows[0], label=0)),
            "inconsistent",
        ),
        (
            lambda rows: rows.__setitem__(0, dataclasses.replace(rows[0], logical_tick=rows[0].logical_tick + 1)),
            "positive endpoint",
        ),
    ],
)
def test_invalid_candidate_metadata_fails_closed(mutate, message: str):
    rows = _make_rows()
    mutate(rows)
    with pytest.raises(ValueError, match=message):
        _sampler.TemporalCompletionBatchSampler(rows, seed=1)


def test_duplicate_positive_event_fails_closed():
    rows = _make_rows()
    with pytest.raises(ValueError, match="duplicate temporal candidate row"):
        _sampler.TemporalCompletionBatchSampler([*rows, rows[0]], seed=1)


def test_missing_event_local_hard_fails_closed():
    rows = _make_rows(trajectory_count=8)
    event = _event(rows[0])
    without_pair = [row for row in rows if not (row.sample_kind == "hard_negative" and _event(row) == event)]
    del event
    with pytest.raises(ValueError, match="task 0 hard pool has only 7"):
        _sampler.TemporalCompletionBatchSampler(without_pair, seed=1, batches_per_epoch=1)


def test_missing_entire_task_hard_pool_fails_closed():
    rows = [row for row in _make_rows() if not (row.sample_kind == "hard_negative" and row.task_index == 3)]
    with pytest.raises(ValueError, match=r"task 3 hard pool has only 0"):
        _sampler.TemporalCompletionBatchSampler(rows, seed=1)


def test_each_task_requires_eight_unique_positive_events():
    rows = [
        row
        for row in _make_rows(trajectory_count=8)
        if not (row.trajectory_id == "trajectory-5" and row.task_index == 3)
    ]
    with pytest.raises(ValueError, match="task 3 needs at least 8 positive"):
        _sampler.TemporalCompletionBatchSampler(rows, seed=1)


def test_natural_eval_batches_preserve_all_rows_once_and_keep_partial_batch():
    rows = _make_rows(trajectory_count=1)
    sampler = _sampler.NaturalTemporalEvalBatchSampler(rows, batch_size=9)
    batches = list(sampler)

    assert len(batches) == 2
    assert [index for batch in batches for index in batch] == list(range(len(rows)))
    assert len(batches[-1]) == 7


def test_distributed_loading_is_explicitly_rejected():
    with pytest.raises(ValueError, match="single-process"):
        _sampler.TemporalCompletionBatchSampler(_make_rows(), seed=1, num_replicas=2, rank=0)
