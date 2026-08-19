import dataclasses

import numpy as np
import pytest

from openpi.training import temporal_completion_data as temporal_data
from scripts import extract_temporal_completion_features as extract


def _row(
    *,
    prompt_index=0,
    episodes=(0, 0, 0),
    frames=None,
    label=1,
    sample_kind="positive",
):
    if frames is None:
        frames = (30, 45, 60) if label else (15, 30, 45)
    return temporal_data.TemporalSampleRow(
        trajectory_id="full-000000",
        full_episode_id=0,
        task_index=prompt_index,
        split="train",
        logical_tick=60 if label else 45,
        label=label,
        sample_kind=sample_kind,
        boundary_tick=60,
        prompt_index=prompt_index,
        history_logical_ticks=(30, 45, 60) if label else (15, 30, 45),
        source_episode_ids=episodes,
        source_frame_indices=frames,
        terminal_hold_flags=(False, False, False),
    )


def _transition_row(*, step: int = 0, task_index: int = 1):
    if step == 0:
        return temporal_data.TemporalSampleRow(
            trajectory_id="full-000000",
            full_episode_id=0,
            task_index=task_index,
            split="train",
            logical_tick=0,
            label=0,
            sample_kind="transition_negative",
            boundary_tick=60,
            prompt_index=task_index,
            history_logical_ticks=(-30, -15, 0),
            source_episode_ids=(0, 0, 1),
            source_frame_indices=(84, 99, 0),
            terminal_hold_flags=(False, False, False),
        )
    return temporal_data.TemporalSampleRow(
        trajectory_id="full-000000",
        full_episode_id=0,
        task_index=task_index,
        split="train",
        logical_tick=15,
        label=0,
        sample_kind="transition_negative",
        boundary_tick=60,
        prompt_index=task_index,
        history_logical_ticks=(-15, 0, 15),
        source_episode_ids=(0, 1, 1),
        source_frame_indices=(99, 0, 15),
        terminal_hold_flags=(False, False, False),
    )


PROMPTS = {0: "finish task zero", 1: "finish task one", 2: "finish task two", 3: "finish task three"}


def test_plan_never_reads_supervision_fields():
    class SourceOnlyRow:
        prompt_index = 0
        source_episode_ids = (0, 0, 0)
        source_frame_indices = (0, 15, 30)

        @property
        def label(self):
            raise AssertionError("feature planning must not inspect labels")

        @property
        def sample_kind(self):
            raise AssertionError("feature planning must not inspect sample kinds")

    plan = extract.build_prefix_feature_plan((SourceOnlyRow(),), PROMPTS)
    assert len(plan.keys) == 3


def test_plan_deduplicates_exact_source_frame_prompt_and_preserves_old_prompt():
    first = _row(label=0, sample_kind="hard_negative")
    second = dataclasses.replace(_transition_row(step=1), source_frame_indices=(45, 0, 15))
    plan = extract.build_prefix_feature_plan((first, second), PROMPTS)

    assert len(plan.keys) == 5
    assert plan.row_key_indices.shape == (2, 3)
    shared_old = next(
        key
        for key in plan.keys
        if key.source_episode_id == 0 and key.source_frame_index == 45 and key.prompt == PROMPTS[0]
    )
    assert plan.row_key_indices[0, 2] == plan.row_key_indices[1, 0] == plan.keys.index(shared_old)
    current = next(key for key in plan.keys if key.source_episode_id == 1 and key.source_frame_index == 0)
    assert current.prompt_index == 1
    assert current.prompt == PROMPTS[1]


def test_current_boundary_pixels_keep_new_prompt_and_deduplicate():
    task0 = _transition_row(step=0, task_index=1)
    task1 = temporal_data.TemporalSampleRow(
        trajectory_id="full-000000",
        full_episode_id=0,
        task_index=1,
        split="train",
        logical_tick=30,
        label=1,
        sample_kind="positive",
        boundary_tick=30,
        prompt_index=1,
        history_logical_ticks=(0, 15, 30),
        source_episode_ids=(1, 1, 1),
        source_frame_indices=(0, 15, 30),
        terminal_hold_flags=(False, False, False),
    )
    plan = extract.build_prefix_feature_plan((task0, task1), PROMPTS)

    same_pixel = [key for key in plan.keys if (key.source_episode_id, key.source_frame_index) == (1, 0)]
    assert {(key.prompt_index, key.prompt) for key in same_pixel} == {(1, PROMPTS[1])}


def test_assembly_uses_oldest_to_current_indices_and_requires_fp32_model_output():
    plan = extract.build_prefix_feature_plan((_row(),), PROMPTS)
    unique = np.stack(
        [np.full((4,), index, dtype=np.float32) for index in range(len(plan.keys))],
        axis=0,
    )
    history = extract.assemble_prefix_history(plan, unique, storage_dtype=np.float16)

    assert history.shape == (1, 3, 4)
    assert history.dtype == np.float16
    np.testing.assert_array_equal(history[0, :, 0], plan.row_key_indices[0].astype(np.float16))
    with pytest.raises(ValueError, match="must be FP32"):
        extract.assemble_prefix_history(plan, unique.astype(np.float16))
    with pytest.raises(ValueError, match="non-finite"):
        extract.assemble_prefix_history(plan, np.full_like(unique, np.nan))


def test_output_guard_rejects_data_checkpoint_and_manifest_roots(tmp_path):
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "checkpoint"
    manifests = tmp_path / "manifests"
    for path in (dataset, checkpoint, manifests):
        path.mkdir()

    for output in (dataset / "cache.npz", checkpoint / "cache.npz", manifests / "cache.npz"):
        with pytest.raises(ValueError, match="protected"):
            extract.assert_safe_output(output, (dataset, checkpoint, manifests))

    extract.assert_safe_output(tmp_path / "features" / "cache.npz", (dataset, checkpoint, manifests))
