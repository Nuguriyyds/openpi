import dataclasses
import hashlib

import numpy as np
import pytest

from openpi.training import temporal_completion_data as temporal_data
from scripts import extract_temporal_completion_features as extract


def _row(
    *,
    prompt_index=0,
    episodes=(0, 0, 1),
    frames=(30, 45, 0),
    label=1,
    sample_kind="positive",
):
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
    second = dataclasses.replace(
        _row(),
        source_episode_ids=(0, 1, 1),
        source_frame_indices=(45, 0, 15),
    )
    plan = extract.build_prefix_feature_plan((first, second), PROMPTS)

    assert len(plan.keys) == 4
    assert plan.row_key_indices.shape == (2, 3)
    assert plan.row_key_indices[0, 1] == plan.row_key_indices[1, 0]
    assert plan.row_key_indices[0, 2] == plan.row_key_indices[1, 1]
    cross_boundary = next(key for key in plan.keys if key.source_episode_id == 1 and key.source_frame_index == 0)
    assert cross_boundary.prompt_index == 0
    assert cross_boundary.prompt == PROMPTS[0]


def test_same_pixels_with_different_prompt_are_distinct_feature_keys():
    task0 = _row(prompt_index=0, episodes=(0, 1, 1), frames=(45, 0, 15))
    task1 = dataclasses.replace(
        _row(prompt_index=1, episodes=(1, 1, 1), frames=(0, 15, 30)),
        trajectory_id="full-000001",
        full_episode_id=1,
    )
    plan = extract.build_prefix_feature_plan((task0, task1), PROMPTS)

    same_pixel = [key for key in plan.keys if (key.source_episode_id, key.source_frame_index) == (1, 0)]
    assert {(key.prompt_index, key.prompt) for key in same_pixel} == {
        (0, PROMPTS[0]),
        (1, PROMPTS[1]),
    }

    duplicate_prompt_plan = extract.build_prefix_feature_plan(
        (task0, task1),
        {**PROMPTS, 1: PROMPTS[0]},
    )
    duplicate_pixel = [
        key
        for key in duplicate_prompt_plan.keys
        if (key.source_episode_id, key.source_frame_index, key.prompt) == (1, 0, PROMPTS[0])
    ]
    assert len(duplicate_pixel) == 1


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


def test_metadata_and_tree_fingerprints_change_on_content_change(tmp_path):
    dataset = tmp_path / "dataset"
    meta = dataset / "meta"
    nested = meta / "tasks"
    nested.mkdir(parents=True)
    info = meta / "info.json"
    tasks = nested / "tasks.jsonl"
    info.write_text('{"fps": 30}', encoding="utf-8")
    tasks.write_text('{"task_index": 0}', encoding="utf-8")

    files = extract.metadata_files(dataset)
    assert files == tuple(sorted((info.resolve(), tasks.resolve()), key=lambda path: path.as_posix()))
    first = temporal_data.fingerprint_files(files)
    tasks.write_text('{"task_index": 1}', encoding="utf-8")
    second = temporal_data.fingerprint_files(extract.metadata_files(dataset))
    assert first != second
    assert extract.tree_fingerprint(meta) == second


@dataclasses.dataclass(frozen=True)
class _TinyConfig:
    width: int
    values: np.ndarray


def test_preprocess_fingerprint_is_stable_and_sensitive_to_every_identity_input():
    assets = hashlib.sha256(b"assets").hexdigest()
    code = hashlib.sha256(b"code").hexdigest()
    model = _TinyConfig(width=4, values=np.asarray([1.0, 2.0], dtype=np.float32))
    data = _TinyConfig(width=3, values=np.asarray([3], dtype=np.int16))

    kwargs = {
        "model_config": model,
        "data_config": data,
        "prompts": PROMPTS,
        "checkpoint_assets_fingerprint": assets,
        "code_fingerprint": code,
    }
    first = extract.make_preprocess_fingerprint(**kwargs)
    assert first == extract.make_preprocess_fingerprint(**kwargs)
    assert len(first) == 64
    assert first != extract.make_preprocess_fingerprint(**{**kwargs, "prompts": {**PROMPTS, 0: "changed"}})
    assert first != extract.make_preprocess_fingerprint(
        **{**kwargs, "model_config": _TinyConfig(width=5, values=model.values)}
    )
    assert first != extract.make_preprocess_fingerprint(
        **{**kwargs, "checkpoint_assets_fingerprint": hashlib.sha256(b"new assets").hexdigest()}
    )


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
