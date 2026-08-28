import json

import pytest

from openpi.training import breakfast_done_data


def _write_source(tmp_path, *, changed_prompt_episode: int | None = None):
    dataset_root = tmp_path / "dataset"
    annotation_root = tmp_path / "annotations"
    (dataset_root / "meta").mkdir(parents=True)
    annotation_root.mkdir()
    (dataset_root / "meta" / "info.json").write_text(json.dumps({"fps": 30}), encoding="utf-8")
    with (dataset_root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in range(10):
            handle.write(json.dumps({"episode_index": episode, "length": 160}) + "\n")
            prompts = ["task zero", "task one", "task two", "task three"]
            if episode == changed_prompt_episode:
                prompts[-1] = "different task"
            annotation = {
                "index": episode,
                "sub_task": {
                    "subtask": prompts,
                    "start_frame": [0, 40, 80, 120],
                    "end_frame": [40, 80, 120, 160],
                },
                "task_end": {"status": "ok", "frame": 150},
            }
            (annotation_root / f"episode_{episode:06d}.json").write_text(json.dumps(annotation), encoding="utf-8")
    return dataset_root, annotation_root


def test_load_breakfast_done_dataset_builds_labels_and_feature_plan(tmp_path):
    dataset_root, annotation_root = _write_source(tmp_path)

    dataset = breakfast_done_data.load_breakfast_done_dataset(
        dataset_root,
        annotation_root,
        continue_drop_ratio=0,
    )

    assert len(dataset.split_episode_ids["train"]) == 9
    assert len(dataset.split_episode_ids["val"]) == 1
    assert tuple(dataset.task_prompts.values()) == ("task zero", "task one", "task two", "task three")
    assert {sample.label for sample in dataset.samples} == {0, 1}
    assert all(sample.history_frames[-1] == sample.query_frame for sample in dataset.samples)
    assert any(sample.sampling_source == "exact_boundary" for sample in dataset.samples)
    plan = breakfast_done_data.build_feature_plan(dataset.samples)
    assert plan.history_indices.shape == (len(dataset.samples), 3)
    assert len(plan.keys) < len(dataset.samples) * 3


def test_load_breakfast_done_dataset_rejects_prompt_drift(tmp_path):
    dataset_root, annotation_root = _write_source(tmp_path, changed_prompt_episode=1)

    with pytest.raises(ValueError, match="changes the ordered sub-task descriptions"):
        breakfast_done_data.load_breakfast_done_dataset(dataset_root, annotation_root)
