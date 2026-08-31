import json

import numpy as np

from openpi.training import breakfast_done_data
from scripts import train_token_progress_head


def _write_source(tmp_path):
    dataset_root = tmp_path / "dataset"
    annotation_root = tmp_path / "annotations"
    cache_root = tmp_path / "cache"
    shard_root = cache_root / "shard_0"
    (dataset_root / "meta").mkdir(parents=True)
    annotation_root.mkdir()
    shard_root.mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text(json.dumps({"fps": 30}), encoding="utf-8")
    with (dataset_root / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in range(10):
            handle.write(json.dumps({"episode_index": episode, "length": 160}) + "\n")
            task_end = {"status": "missing"} if episode == 0 else {"status": "ok", "frame": 150}
            annotation = {
                "index": episode,
                "sub_task": {
                    "subtask": ["task zero", "task one", "task two", "task three"],
                    "start_frame": [0, 40, 80, 120],
                    "end_frame": [40, 80, 120, 160],
                },
                "task_end": task_end,
            }
            (annotation_root / f"episode_{episode:06d}.json").write_text(
                json.dumps(annotation), encoding="utf-8"
            )
    dataset = breakfast_done_data.load_breakfast_done_dataset(dataset_root, annotation_root)
    np.save(shard_root / "sample_ids.npy", np.asarray([sample.sample_id for sample in dataset.samples]))
    return dataset_root, annotation_root, cache_root, dataset


def test_build_progress_targets_uses_task_boundaries_and_masks_missing_terminal(tmp_path):
    dataset_root, annotation_root, cache_root, dataset = _write_source(tmp_path)

    targets, valid = train_token_progress_head.build_progress_targets(
        cache_root, {"dataset_root": str(dataset_root)}, annotation_root
    )

    sample_rows = {sample.sample_id: index for index, sample in enumerate(dataset.samples)}
    first_boundary = next(
        sample for sample in dataset.samples if sample.episode_index == 1 and sample.sampling_source == "exact_boundary"
    )
    missing_terminal_rows = [
        row
        for row, sample in enumerate(dataset.samples)
        if sample.episode_index == 0 and sample.current_sub_task == breakfast_done_data.SUB_TASK_IDS[-1]
    ]
    assert targets[sample_rows[first_boundary.sample_id]] == 1.0
    assert missing_terminal_rows
    assert not np.any(valid[missing_terminal_rows])
    assert np.all((targets[valid] >= 0.0) & (targets[valid] <= 1.0))
