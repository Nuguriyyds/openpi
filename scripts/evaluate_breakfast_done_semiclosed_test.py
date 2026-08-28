from typing import ClassVar

import numpy as np

from openpi.training.breakfast_done_data import DoneSample
from scripts import evaluate_breakfast_done_semiclosed


class _Dataset:
    episode_data_index: ClassVar = {
        "from": np.asarray([0, 1000]),
        "to": np.asarray([900, 1900]),
    }


def _sample(episode: int, task: str, frame: int, sample_type: str) -> DoneSample:
    return DoneSample(
        sample_id=f"{episode}-{task}",
        split="val",
        episode_index=episode,
        query_frame=frame,
        history_frames=(frame - 30, frame - 15, frame),
        current_sub_task=task,
        prompt=task,
        label=1,
        sample_type=sample_type,
        sampling_source="base",
    )


def test_build_specs_uses_qwen_tick_boundaries_and_excludes_missing_terminal():
    prompts = {f"task_{index}": f"prompt {index}" for index in range(4)}
    samples = [
        *(_sample(0, f"task_{index}", 150 * (index + 1), "transition") for index in range(3)),
        _sample(0, "task_3", 750, "terminal"),
        *(_sample(1, f"task_{index}", 150 * (index + 1), "transition") for index in range(3)),
    ]

    specs, missing_terminal = evaluate_breakfast_done_semiclosed._build_specs(  # noqa: SLF001
        samples=samples,
        test_episode_ids=(0, 1),
        task_prompts=prompts,
        dataset=_Dataset(),
    )

    assert len(specs) == 1
    assert specs[0].gt_end_frames == (150, 300, 450, 750)
    assert specs[0].subtask_start_frames == (0, 165, 315, 465)
    assert missing_terminal == (1,)
