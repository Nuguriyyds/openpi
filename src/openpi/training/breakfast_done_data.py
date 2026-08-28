"""Build breakfast done samples directly from LeRobot boundary annotations."""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import json
from pathlib import Path
import random
from typing import Any, Literal

import numpy as np

Split = Literal["train", "val"]
SampleType = Literal["continue", "transition", "terminal"]
SUB_TASK_IDS = (
    "load_bread_into_toaster",
    "activate_toaster",
    "pour_drink_into_cup",
    "place_toasted_bread_on_plate",
)


@dataclasses.dataclass(frozen=True)
class DoneSample:
    sample_id: str
    split: Split
    episode_index: int
    query_frame: int
    history_frames: tuple[int, int, int]
    current_sub_task: str
    prompt: str
    label: int
    sample_type: SampleType
    sampling_source: str


@dataclasses.dataclass(frozen=True)
class BreakfastEpisode:
    index: int
    length: int
    stage_starts: tuple[int, int, int, int]
    terminal_frame: int | None

    @property
    def sampling_end_frame(self) -> int:
        return self.length - 1 if self.terminal_frame is None else self.terminal_frame


@dataclasses.dataclass(frozen=True)
class BreakfastDoneDataset:
    samples: tuple[DoneSample, ...]
    episodes: tuple[BreakfastEpisode, ...]
    task_prompts: dict[str, str]
    split_episode_ids: dict[Split, tuple[int, ...]]


@dataclasses.dataclass(frozen=True, order=True)
class FeatureKey:
    episode_index: int
    frame_index: int
    prompt: str


@dataclasses.dataclass(frozen=True)
class FeaturePlan:
    keys: tuple[FeatureKey, ...]
    history_indices: np.ndarray


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_lengths(dataset_root: Path) -> tuple[int, dict[int, int]]:
    info = _read_json(dataset_root / "meta" / "info.json")
    fps = int(info["fps"])
    lengths: dict[int, int] = {}
    with (dataset_root / "meta" / "episodes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                lengths[int(item["episode_index"])] = int(item["length"])
    return fps, lengths


def _load_episodes(
    dataset_root: Path,
    annotation_root: Path,
) -> tuple[tuple[BreakfastEpisode, ...], dict[str, str]]:
    _, lengths = _episode_lengths(dataset_root)
    episodes: list[BreakfastEpisode] = []
    task_prompts: dict[str, str] | None = None
    paths = sorted(annotation_root.glob("episode_*.json"))
    if not paths:
        raise FileNotFoundError(f"no boundary annotations found under {annotation_root}")

    for path in paths:
        annotation = _read_json(path)
        episode_index = int(annotation["index"])
        if episode_index not in lengths:
            raise ValueError(f"annotation {path} references an unknown episode")
        length = lengths[episode_index]
        sub_tasks = annotation["sub_task"]
        descriptions = tuple(str(value).strip() for value in sub_tasks["subtask"])
        starts = tuple(int(value) for value in sub_tasks["start_frame"])
        ends = tuple(int(value) for value in sub_tasks["end_frame"])
        if not (len(descriptions) == len(starts) == len(ends) == len(SUB_TASK_IDS)):
            raise ValueError(f"annotation {path} must contain exactly four sub-tasks")
        prompts = dict(zip(SUB_TASK_IDS, descriptions, strict=True))
        if task_prompts is None:
            task_prompts = prompts
        elif prompts != task_prompts:
            raise ValueError(f"annotation {path} changes the ordered sub-task descriptions")
        if any(end != next_start for end, next_start in zip(ends[:-1], starts[1:], strict=True)):
            raise ValueError(f"annotation {path} contains a gap or overlap")
        if not all(0 <= start < end <= length for start, end in zip(starts, ends, strict=True)):
            raise ValueError(f"annotation {path} contains an invalid sub-task range")

        task_end = annotation.get("task_end")
        terminal_frame = None
        if task_end is not None and task_end.get("status") == "ok":
            terminal_frame = int(task_end["frame"])
            if not starts[-1] <= terminal_frame < length:
                raise ValueError(f"annotation {path} has a task_end outside the final sub-task")
        episodes.append(
            BreakfastEpisode(
                index=episode_index,
                length=length,
                stage_starts=starts,  # type: ignore[arg-type]
                terminal_frame=terminal_frame,
            )
        )
    assert task_prompts is not None
    return tuple(episodes), task_prompts


def _assign_splits(
    episode_ids: Sequence[int],
    *,
    seed: int,
    val_ratio: float,
) -> tuple[dict[int, Split], dict[Split, tuple[int, ...]]]:
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between zero and one")
    shuffled = sorted(episode_ids)
    random.Random(seed).shuffle(shuffled)
    val_count = round(len(shuffled) * val_ratio)
    if val_count == 0:
        raise ValueError("validation split is empty")
    val_ids = set(shuffled[:val_count])
    split_ids: dict[Split, tuple[int, ...]] = {
        "train": tuple(sorted(set(episode_ids) - val_ids)),
        "val": tuple(sorted(val_ids)),
    }
    return (
        {episode_id: "val" if episode_id in val_ids else "train" for episode_id in episode_ids},
        split_ids,
    )


def _history(query_frame: int, history_step: int) -> tuple[int, int, int]:
    return max(0, query_frame - 2 * history_step), max(0, query_frame - history_step), query_frame


def _sample_episode(
    episode: BreakfastEpisode,
    split: Split,
    offset: int,
    task_prompts: dict[str, str],
    *,
    stride: int,
    history_step: int,
) -> list[DoneSample]:
    samples: list[DoneSample] = []
    current_index = 0
    for query_frame in range(offset, episode.length, stride):
        if query_frame < episode.stage_starts[0]:
            continue
        sample_type: SampleType = "continue"
        if current_index + 1 < len(SUB_TASK_IDS) and query_frame >= episode.stage_starts[current_index + 1]:
            sample_type = "transition"
        elif (
            current_index == len(SUB_TASK_IDS) - 1
            and episode.terminal_frame is not None
            and query_frame >= episode.terminal_frame
        ):
            sample_type = "terminal"
        task_id = SUB_TASK_IDS[current_index]
        samples.append(
            DoneSample(
                sample_id=f"ep{episode.index:06d}_o{offset:02d}_q{query_frame:06d}",
                split=split,
                episode_index=episode.index,
                query_frame=query_frame,
                history_frames=_history(query_frame, history_step),
                current_sub_task=task_id,
                prompt=task_prompts[task_id],
                label=int(sample_type != "continue"),
                sample_type=sample_type,
                sampling_source="base",
            )
        )
        if sample_type == "terminal":
            break
        if sample_type == "transition":
            current_index += 1
    return samples


def _exact_boundary_samples(
    episode: BreakfastEpisode,
    task_prompts: dict[str, str],
    *,
    stride: int,
    history_step: int,
) -> list[DoneSample]:
    boundaries: list[tuple[int, int, SampleType]] = [
        (index - 1, frame, "transition") for index, frame in enumerate(episode.stage_starts[1:], start=1)
    ]
    if episode.terminal_frame is not None:
        boundaries.append((len(SUB_TASK_IDS) - 1, episode.terminal_frame, "terminal"))
    return [
        DoneSample(
            sample_id=f"ep{episode.index:06d}_o{frame % stride:02d}_q{frame:06d}_exact_boundary",
            split="train",
            episode_index=episode.index,
            query_frame=frame,
            history_frames=_history(frame, history_step),
            current_sub_task=SUB_TASK_IDS[task_index],
            prompt=task_prompts[SUB_TASK_IDS[task_index]],
            label=1,
            sample_type=sample_type,
            sampling_source="exact_boundary",
        )
        for task_index, frame, sample_type in boundaries
    ]


def _drop_train_continues(
    samples: list[DoneSample],
    episodes: Sequence[BreakfastEpisode],
    *,
    ratio: float,
    seed: int,
) -> list[DoneSample]:
    if not 0 <= ratio < 1:
        raise ValueError("continue_drop_ratio must be in [0, 1)")
    episode_by_id = {episode.index: episode for episode in episodes}
    task_index = {task_id: index for index, task_id in enumerate(SUB_TASK_IDS)}
    continues = [sample for sample in samples if sample.split == "train" and sample.sample_type == "continue"]
    candidates: list[str] = []
    for sample in continues:
        episode = episode_by_id[sample.episode_index]
        index = task_index[sample.current_sub_task]
        stage_start = episode.stage_starts[index]
        stage_end = episode.stage_starts[index + 1] if index + 1 < len(SUB_TASK_IDS) else episode.sampling_end_frame + 1
        progress = (sample.query_frame - stage_start) / (stage_end - stage_start)
        if 0.15 <= progress <= 0.85:
            candidates.append(sample.sample_id)
    drop_count = round(len(continues) * ratio)
    if drop_count > len(candidates):
        raise ValueError("middle Continue samples cannot satisfy continue drop ratio")
    dropped = set(random.Random(seed).sample(candidates, drop_count))
    return [sample for sample in samples if sample.sample_id not in dropped]


def load_breakfast_done_dataset(
    dataset_root: str | Path,
    annotation_root: str | Path,
    *,
    seed: int = 42,
    val_ratio: float = 0.1,
    planner_hz: int = 2,
    history_step_seconds: float = 0.5,
    train_offsets: int = 2,
    continue_drop_ratio: float = 0.25,
) -> BreakfastDoneDataset:
    dataset_root = Path(dataset_root)
    annotation_root = Path(annotation_root)
    fps, _ = _episode_lengths(dataset_root)
    if fps % planner_hz:
        raise ValueError("dataset FPS must be divisible by planner_hz")
    stride = fps // planner_hz
    history_step_float = fps * history_step_seconds
    history_step = round(history_step_float)
    if history_step <= 0 or abs(history_step - history_step_float) > 1e-9:
        raise ValueError("history step cannot be represented exactly in frames")
    if not 1 <= train_offsets <= stride:
        raise ValueError("train_offsets must be between one and the planner stride")

    episodes, task_prompts = _load_episodes(dataset_root, annotation_root)
    episode_splits, split_ids = _assign_splits([episode.index for episode in episodes], seed=seed, val_ratio=val_ratio)
    samples: list[DoneSample] = []
    for episode in episodes:
        split = episode_splits[episode.index]
        offset_count = train_offsets if split == "train" else 1
        offsets = [index * stride // offset_count for index in range(offset_count)]
        for offset in offsets:
            samples.extend(
                _sample_episode(
                    episode,
                    split,
                    offset,
                    task_prompts,
                    stride=stride,
                    history_step=history_step,
                )
            )
        if split == "train":
            samples.extend(
                _exact_boundary_samples(
                    episode,
                    task_prompts,
                    stride=stride,
                    history_step=history_step,
                )
            )
    samples = _drop_train_continues(samples, episodes, ratio=continue_drop_ratio, seed=seed)
    ordered_samples = tuple(sample for split in ("train", "val") for sample in samples if sample.split == split)
    return BreakfastDoneDataset(ordered_samples, episodes, task_prompts, split_ids)


def build_feature_plan(samples: Sequence[DoneSample]) -> FeaturePlan:
    keys = sorted(
        {
            FeatureKey(sample.episode_index, frame_index, sample.prompt)
            for sample in samples
            for frame_index in sample.history_frames
        }
    )
    key_indices = {key: index for index, key in enumerate(keys)}
    history_indices = np.asarray(
        [
            [
                key_indices[FeatureKey(sample.episode_index, frame_index, sample.prompt)]
                for frame_index in sample.history_frames
            ]
            for sample in samples
        ],
        dtype=np.int32,
    )
    return FeaturePlan(keys=tuple(keys), history_indices=history_indices)
