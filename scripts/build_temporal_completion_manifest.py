"""Audit breakfast datasets and seal the temporal-completion split manifest.

The script is intentionally read-only with respect to both LeRobot datasets.
It derives each subtask's task index from its parquet rows (never from the
episode number), verifies the corresponding prompt metadata, and accepts
full-trajectory identity only through an explicit JSON map.

Identity-map schema::

    {
      "schema_version": 3,
      "matches": [{
        "group_id": 0,
        "full_episode_id": 17,
        "subtask_episode_ids": [0, 1, 2, 3]
      }],
      "ambiguities": [{
        "group_ids": [1, 2],
        "full_episode_ids": [18, 19],
        "candidates": [
          {"group_id": 1, "full_episode_id": 18},
          {"group_id": 1, "full_episode_id": 19},
          {"group_id": 2, "full_episode_id": 18}
        ],
        "reason": "alignment scores do not separate the candidates"
      }]
    }

Omitted subtask groups and full episodes are retained as quarantined manifest
records by :mod:`openpi.training.temporal_completion_data`.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import json
import os
import pathlib
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from openpi.training import temporal_completion_data as temporal_data

IDENTITY_MAP_SCHEMA_VERSION = 3
AUDIT_SUMMARY_SCHEMA_VERSION = 1
REQUIRED_METADATA_NAMES = ("info.json", "episodes.jsonl", "tasks.jsonl")


@dataclasses.dataclass(frozen=True)
class DatasetAudit:
    root: pathlib.Path
    repo_id: str
    metadata_files: tuple[pathlib.Path, ...]
    episode_count: int
    prompts_by_task: Mapping[int, str]
    subtask_episodes: tuple[temporal_data.SubtaskEpisodeRecord, ...] = ()
    full_episodes: tuple[temporal_data.FullEpisodeRecord, ...] = ()


@dataclasses.dataclass(frozen=True)
class LoadedIdentityMap:
    matches: tuple[temporal_data.IdentityMatchRecord, ...]
    ambiguities: tuple[temporal_data.IdentityAmbiguityRecord, ...]


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields do not match schema; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def metadata_files(dataset_root: str | os.PathLike[str]) -> tuple[pathlib.Path, ...]:
    """Returns audited LeRobot metadata files in deterministic path order."""

    root = pathlib.Path(dataset_root).resolve()
    meta_dir = root / "meta"
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"missing LeRobot metadata directory: {meta_dir}")
    for name in REQUIRED_METADATA_NAMES:
        path = meta_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing required LeRobot metadata file: {path}")
    files = tuple(
        sorted(
            (path.resolve() for path in meta_dir.rglob("*") if path.is_file()),
            key=lambda path: path.as_posix(),
        )
    )
    if not files:
        raise ValueError(f"LeRobot metadata directory contains no regular files: {meta_dir}")
    return files


def _read_json(path: pathlib.Path, *, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {context}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain one JSON object: {path}")
    return value


def _read_jsonl(path: pathlib.Path, *, context: str) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read {context}: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{context} row must be an object at {path}:{line_number}")
        records.append(value)
    if not records:
        raise ValueError(f"{context} contains no records: {path}")
    return tuple(records)


def _integer(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an integer, got boolean {value!r}")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{context} must be an integer, got {value!r}") from error
    if value != integer:
        raise ValueError(f"{context} must be an integer, got {value!r}")
    return integer


def _scalar_values(column: pa.ChunkedArray, *, context: str) -> tuple[Any, ...]:
    values: list[Any] = []
    for row_index, raw_value in enumerate(column.combine_chunks().to_pylist()):
        scalar = raw_value
        if isinstance(scalar, list | tuple):
            if len(scalar) != 1:
                raise ValueError(f"{context} row {row_index} is not scalar: {scalar!r}")
            scalar = scalar[0]
        values.append(scalar)
    return tuple(values)


def _load_prompts(meta_dir: pathlib.Path) -> dict[int, str]:
    prompts: dict[int, str] = {}
    for row_number, record in enumerate(_read_jsonl(meta_dir / "tasks.jsonl", context="LeRobot tasks"), start=1):
        if "task_index" not in record or "task" not in record:
            raise ValueError(f"tasks.jsonl row {row_number} must contain task_index and task")
        task_index = _integer(record["task_index"], context=f"tasks.jsonl row {row_number} task_index")
        prompt = str(record["task"])
        if task_index < 0 or not prompt.strip():
            raise ValueError(f"tasks.jsonl row {row_number} has invalid task index or empty prompt")
        if task_index in prompts:
            raise ValueError(f"tasks.jsonl contains duplicate task_index {task_index}")
        prompts[task_index] = prompt
    return prompts


def _load_episode_metadata(meta_dir: pathlib.Path) -> dict[int, Mapping[str, Any]]:
    episodes: dict[int, Mapping[str, Any]] = {}
    for row_number, record in enumerate(_read_jsonl(meta_dir / "episodes.jsonl", context="LeRobot episodes"), start=1):
        if "episode_index" not in record or "length" not in record:
            raise ValueError(f"episodes.jsonl row {row_number} must contain episode_index and length")
        episode_id = _integer(record["episode_index"], context=f"episodes.jsonl row {row_number} episode_index")
        length = _integer(record["length"], context=f"episode {episode_id} metadata length")
        if episode_id < 0 or length <= 0:
            raise ValueError(f"episode {episode_id} has invalid metadata length {length}")
        if episode_id in episodes:
            raise ValueError(f"episodes.jsonl contains duplicate episode_index {episode_id}")
        episodes[episode_id] = record
    return episodes


def _load_dataset_info(meta_dir: pathlib.Path, *, episode_count: int) -> tuple[str, int]:
    info = _read_json(meta_dir / "info.json", context="LeRobot info")
    for field in ("fps", "chunks_size", "data_path"):
        if field not in info:
            raise ValueError(f"LeRobot info.json is missing required field {field!r}")
    fps = float(info["fps"])
    if fps != temporal_data.FPS:
        raise ValueError(f"temporal completion requires exactly {temporal_data.FPS} fps, got {fps}")
    chunks_size = _integer(info["chunks_size"], context="info.json chunks_size")
    if chunks_size <= 0:
        raise ValueError(f"info.json chunks_size must be positive, got {chunks_size}")
    data_path = str(info["data_path"])
    if not data_path.strip():
        raise ValueError("info.json data_path must not be empty")
    if "total_episodes" in info:
        total_episodes = _integer(info["total_episodes"], context="info.json total_episodes")
        if total_episodes != episode_count:
            raise ValueError(f"info.json total_episodes={total_episodes} disagrees with episodes.jsonl={episode_count}")
    return data_path, chunks_size


def _episode_parquet_path(
    root: pathlib.Path,
    data_path: str,
    chunks_size: int,
    episode_id: int,
) -> pathlib.Path:
    try:
        relative = pathlib.Path(data_path.format(episode_chunk=episode_id // chunks_size, episode_index=episode_id))
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError(f"cannot format info.json data_path for episode {episode_id}: {data_path!r}") from error
    if relative.is_absolute():
        raise ValueError(f"info.json data_path must be relative, got {data_path!r}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"info.json data_path escapes dataset root for episode {episode_id}: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"missing episode parquet for episode {episode_id}: {resolved}")
    return resolved


def _validate_episode_tasks_metadata(
    record: Mapping[str, Any],
    *,
    episode_id: int,
    parquet_task_indices: Sequence[int],
    prompts: Mapping[int, str],
) -> None:
    missing = sorted(set(parquet_task_indices) - set(prompts))
    if missing:
        raise ValueError(f"episode {episode_id} parquet uses task indices absent from tasks.jsonl: {missing}")
    if "tasks" not in record:
        raise ValueError(f"episode {episode_id} metadata must contain prompt list field 'tasks'")
    metadata_prompts = record["tasks"]
    if not isinstance(metadata_prompts, list) or not metadata_prompts:
        raise ValueError(f"episode {episode_id} metadata tasks must be a non-empty list")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in metadata_prompts):
        raise ValueError(f"episode {episode_id} metadata contains an invalid prompt")
    expected_prompts = {prompts[task_index] for task_index in parquet_task_indices}
    if set(metadata_prompts) != expected_prompts:
        raise ValueError(
            f"episode {episode_id} prompt metadata disagrees with parquet task indices: "
            f"metadata={sorted(set(metadata_prompts))}, parquet={sorted(expected_prompts)}"
        )


def _audit_episode_parquet(
    parquet_path: pathlib.Path,
    *,
    episode_id: int,
    expected_length: int,
    metadata_record: Mapping[str, Any],
    prompts: Mapping[int, str],
    require_single_task: bool,
) -> tuple[int, ...]:
    schema = pq.read_schema(parquet_path)
    required_columns = {"episode_index", "frame_index", "task_index"}
    missing = sorted(required_columns - set(schema.names))
    if missing:
        raise ValueError(f"episode {episode_id} parquet is missing required columns: {missing}")
    table = pq.read_table(parquet_path, columns=sorted(required_columns))
    if table.num_rows != expected_length:
        raise ValueError(
            f"episode {episode_id} parquet rows={table.num_rows} disagree with metadata length={expected_length}"
        )
    episode_values = tuple(
        _integer(value, context=f"episode {episode_id} parquet episode_index")
        for value in _scalar_values(table["episode_index"], context=f"episode {episode_id} episode_index")
    )
    if any(value != episode_id for value in episode_values):
        raise ValueError(f"episode {episode_id} parquet contains a mismatched episode_index")
    frame_values = tuple(
        _integer(value, context=f"episode {episode_id} parquet frame_index")
        for value in _scalar_values(table["frame_index"], context=f"episode {episode_id} frame_index")
    )
    if frame_values != tuple(range(expected_length)):
        raise ValueError(f"episode {episode_id} parquet frame_index must be exactly 0..{expected_length - 1}")
    task_values = tuple(
        _integer(value, context=f"episode {episode_id} parquet task_index")
        for value in _scalar_values(table["task_index"], context=f"episode {episode_id} task_index")
    )
    unique_tasks = tuple(sorted(set(task_values)))
    if require_single_task and len(unique_tasks) != 1:
        raise ValueError(
            f"subtask episode {episode_id} must contain exactly one parquet task_index, got {unique_tasks}"
        )
    _validate_episode_tasks_metadata(
        metadata_record,
        episode_id=episode_id,
        parquet_task_indices=unique_tasks,
        prompts=prompts,
    )
    return unique_tasks


def audit_lerobot_dataset(
    dataset_root: str | os.PathLike[str],
    *,
    repo_id: str,
    subtask: bool,
) -> DatasetAudit:
    """Performs a fail-closed, read-only metadata and parquet audit."""

    root = pathlib.Path(dataset_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    if not repo_id.strip():
        raise ValueError("source repo_id must not be empty")
    files = metadata_files(root)
    meta_dir = root / "meta"
    prompts = _load_prompts(meta_dir)
    episode_metadata = _load_episode_metadata(meta_dir)
    data_path, chunks_size = _load_dataset_info(meta_dir, episode_count=len(episode_metadata))

    subtask_records: list[temporal_data.SubtaskEpisodeRecord] = []
    full_records: list[temporal_data.FullEpisodeRecord] = []
    episode_prompt: dict[int, str] = {}
    for episode_id in sorted(episode_metadata):
        record = episode_metadata[episode_id]
        length = _integer(record["length"], context=f"episode {episode_id} metadata length")
        parquet_path = _episode_parquet_path(root, data_path, chunks_size, episode_id)
        task_indices = _audit_episode_parquet(
            parquet_path,
            episode_id=episode_id,
            expected_length=length,
            metadata_record=record,
            prompts=prompts,
            require_single_task=subtask,
        )
        if subtask:
            task_index = task_indices[0]
            subtask_records.append(
                temporal_data.SubtaskEpisodeRecord(
                    episode_id=episode_id,
                    task_index=task_index,
                    length=length,
                )
            )
            episode_prompt[episode_id] = prompts[task_index]
        else:
            full_records.append(temporal_data.FullEpisodeRecord(episode_id=episode_id, length=length))

    if subtask:
        groups = temporal_data.build_subtask_groups(subtask_records)
        for group in groups:
            group_prompts = tuple(episode_prompt[episode_id] for episode_id in group.source_episode_ids)
            if group.task_indices != (0, 1, 2, 3):
                # SubtaskGroupRecord currently enforces this too.  Keep the
                # explicit check here so the CLI error remains an audit error.
                raise ValueError(
                    f"subtask group {group.group_id} parquet task order must be (0, 1, 2, 3), got {group.task_indices}"
                )
            if len(set(group_prompts)) != temporal_data.TASKS_PER_TRAJECTORY:
                raise ValueError(
                    f"subtask group {group.group_id} must have four distinct ordered prompts, got {group_prompts}"
                )

    return DatasetAudit(
        root=root,
        repo_id=repo_id,
        metadata_files=files,
        episode_count=len(episode_metadata),
        prompts_by_task=dict(sorted(prompts.items())),
        subtask_episodes=tuple(subtask_records),
        full_episodes=tuple(full_records),
    )


def load_identity_matches(
    path: str | os.PathLike[str],
    *,
    groups: Sequence[temporal_data.SubtaskGroupRecord],
    full_episodes: Sequence[temporal_data.FullEpisodeRecord],
) -> LoadedIdentityMap:
    """Loads and validates an explicit trajectory identity map."""

    identity_path = pathlib.Path(path).resolve()
    document = _read_json(identity_path, context="trajectory identity map")
    _require_exact_keys(
        document,
        {
            "schema_version",
            "matches",
            "ambiguities",
        },
        context="identity map",
    )
    if _integer(document["schema_version"], context="identity map schema_version") != IDENTITY_MAP_SCHEMA_VERSION:
        raise ValueError(f"identity map schema_version must be {IDENTITY_MAP_SCHEMA_VERSION}")
    raw_matches = document["matches"]
    if not isinstance(raw_matches, list) or not raw_matches:
        raise ValueError("identity map matches must be a non-empty list")
    raw_ambiguities = document["ambiguities"]
    if not isinstance(raw_ambiguities, list):
        raise ValueError("identity map ambiguities must be a list")

    group_by_id = {group.group_id: group for group in groups}
    full_ids = {episode.episode_id for episode in full_episodes}
    seen_groups: set[int] = set()
    seen_full: set[int] = set()
    matches: list[temporal_data.IdentityMatchRecord] = []
    for row_number, raw_match in enumerate(raw_matches, start=1):
        if not isinstance(raw_match, dict):
            raise ValueError(f"identity map match {row_number} must be an object")
        _require_exact_keys(
            raw_match,
            {"group_id", "full_episode_id", "subtask_episode_ids"},
            context=f"identity map match {row_number}",
        )
        group_id = _integer(raw_match["group_id"], context=f"identity map match {row_number} group_id")
        full_episode_id = _integer(
            raw_match["full_episode_id"], context=f"identity map match {row_number} full_episode_id"
        )
        if group_id not in group_by_id:
            raise ValueError(f"identity map match {row_number} refers to unknown subtask group {group_id}")
        if full_episode_id not in full_ids:
            raise ValueError(f"identity map match {row_number} refers to unknown full episode {full_episode_id}")
        if group_id in seen_groups:
            raise ValueError(f"identity map is not bijective: duplicate subtask group {group_id}")
        if full_episode_id in seen_full:
            raise ValueError(f"identity map is not bijective: duplicate full episode {full_episode_id}")
        seen_groups.add(group_id)
        seen_full.add(full_episode_id)

        raw_source_ids = raw_match["subtask_episode_ids"]
        if not isinstance(raw_source_ids, list):
            raise ValueError(f"identity map match {row_number} subtask_episode_ids must be a list")
        source_ids = tuple(
            _integer(value, context=f"identity map match {row_number} subtask_episode_ids") for value in raw_source_ids
        )
        expected_source_ids = group_by_id[group_id].source_episode_ids
        if source_ids != expected_source_ids:
            raise ValueError(
                f"identity map match {row_number} source episodes disagree with audited group {group_id}: "
                f"map={source_ids}, audited={expected_source_ids}"
            )

        matches.append(
            temporal_data.IdentityMatchRecord(
                group_id=group_id,
                full_episode_id=full_episode_id,
            )
        )

    ambiguities: list[temporal_data.IdentityAmbiguityRecord] = []
    for row_number, raw_ambiguity in enumerate(raw_ambiguities, start=1):
        if not isinstance(raw_ambiguity, dict):
            raise ValueError(f"identity map ambiguity {row_number} must be an object")
        _require_exact_keys(
            raw_ambiguity,
            {"group_ids", "full_episode_ids", "candidates", "reason"},
            context=f"identity map ambiguity {row_number}",
        )
        raw_group_ids = raw_ambiguity["group_ids"]
        raw_full_episode_ids = raw_ambiguity["full_episode_ids"]
        raw_candidates = raw_ambiguity["candidates"]
        if not isinstance(raw_group_ids, list) or not raw_group_ids:
            raise ValueError(f"identity map ambiguity {row_number} group_ids must be a non-empty list")
        if not isinstance(raw_full_episode_ids, list) or not raw_full_episode_ids:
            raise ValueError(f"identity map ambiguity {row_number} full_episode_ids must be a non-empty list")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(f"identity map ambiguity {row_number} candidates must be a non-empty list")
        group_ids = tuple(
            sorted(_integer(value, context=f"identity map ambiguity {row_number} group_ids") for value in raw_group_ids)
        )
        full_episode_ids = tuple(
            sorted(
                _integer(value, context=f"identity map ambiguity {row_number} full_episode_ids")
                for value in raw_full_episode_ids
            )
        )
        if len(set(group_ids)) != len(group_ids):
            raise ValueError(f"identity map ambiguity {row_number} repeats a subtask group")
        if len(set(full_episode_ids)) != len(full_episode_ids):
            raise ValueError(f"identity map ambiguity {row_number} repeats a full episode")
        unknown_groups = set(group_ids) - set(group_by_id)
        if unknown_groups:
            raise ValueError(
                f"identity map ambiguity {row_number} refers to unknown subtask groups {sorted(unknown_groups)}"
            )
        unknown_full = set(full_episode_ids) - full_ids
        if unknown_full:
            raise ValueError(
                f"identity map ambiguity {row_number} refers to unknown full episodes {sorted(unknown_full)}"
            )
        matched_group_overlap = set(group_ids) & seen_groups
        if matched_group_overlap:
            raise ValueError(
                f"identity map ambiguity {row_number} overlaps matched or prior-ambiguity subtask groups "
                f"{sorted(matched_group_overlap)}"
            )
        matched_full_overlap = set(full_episode_ids) & seen_full
        if matched_full_overlap:
            raise ValueError(
                f"identity map ambiguity {row_number} overlaps matched or prior-ambiguity full episodes "
                f"{sorted(matched_full_overlap)}"
            )

        candidates: list[tuple[int, int]] = []
        for candidate_number, raw_candidate in enumerate(raw_candidates, start=1):
            context = f"identity map ambiguity {row_number} candidate {candidate_number}"
            if not isinstance(raw_candidate, dict):
                raise ValueError(f"{context} must be an object")
            _require_exact_keys(raw_candidate, {"group_id", "full_episode_id"}, context=context)
            candidates.append(
                (
                    _integer(raw_candidate["group_id"], context=f"{context} group_id"),
                    _integer(raw_candidate["full_episode_id"], context=f"{context} full_episode_id"),
                )
            )
        candidate_pairs = tuple(sorted(candidates))
        reason = raw_ambiguity["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"identity map ambiguity {row_number} reason must be non-empty")
        ambiguity = temporal_data.IdentityAmbiguityRecord(
            group_ids=group_ids,
            full_episode_ids=full_episode_ids,
            candidate_pairs=candidate_pairs,
            reason=reason.strip(),
        )
        ambiguities.append(ambiguity)
        seen_groups.update(group_ids)
        seen_full.update(full_episode_ids)

    return LoadedIdentityMap(matches=tuple(matches), ambiguities=tuple(ambiguities))


def _require_output_outside_sources(
    path: str | os.PathLike[str],
    *,
    source_roots: Sequence[pathlib.Path],
    context: str,
) -> pathlib.Path:
    output = pathlib.Path(path).resolve()
    for root in source_roots:
        if output == root or output.is_relative_to(root):
            raise ValueError(f"{context} must not be written inside source dataset {root}: {output}")
    return output


def _write_json_once(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == encoded:
            return
        raise FileExistsError(f"refusing to rewrite sealed audit summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def build_manifest(
    *,
    subtask_root: str | os.PathLike[str],
    subtask_repo_id: str,
    full_root: str | os.PathLike[str],
    full_repo_id: str,
    identity_map_path: str | os.PathLike[str],
) -> tuple[temporal_data.TemporalCompletionManifest, Mapping[str, Any]]:
    """Audits both sources and constructs a sealed in-memory manifest."""

    subtask_audit = audit_lerobot_dataset(subtask_root, repo_id=subtask_repo_id, subtask=True)
    full_audit = audit_lerobot_dataset(full_root, repo_id=full_repo_id, subtask=False)
    groups = temporal_data.build_subtask_groups(subtask_audit.subtask_episodes)
    identity_map = load_identity_matches(
        identity_map_path,
        groups=groups,
        full_episodes=full_audit.full_episodes,
    )
    identities = temporal_data.build_trajectory_identities(
        groups,
        full_audit.full_episodes,
        identity_map.matches,
        identity_map.ambiguities,
    )
    manifest = temporal_data.create_temporal_manifest(
        identities,
        source_subtask_repo_id=subtask_repo_id,
        source_subtask_root=subtask_audit.root,
        source_full_repo_id=full_repo_id,
        source_full_root=full_audit.root,
        task_prompts=tuple(subtask_audit.prompts_by_task[index] for index in range(4)),
    )
    mapping_counts: dict[str, int] = {}
    exclusion_counts: dict[str, int] = {}
    for trajectory in manifest.trajectories:
        mapping_counts[trajectory.mapping_status] = mapping_counts.get(trajectory.mapping_status, 0) + 1
        if trajectory.exclusion_reason is not None:
            exclusion_counts[trajectory.exclusion_reason] = exclusion_counts.get(trajectory.exclusion_reason, 0) + 1
    summary: Mapping[str, Any] = {
        "schema_version": AUDIT_SUMMARY_SCHEMA_VERSION,
        "subtask": {
            "root": str(subtask_audit.root),
            "repo_id": subtask_repo_id,
            "episode_count": subtask_audit.episode_count,
            "group_count": len(groups),
            "metadata_files": [str(path) for path in subtask_audit.metadata_files],
            "ordered_prompts": [subtask_audit.prompts_by_task[index] for index in range(4)],
        },
        "full": {
            "root": str(full_audit.root),
            "repo_id": full_repo_id,
            "episode_count": full_audit.episode_count,
            "metadata_files": [str(path) for path in full_audit.metadata_files],
        },
        "identity": {
            "match_count": len(identity_map.matches),
            "ambiguity_component_count": len(identity_map.ambiguities),
            "mapping_status_counts": dict(sorted(mapping_counts.items())),
            "exclusion_reason_counts": dict(sorted(exclusion_counts.items())),
        },
        "split_counts": manifest.split_counts.to_dict(),
    }
    return manifest, summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtask-root", type=pathlib.Path, required=True)
    parser.add_argument("--subtask-repo-id", required=True)
    parser.add_argument("--full-root", type=pathlib.Path, required=True)
    parser.add_argument("--full-repo-id", required=True)
    parser.add_argument("--identity-map", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--audit-summary",
        type=pathlib.Path,
        help="Optional immutable JSON audit summary (also printed to stdout).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    source_roots = (args.subtask_root.resolve(), args.full_root.resolve())
    output = _require_output_outside_sources(args.output, source_roots=source_roots, context="manifest output")
    audit_summary_path = None
    if args.audit_summary is not None:
        audit_summary_path = _require_output_outside_sources(
            args.audit_summary,
            source_roots=source_roots,
            context="audit summary",
        )
        if audit_summary_path == output:
            raise ValueError("manifest output and audit summary must be different paths")

    manifest, summary = build_manifest(
        subtask_root=args.subtask_root,
        subtask_repo_id=args.subtask_repo_id,
        full_root=args.full_root,
        full_repo_id=args.full_repo_id,
        identity_map_path=args.identity_map,
    )
    temporal_data.save_temporal_manifest(output, manifest)
    if audit_summary_path is not None:
        _write_json_once(audit_summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
