from __future__ import annotations

import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.training import temporal_completion_data as temporal_data
from scripts import build_temporal_completion_manifest as build_manifest

DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
PROMPTS = {index: f"breakfast subtask {index}" for index in range(4)}


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_dataset(
    root: pathlib.Path,
    episode_ids: list[int],
    *,
    length: int,
    task_by_episode: dict[int, int],
) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "chunks_size": 1000,
                "data_path": DATA_PATH,
                "total_episodes": len(episode_ids),
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        meta / "tasks.jsonl",
        [{"task_index": task_index, "task": prompt} for task_index, prompt in PROMPTS.items()],
    )
    _write_jsonl(
        meta / "episodes.jsonl",
        [
            {
                "episode_index": episode_id,
                "length": length,
                "tasks": [PROMPTS[task_by_episode[episode_id]]],
            }
            for episode_id in episode_ids
        ],
    )
    for episode_id in episode_ids:
        path = root / DATA_PATH.format(episode_chunk=episode_id // 1000, episode_index=episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "episode_index": [episode_id] * length,
                    "frame_index": list(range(length)),
                    "task_index": [task_by_episode[episode_id]] * length,
                }
            ),
            path,
        )


def _identity_document(
    subtask_root: pathlib.Path,
    full_root: pathlib.Path,
    matches: list[tuple[int, int]],
) -> dict[str, object]:
    del subtask_root, full_root
    return {
        "schema_version": 3,
        "matches": [
            {
                "group_id": group_id,
                "full_episode_id": full_episode_id,
                "subtask_episode_ids": list(range(group_id * 4, group_id * 4 + 4)),
            }
            for group_id, full_episode_id in matches
        ],
        "ambiguities": [],
    }


def _append_ambiguity(
    document: dict[str, object],
    identity_directory: pathlib.Path,
    *,
    group_ids: list[int],
    full_episode_ids: list[int],
    candidates: list[tuple[int, int]],
    reason: str = "synthetic candidates are tied",
) -> None:
    del identity_directory
    ambiguity = {
        "group_ids": group_ids,
        "full_episode_ids": full_episode_ids,
        "candidates": [
            {"group_id": group_id, "full_episode_id": full_episode_id} for group_id, full_episode_id in candidates
        ],
        "reason": reason,
    }
    ambiguities = document["ambiguities"]
    assert isinstance(ambiguities, list)
    ambiguities.append(ambiguity)


def test_cli_audits_parquet_identity_and_seals_trajectory_split(tmp_path, capsys) -> None:
    subtask_root = tmp_path / "subtasks"
    full_root = tmp_path / "full"
    subtask_ids = list(range(24))
    full_ids = list(range(100, 106))
    _make_dataset(
        subtask_root,
        subtask_ids,
        length=45,
        task_by_episode={episode_id: episode_id % 4 for episode_id in subtask_ids},
    )
    _make_dataset(
        full_root,
        full_ids,
        length=180,
        task_by_episode={episode_id: episode_id % 4 for episode_id in full_ids},
    )
    # Deliberately reverse identity: the implementation must consume this
    # identity map and must never infer equality/order from episode numbers.
    matches = list(zip(range(6), reversed(full_ids), strict=True))
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(_identity_document(subtask_root, full_root, matches)), encoding="utf-8")
    output_path = tmp_path / "output" / "manifest.json"
    summary_path = tmp_path / "output" / "audit.json"

    build_manifest.main(
        [
            "--subtask-root",
            str(subtask_root),
            "--subtask-repo-id",
            "test/subtasks",
            "--full-root",
            str(full_root),
            "--full-repo-id",
            "test/full",
            "--identity-map",
            str(identity_path),
            "--output",
            str(output_path),
            "--audit-summary",
            str(summary_path),
        ]
    )

    manifest = temporal_data.TemporalCompletionManifest.from_dict(json.loads(output_path.read_text(encoding="utf-8")))
    assert manifest.split_counts.to_dict() == {"train": 4, "val": 1, "test": 1}
    assert {record.mapping_status for record in manifest.trajectories} == {"matched"}
    assert {record.group_id: record.full_episode_id for record in manifest.trajectories} == dict(matches)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["identity"]["match_count"] == 6
    assert summary["identity"]["mapping_status_counts"] == {"matched": 6}
    assert json.loads(capsys.readouterr().out) == summary


def test_cli_builds_subtask_logical_split_without_identity_map(tmp_path, capsys) -> None:
    subtask_root = tmp_path / "subtasks"
    _make_dataset(
        subtask_root,
        list(range(24)),
        length=45,
        task_by_episode={episode_id: episode_id % 4 for episode_id in range(24)},
    )
    output_path = tmp_path / "output" / "manifest.json"
    summary_path = tmp_path / "output" / "audit.json"

    build_manifest.main(
        [
            "--subtask-root",
            str(subtask_root),
            "--subtask-repo-id",
            "test/subtasks",
            "--output",
            str(output_path),
            "--audit-summary",
            str(summary_path),
        ]
    )

    manifest = temporal_data.load_temporal_manifest(output_path)
    assert manifest.trajectory_source == "subtask_logical"
    assert manifest.source_full_repo_id is None
    assert manifest.split_counts.to_dict() == {"train": 4, "val": 1, "test": 1}
    assert {record.mapping_status for record in manifest.trajectories} == {"subtask_only"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["trajectory_source"] == "subtask_logical"
    assert summary["full"] is None
    assert summary["identity"]["required"] is False
    assert json.loads(capsys.readouterr().out) == summary


def test_manifest_builder_rejects_partial_full_identity_arguments(tmp_path) -> None:
    subtask_root = tmp_path / "subtasks"
    _make_dataset(
        subtask_root,
        list(range(4)),
        length=45,
        task_by_episode={episode_id: episode_id % 4 for episode_id in range(4)},
    )

    with pytest.raises(ValueError, match="must be supplied together"):
        build_manifest.build_manifest(
            subtask_root=subtask_root,
            subtask_repo_id="test/subtasks",
            full_root=tmp_path / "full",
        )


def test_metadata_listing_includes_every_nested_regular_meta_file(tmp_path) -> None:
    root = tmp_path / "dataset"
    _make_dataset(root, [0], length=1, task_by_episode={0: 0})
    before_files = build_manifest.metadata_files(root)
    extra = root / "meta" / "nested" / "alignment.json"
    extra.parent.mkdir()
    extra.write_text('{"version": 1}\n', encoding="utf-8")

    after_files = build_manifest.metadata_files(root)
    assert after_files == tuple(sorted((*before_files, extra.resolve()), key=lambda path: path.as_posix()))


def test_subtask_order_is_derived_from_parquet_not_episode_id(tmp_path) -> None:
    root = tmp_path / "subtasks"
    task_by_episode = {0: 0, 1: 1, 2: 1, 3: 3}
    _make_dataset(root, list(range(4)), length=45, task_by_episode=task_by_episode)

    with pytest.raises(ValueError, match="task order must be \\(0, 1, 2, 3\\)"):
        build_manifest.audit_lerobot_dataset(root, repo_id="test/subtasks", subtask=True)


def test_identity_map_is_explicit_and_bijective(tmp_path) -> None:
    subtask_root = tmp_path / "subtasks"
    full_root = tmp_path / "full"
    _make_dataset(
        subtask_root,
        list(range(8)),
        length=45,
        task_by_episode={episode_id: episode_id % 4 for episode_id in range(8)},
    )
    _make_dataset(full_root, [20, 21], length=180, task_by_episode={20: 0, 21: 1})
    subtask_audit = build_manifest.audit_lerobot_dataset(subtask_root, repo_id="test/subtasks", subtask=True)
    full_audit = build_manifest.audit_lerobot_dataset(full_root, repo_id="test/full", subtask=False)
    groups = temporal_data.build_subtask_groups(subtask_audit.subtask_episodes)

    document = _identity_document(subtask_root, full_root, [(0, 20), (1, 20)])
    identity_path = tmp_path / "duplicate.json"
    identity_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="not bijective: duplicate full episode 20"):
        build_manifest.load_identity_matches(
            identity_path,
            groups=groups,
            full_episodes=full_audit.full_episodes,
        )

    valid_document = _identity_document(subtask_root, full_root, [(0, 20)])
    identity_path = tmp_path / "partial-but-valid.json"
    identity_path.write_text(json.dumps(valid_document), encoding="utf-8")
    manifest, summary = build_manifest.build_manifest(
        subtask_root=subtask_root,
        subtask_repo_id="test/subtasks",
        full_root=full_root,
        full_repo_id="test/full",
        identity_map_path=identity_path,
    )
    assert manifest.split_counts.to_dict() == {"train": 1, "val": 0, "test": 0}
    assert {record.mapping_status for record in manifest.trajectories} == {
        "matched",
        "subtask_only",
        "full_only",
    }
    assert summary["identity"]["mapping_status_counts"] == {
        "full_only": 1,
        "matched": 1,
        "subtask_only": 1,
    }


def test_identity_v3_rejects_legacy_schema(tmp_path) -> None:
    subtask_root = tmp_path / "bundle" / "subtasks"
    full_root = tmp_path / "bundle" / "full"
    _make_dataset(
        subtask_root,
        list(range(4)),
        length=45,
        task_by_episode={episode_id: episode_id % 4 for episode_id in range(4)},
    )
    _make_dataset(full_root, [20], length=180, task_by_episode={20: 0})
    subtask_audit = build_manifest.audit_lerobot_dataset(subtask_root, repo_id="test/subtasks", subtask=True)
    full_audit = build_manifest.audit_lerobot_dataset(full_root, repo_id="test/full", subtask=False)
    groups = temporal_data.build_subtask_groups(subtask_audit.subtask_episodes)
    document = _identity_document(subtask_root, full_root, [(0, 20)])
    identity_path = subtask_root.parent / "identity.json"
    document["schema_version"] = 1
    identity_path.write_text(json.dumps(document), encoding="utf-8")
    kwargs = {
        "groups": groups,
        "full_episodes": full_audit.full_episodes,
    }
    with pytest.raises(ValueError, match="schema_version must be 3"):
        build_manifest.load_identity_matches(identity_path, **kwargs)


def test_ambiguity_is_explicitly_quarantined_on_both_identity_sides(tmp_path) -> None:
    subtask_root = tmp_path / "subtasks"
    full_root = tmp_path / "full"
    _make_dataset(
        subtask_root,
        list(range(12)),
        length=45,
        task_by_episode={episode_id: episode_id % 4 for episode_id in range(12)},
    )
    _make_dataset(full_root, [20, 21, 22], length=180, task_by_episode={20: 0, 21: 1, 22: 2})
    document = _identity_document(subtask_root, full_root, [(0, 20)])
    _append_ambiguity(
        document,
        tmp_path,
        group_ids=[1, 2],
        full_episode_ids=[21, 22],
        candidates=[(1, 21), (1, 22), (2, 21)],
        reason="two alignments remain tied",
    )
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(document), encoding="utf-8")

    manifest, summary = build_manifest.build_manifest(
        subtask_root=subtask_root,
        subtask_repo_id="test/subtasks",
        full_root=full_root,
        full_repo_id="test/full",
        identity_map_path=identity_path,
    )

    assert manifest.split_counts.to_dict() == {"train": 1, "val": 0, "test": 0}
    assert summary["identity"]["ambiguity_component_count"] == 1
    assert summary["identity"]["mapping_status_counts"] == {"ambiguous": 4, "matched": 1}
    ambiguous = [record for record in manifest.trajectories if record.mapping_status == "ambiguous"]
    assert len(ambiguous) == 4
    assert {record.group_id for record in ambiguous if record.group_id is not None} == {1, 2}
    assert {record.full_episode_id for record in ambiguous if record.full_episode_id is not None} == {21, 22}
    assert all((record.group_id is None) != (record.full_episode_id is None) for record in ambiguous)
    assert all(record.split is None for record in ambiguous)
    assert {record.exclusion_reason for record in ambiguous} == {"ambiguous_identity:two alignments remain tied"}


def test_ambiguities_cannot_overlap_matches_or_each_other(tmp_path) -> None:
    subtask_root = tmp_path / "subtasks"
    full_root = tmp_path / "full"
    _make_dataset(
        subtask_root,
        list(range(12)),
        length=45,
        task_by_episode={episode_id: episode_id % 4 for episode_id in range(12)},
    )
    _make_dataset(full_root, [20, 21, 22], length=180, task_by_episode={20: 0, 21: 1, 22: 2})
    subtask_audit = build_manifest.audit_lerobot_dataset(subtask_root, repo_id="test/subtasks", subtask=True)
    full_audit = build_manifest.audit_lerobot_dataset(full_root, repo_id="test/full", subtask=False)
    groups = temporal_data.build_subtask_groups(subtask_audit.subtask_episodes)
    kwargs = {
        "groups": groups,
        "full_episodes": full_audit.full_episodes,
    }

    document = _identity_document(subtask_root, full_root, [(0, 20)])
    _append_ambiguity(document, tmp_path, group_ids=[0], full_episode_ids=[21], candidates=[(0, 21)])
    identity_path = tmp_path / "matched-overlap.json"
    identity_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps matched or prior-ambiguity subtask groups"):
        build_manifest.load_identity_matches(identity_path, **kwargs)

    document = _identity_document(subtask_root, full_root, [(0, 20)])
    _append_ambiguity(document, tmp_path, group_ids=[1], full_episode_ids=[21], candidates=[(1, 21)])
    _append_ambiguity(document, tmp_path, group_ids=[2], full_episode_ids=[21], candidates=[(2, 21)])
    identity_path = tmp_path / "ambiguity-overlap.json"
    identity_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps matched or prior-ambiguity full episodes"):
        build_manifest.load_identity_matches(identity_path, **kwargs)
