from __future__ import annotations

import hashlib
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
    evidence_directory = subtask_root.parent / "evidence"
    evidence_directory.mkdir(exist_ok=True)
    artifacts: dict[tuple[int, int], dict[str, str]] = {}
    for group_id, full_episode_id in matches:
        relative_path = pathlib.Path("evidence") / f"match-{group_id}-{full_episode_id}.json"
        payload = f'{{"group_id":{group_id},"full_episode_id":{full_episode_id}}}\n'.encode()
        (subtask_root.parent / relative_path).write_bytes(payload)
        artifacts[(group_id, full_episode_id)] = {
            "path": relative_path.as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {
        "schema_version": 2,
        "source_subtask_metadata_fingerprint": build_manifest.source_metadata_fingerprint(subtask_root),
        "source_full_metadata_fingerprint": build_manifest.source_metadata_fingerprint(full_root),
        "matches": [
            {
                "group_id": group_id,
                "full_episode_id": full_episode_id,
                "subtask_episode_ids": list(range(group_id * 4, group_id * 4 + 4)),
                "evidence": {
                    "method": "synthetic_state_action_alignment",
                    "artifacts": [artifacts[(group_id, full_episode_id)]],
                },
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
    name = f"ambiguity-{'-'.join(map(str, group_ids))}-{'-'.join(map(str, full_episode_ids))}.json"
    relative_path = pathlib.Path("evidence") / name
    payload = json.dumps({"candidates": candidates}, sort_keys=True).encode()
    artifact_path = identity_directory / relative_path
    artifact_path.parent.mkdir(exist_ok=True)
    artifact_path.write_bytes(payload)
    ambiguity = {
        "group_ids": group_ids,
        "full_episode_ids": full_episode_ids,
        "candidates": [
            {"group_id": group_id, "full_episode_id": full_episode_id} for group_id, full_episode_id in candidates
        ],
        "reason": reason,
        "evidence": {
            "method": "synthetic_tie_audit",
            "artifacts": [{"path": relative_path.as_posix(), "sha256": hashlib.sha256(payload).hexdigest()}],
        },
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
    # evidence map and must never infer equality/order from episode numbers.
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
    assert manifest.subtask_metadata_fingerprint == build_manifest.source_metadata_fingerprint(subtask_root)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["metadata_fingerprint_convention"] == (
        "all_regular_files_recursively_below_meta_with_resolved_paths"
    )
    assert summary["identity"]["verified_match_count"] == 6
    assert summary["identity"]["mapping_status_counts"] == {"matched": 6}
    assert json.loads(capsys.readouterr().out) == summary


def test_metadata_fingerprint_includes_every_nested_regular_meta_file(tmp_path) -> None:
    root = tmp_path / "dataset"
    _make_dataset(root, [0], length=1, task_by_episode={0: 0})
    before_files = build_manifest.metadata_files(root)
    before_fingerprint = build_manifest.source_metadata_fingerprint(root)
    extra = root / "meta" / "nested" / "alignment.json"
    extra.parent.mkdir()
    extra.write_text('{"version": 1}\n', encoding="utf-8")

    after_files = build_manifest.metadata_files(root)
    assert after_files == tuple(sorted((*before_files, extra.resolve()), key=lambda path: path.as_posix()))
    assert build_manifest.source_metadata_fingerprint(root) != before_fingerprint


def test_subtask_order_is_derived_from_parquet_not_episode_id(tmp_path) -> None:
    root = tmp_path / "subtasks"
    task_by_episode = {0: 0, 1: 1, 2: 1, 3: 3}
    _make_dataset(root, list(range(4)), length=45, task_by_episode=task_by_episode)

    with pytest.raises(ValueError, match="task order must be \\(0, 1, 2, 3\\)"):
        build_manifest.audit_lerobot_dataset(root, repo_id="test/subtasks", subtask=True)


def test_identity_map_is_source_locked_evidence_backed_and_bijective(tmp_path) -> None:
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
            subtask_metadata_fingerprint=subtask_audit.metadata_fingerprint,
            full_metadata_fingerprint=full_audit.metadata_fingerprint,
        )

    document = _identity_document(subtask_root, full_root, [(0, 20)])
    document["source_full_metadata_fingerprint"] = "0" * 64
    identity_path = tmp_path / "stale.json"
    identity_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="full metadata fingerprint is stale"):
        build_manifest.load_identity_matches(
            identity_path,
            groups=groups,
            full_episodes=full_audit.full_episodes,
            subtask_metadata_fingerprint=subtask_audit.metadata_fingerprint,
            full_metadata_fingerprint=full_audit.metadata_fingerprint,
        )

    document = _identity_document(subtask_root, full_root, [(0, 20)])
    document["matches"][0]["evidence"]["artifacts"] = []  # type: ignore[index]
    identity_path = tmp_path / "no-evidence.json"
    identity_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="artifacts must be a non-empty list"):
        build_manifest.load_identity_matches(
            identity_path,
            groups=groups,
            full_episodes=full_audit.full_episodes,
            subtask_metadata_fingerprint=subtask_audit.metadata_fingerprint,
            full_metadata_fingerprint=full_audit.metadata_fingerprint,
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


def test_identity_v2_recomputes_relative_artifact_hash_and_rejects_missing_or_tampered_file(
    tmp_path, monkeypatch
) -> None:
    subtask_root = tmp_path / "subtasks"
    full_root = tmp_path / "full"
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
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(document), encoding="utf-8")

    unrelated_cwd = tmp_path / "cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    loaded = build_manifest.load_identity_matches(
        identity_path,
        groups=groups,
        full_episodes=full_audit.full_episodes,
        subtask_metadata_fingerprint=subtask_audit.metadata_fingerprint,
        full_metadata_fingerprint=full_audit.metadata_fingerprint,
    )
    assert len(loaded.matches) == 1
    assert loaded.ambiguities == ()

    artifact = tmp_path / "evidence" / "match-0-20.json"
    artifact.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_manifest.load_identity_matches(
            identity_path,
            groups=groups,
            full_episodes=full_audit.full_episodes,
            subtask_metadata_fingerprint=subtask_audit.metadata_fingerprint,
            full_metadata_fingerprint=full_audit.metadata_fingerprint,
        )

    artifact.unlink()
    with pytest.raises(FileNotFoundError, match="file is missing or not regular"):
        build_manifest.load_identity_matches(
            identity_path,
            groups=groups,
            full_episodes=full_audit.full_episodes,
            subtask_metadata_fingerprint=subtask_audit.metadata_fingerprint,
            full_metadata_fingerprint=full_audit.metadata_fingerprint,
        )


def test_identity_v2_rejects_legacy_schema_and_artifact_path_escape(tmp_path) -> None:
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
        "subtask_metadata_fingerprint": subtask_audit.metadata_fingerprint,
        "full_metadata_fingerprint": full_audit.metadata_fingerprint,
    }
    with pytest.raises(ValueError, match="schema_version must be 2"):
        build_manifest.load_identity_matches(identity_path, **kwargs)

    document["schema_version"] = 2
    outside = tmp_path / "outside.json"
    outside.write_text("evidence", encoding="utf-8")
    document["matches"][0]["evidence"]["artifacts"] = [  # type: ignore[index]
        {"path": "../outside.json", "sha256": hashlib.sha256(b"evidence").hexdigest()}
    ]
    identity_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes the identity-map directory"):
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
    assert len({record.evidence_fingerprint for record in ambiguous}) == 1
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
        "subtask_metadata_fingerprint": subtask_audit.metadata_fingerprint,
        "full_metadata_fingerprint": full_audit.metadata_fingerprint,
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
