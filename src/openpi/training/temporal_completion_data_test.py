from __future__ import annotations

import dataclasses

import pytest

from openpi.training import temporal_completion_data as temporal_data


def _episodes_for_groups(
    count: int, *, lengths: tuple[int, int, int, int] = (60, 60, 60, 60)
) -> tuple[temporal_data.SubtaskEpisodeRecord, ...]:
    return tuple(
        temporal_data.SubtaskEpisodeRecord(
            episode_id=group_id * 4 + task_index,
            task_index=task_index,
            length=lengths[task_index],
        )
        for group_id in range(count)
        for task_index in range(4)
    )


def _group(*, lengths: tuple[int, int, int, int] = (100, 100, 100, 100)):
    return temporal_data.build_subtask_groups(_episodes_for_groups(1, lengths=lengths))[0]


def _identities(
    count: int,
    *,
    extra_full: int = 0,
    lengths: tuple[int, int, int, int] = (60, 60, 60, 60),
):
    groups = temporal_data.build_subtask_groups(_episodes_for_groups(count, lengths=lengths))
    full_episodes = tuple(
        temporal_data.FullEpisodeRecord(episode_id=index, length=sum(lengths) + 7)
        for index in range(count + extra_full)
    )
    matches = tuple(
        temporal_data.IdentityMatchRecord(
            group_id=index,
            full_episode_id=index,
        )
        for index in range(count)
    )
    return temporal_data.build_trajectory_identities(groups, full_episodes, matches)


def _manifest(tmp_path, count: int = 10, *, extra_full: int = 0):
    return temporal_data.create_temporal_manifest(
        _identities(count, extra_full=extra_full),
        source_subtask_repo_id="org/breakfast-subtasks",
        source_subtask_root=tmp_path / "subtasks",
        source_full_repo_id="org/breakfast-full",
        source_full_root=tmp_path / "full",
        task_prompts=("task 0", "task 1", "task 2", "task 3"),
    )


def test_group_audit_builds_exact_four_stage_identity_and_integer_boundaries():
    group = _group(lengths=(15, 1, 14, 16))

    assert group.source_episode_ids == (0, 1, 2, 3)
    assert group.task_indices == (0, 1, 2, 3)
    assert group.boundaries == (15, 16, 30, 46)
    assert group.positive_ticks == (14, 0, 13, 15)


@pytest.mark.parametrize(
    ("episodes", "message"),
    [
        (_episodes_for_groups(1)[:-1], "cannot form complete"),
        (_episodes_for_groups(2)[1:], "must be exactly 0"),
        (
            (*_episodes_for_groups(1), _episodes_for_groups(1)[0]),
            "duplicate episode_id",
        ),
        (
            (
                temporal_data.SubtaskEpisodeRecord(0, 0, 60),
                temporal_data.SubtaskEpisodeRecord(1, 2, 60),
                temporal_data.SubtaskEpisodeRecord(2, 1, 60),
                temporal_data.SubtaskEpisodeRecord(3, 3, 60),
            ),
            "task order",
        ),
    ],
)
def test_group_audit_fails_closed_for_missing_duplicate_or_misordered_episodes(episodes, message):
    with pytest.raises(ValueError, match=message):
        temporal_data.build_subtask_groups(episodes)


def test_episode_records_reject_empty_or_invalid_tasks():
    with pytest.raises(ValueError, match="empty"):
        temporal_data.SubtaskEpisodeRecord(episode_id=0, task_index=0, length=0)
    with pytest.raises(ValueError, match="task_index"):
        temporal_data.SubtaskEpisodeRecord(episode_id=0, task_index=4, length=1)


@pytest.mark.parametrize(("boundary", "expected"), [(15, 15), (16, 30), (29, 30)])
def test_positive_tick_uses_integer_ceil_for_remainders_zero_one_and_fourteen(boundary, expected):
    assert temporal_data.ceil_to_tick(boundary) == expected


def test_prompt_history_reachability_rejects_44_frames_and_accepts_45():
    assert not temporal_data.positive_is_reachable(44, previous_switch_tick=0)
    assert temporal_data.positive_is_reachable(45, previous_switch_tick=0)
    assert not temporal_data.positive_is_reachable(29, previous_switch_tick=None)
    assert temporal_data.positive_is_reachable(30, previous_switch_tick=None)


def test_global_ticks_do_not_reset_phase_at_subtask_boundaries():
    group = _group(lengths=(17, 60, 60, 60))

    assert group.boundaries[0] == 17
    assert temporal_data.global_feature_ticks(group, 0) == (0, 15, 30)
    assert temporal_data.global_feature_ticks(group, 1)[0] == 45
    assert all(tick % 15 == 0 for task in range(4) for tick in temporal_data.global_feature_ticks(group, task))


def test_logical_source_mapping_crosses_subtasks_and_holds_task4_terminal_frame():
    group = _group(lengths=(100, 100, 100, 100))

    assert temporal_data.map_logical_frame(group, 99) == temporal_data.SourceFrameReference(
        episode_id=0, frame_index=99, terminal_hold=False
    )
    assert temporal_data.map_logical_frame(group, 100) == temporal_data.SourceFrameReference(
        episode_id=1, frame_index=0, terminal_hold=False
    )
    assert temporal_data.map_logical_frame(group, 300) == temporal_data.SourceFrameReference(
        episode_id=3, frame_index=0, terminal_hold=False
    )
    assert temporal_data.map_logical_frame(group, 400) == temporal_data.SourceFrameReference(
        episode_id=3, frame_index=99, terminal_hold=True
    )
    assert temporal_data.map_logical_frame(group, 405) == temporal_data.SourceFrameReference(
        episode_id=3, frame_index=99, terminal_hold=True
    )
    with pytest.raises(ValueError, match="terminal hold limit"):
        temporal_data.map_logical_frame(group, 406)


def test_exact_45_frame_activation_produces_one_triplet_without_padding():
    group = _group(lengths=(30, 45, 45, 45))
    rows = temporal_data.build_temporal_sample_rows(
        group, trajectory_id="full-000000", full_episode_id=0, split="train"
    )

    # ``length=45`` means inclusive E=44, so every subtask is excluded by
    # the locked E>=45 gate; there is no padding or cross-subtask fallback.
    assert rows == ()


def test_unreachable_positive_quarantines_whole_trajectory(tmp_path):
    valid = _identities(1)[0]
    short_group = _group(lengths=(14, 60, 60, 60))
    short_identity = temporal_data.TrajectoryIdentityRecord(
        trajectory_id="full-000001",
        mapping_status="matched",
        group=dataclasses.replace(
            short_group,
            group_id=1,
            source_episode_ids=(4, 5, 6, 7),
        ),
        full_episode=temporal_data.FullEpisodeRecord(1, 201),
        exclusion_reason=None,
    )
    manifest = temporal_data.create_temporal_manifest(
        (valid, short_identity),
        source_subtask_repo_id="sub",
        source_subtask_root=tmp_path,
        source_full_repo_id="full",
        source_full_root=tmp_path,
        task_prompts=("task 0", "task 1", "task 2", "task 3"),
    )

    record = next(record for record in manifest.trajectories if record.trajectory_id == "full-000001")
    assert record.split is not None
    assert record.exclusion_reason is None


def test_natural_index_has_one_positive_per_task_and_exact_negative_pools():
    group = _group(lengths=(100, 100, 100, 100))
    rows = temporal_data.build_temporal_sample_rows(group, trajectory_id="full-000000", full_episode_id=0, split="val")

    positives = [row for row in rows if row.sample_kind == "positive"]
    assert [(row.task_index, row.logical_tick) for row in positives] == [(task, 99) for task in range(4)]
    assert len({(row.trajectory_id, row.task_index) for row in positives}) == 4

    for row in rows:
        distance = row.boundary_tick - row.logical_tick
        if row.sample_kind == "hard_negative":
            assert distance == 15
            assert row.label == 0
        elif row.sample_kind == "ordinary_negative":
            assert distance >= 30
            assert distance % 15 == 0
            assert row.label == 0
        assert tuple(
            b - a for a, b in zip(row.history_logical_ticks[:-1], row.history_logical_ticks[1:], strict=True)
        ) == (15, 15)


def test_history_carry_rows_keep_previous_episode_and_prompt_identity():
    group = _group(lengths=(100, 100, 100, 100))
    rows = temporal_data.build_history_carry_transition_rows(
        group,
        trajectory_id="full-000000",
        full_episode_id=0,
        split="train",
    )

    assert len(rows) == 6
    step0 = next(row for row in rows if row.task_index == 1 and row.logical_tick == 0)
    step1 = next(row for row in rows if row.task_index == 1 and row.logical_tick == 15)
    assert step0.label == step1.label == 0
    assert step0.source_episode_ids == (0, 0, 1)
    assert step0.source_frame_indices == (84, 99, 0)
    assert step0.history_prompt_indices == (0, 0, 1)
    assert step1.source_episode_ids == (0, 1, 1)
    assert step1.source_frame_indices == (99, 0, 15)
    assert step1.history_prompt_indices == (0, 1, 1)
    assert all(row.task_index in (1, 2, 3) for row in rows)


def test_positive_history_stays_inside_one_subtask_without_terminal_hold():
    group = _group(lengths=(100, 100, 100, 100))
    rows = temporal_data.build_temporal_sample_rows(group, trajectory_id="full-000000", full_episode_id=0, split="test")
    task0_positive = next(row for row in rows if row.task_index == 0 and row.label == 1)
    task3_positive = next(row for row in rows if row.task_index == 3 and row.label == 1)

    assert task0_positive.prompt_index == 0
    assert task0_positive.history_logical_ticks == (69, 84, 99)
    assert task0_positive.source_episode_ids == (0, 0, 0)
    assert task0_positive.source_frame_indices == (69, 84, 99)
    assert task3_positive.prompt_index == 3
    assert task3_positive.source_episode_ids == (3, 3, 3)
    assert task3_positive.source_frame_indices == (69, 84, 99)
    assert task3_positive.terminal_hold_flags == (False, False, False)


def test_identity_mapping_is_bijective_and_records_737_versus_736_style_unmatched_full():
    identities = _identities(2, extra_full=1)

    assert [identity.mapping_status for identity in identities].count("matched") == 2
    full_only = [identity for identity in identities if identity.mapping_status == "full_only"]
    assert len(full_only) == 1
    assert full_only[0].full_episode.episode_id == 2
    assert full_only[0].exclusion_reason == "no_subtask_group_mapping"


def test_identity_mapping_rejects_unknown_or_duplicate_pairs():
    groups = temporal_data.build_subtask_groups(_episodes_for_groups(2))
    full = (temporal_data.FullEpisodeRecord(0, 240), temporal_data.FullEpisodeRecord(1, 240))
    with pytest.raises(ValueError, match="unknown subtask group"):
        temporal_data.build_trajectory_identities(groups, full, (temporal_data.IdentityMatchRecord(9, 0),))
    with pytest.raises(ValueError, match="multiple full-trajectory mappings"):
        temporal_data.build_trajectory_identities(
            groups,
            full,
            (
                temporal_data.IdentityMatchRecord(0, 0),
                temporal_data.IdentityMatchRecord(0, 1),
            ),
        )


def test_ambiguity_reserves_both_sides_and_does_not_fall_through_to_unmatched():
    groups = temporal_data.build_subtask_groups(_episodes_for_groups(3))
    full = tuple(temporal_data.FullEpisodeRecord(index, 240) for index in range(3))
    ambiguity = temporal_data.IdentityAmbiguityRecord(
        group_ids=(1, 2),
        full_episode_ids=(1, 2),
        candidate_pairs=((1, 1), (1, 2), (2, 1)),
        reason="alignment candidates are tied",
    )

    identities = temporal_data.build_trajectory_identities(
        groups,
        full,
        (temporal_data.IdentityMatchRecord(0, 0),),
        (ambiguity,),
    )

    assert [identity.mapping_status for identity in identities].count("matched") == 1
    ambiguous = [identity for identity in identities if identity.mapping_status == "ambiguous"]
    assert len(ambiguous) == 4
    assert {identity.group.group_id for identity in ambiguous if identity.group is not None} == {1, 2}
    assert {identity.full_episode.episode_id for identity in ambiguous if identity.full_episode is not None} == {1, 2}
    assert all((identity.group is None) != (identity.full_episode is None) for identity in ambiguous)
    assert not any(identity.mapping_status in ("subtask_only", "full_only") for identity in identities)


def test_ambiguity_schema_rejects_incomplete_candidates_and_cross_component_overlap():
    with pytest.raises(ValueError, match="every ambiguous group and full episode"):
        temporal_data.IdentityAmbiguityRecord(
            group_ids=(0, 1),
            full_episode_ids=(0,),
            candidate_pairs=((0, 0),),
            reason="missing candidate",
        )

    groups = temporal_data.build_subtask_groups(_episodes_for_groups(3))
    full = tuple(temporal_data.FullEpisodeRecord(index, 240) for index in range(3))
    first = temporal_data.IdentityAmbiguityRecord((0,), (0,), ((0, 0),), "first tie")
    second = temporal_data.IdentityAmbiguityRecord((1,), (0,), ((1, 0),), "second tie")
    with pytest.raises(ValueError, match="repeats full episodes from another ambiguity"):
        temporal_data.build_trajectory_identities(groups, full, (), (first, second))


def test_manifest_requires_ambiguous_rows_to_be_one_sided_and_explicitly_excluded(tmp_path):
    groups = temporal_data.build_subtask_groups(_episodes_for_groups(2))
    full = tuple(temporal_data.FullEpisodeRecord(index, 240) for index in range(2))
    identities = temporal_data.build_trajectory_identities(
        groups,
        full,
        (temporal_data.IdentityMatchRecord(0, 0),),
        (
            temporal_data.IdentityAmbiguityRecord(
                (1,),
                (1,),
                ((1, 1),),
                "one unresolved pair",
            ),
        ),
    )
    manifest = temporal_data.create_temporal_manifest(
        identities,
        source_subtask_repo_id="subtasks",
        source_subtask_root=tmp_path / "subtasks",
        source_full_repo_id="full",
        source_full_root=tmp_path / "full",
        task_prompts=("task 0", "task 1", "task 2", "task 3"),
    )
    ambiguous_index = next(
        index for index, record in enumerate(manifest.trajectories) if record.mapping_status == "ambiguous"
    )
    records = list(manifest.trajectories)
    records[ambiguous_index] = dataclasses.replace(records[ambiguous_index], exclusion_reason="generic quarantine")
    with pytest.raises(ValueError, match="lacks an explicit ambiguity reason"):
        temporal_data.validate_temporal_manifest(dataclasses.replace(manifest, trajectories=tuple(records)))


def test_split_counts_for_736_are_exact_and_split_is_deterministic(tmp_path):
    identities = _identities(736)
    kwargs = {
        "source_subtask_repo_id": "subtasks",
        "source_subtask_root": tmp_path / "subtasks",
        "source_full_repo_id": "full",
        "source_full_root": tmp_path / "full",
        "task_prompts": ("task 0", "task 1", "task 2", "task 3"),
    }
    first = temporal_data.create_temporal_manifest(identities, **kwargs)
    second = temporal_data.create_temporal_manifest(tuple(reversed(identities)), **kwargs)

    assert first.split_counts == temporal_data.SplitCounts(train=530, val=59, test=147)
    assert first.to_dict() == second.to_dict()
    split_ids = {
        split: {record.trajectory_id for record in first.trajectories if record.split == split}
        for split in temporal_data.SPLIT_NAMES
    }
    assert not (split_ids["train"] & split_ids["val"])
    assert not (split_ids["train"] & split_ids["test"])
    assert not (split_ids["val"] & split_ids["test"])


def test_subtask_logical_manifest_needs_no_full_identity_and_materializes_rows(tmp_path):
    groups = temporal_data.build_subtask_groups(_episodes_for_groups(6, lengths=(60, 60, 60, 60)))

    manifest = temporal_data.create_subtask_temporal_manifest(
        groups,
        source_subtask_repo_id="org/breakfast-subtasks",
        source_subtask_root=tmp_path / "subtasks",
        task_prompts=("task 0", "task 1", "task 2", "task 3"),
    )

    assert manifest.schema_version == 4
    assert manifest.trajectory_source == "subtask_logical"
    assert manifest.source_full_repo_id is None
    assert manifest.source_full_root is None
    assert manifest.split_counts == temporal_data.SplitCounts(train=4, val=1, test=1)
    assert {record.mapping_status for record in manifest.trajectories} == {"subtask_only"}
    assert all(record.full_episode_id is None for record in manifest.trajectories)

    test_record = next(record for record in manifest.trajectories if record.split == "test")
    rows = temporal_data.build_manifest_sample_rows(manifest, "test")
    assert rows
    assert {row.trajectory_id for row in rows} == {test_record.trajectory_id}
    assert {row.full_episode_id for row in rows} == {test_record.group_id}


def test_unmatched_full_is_quarantined_before_split(tmp_path):
    manifest = _manifest(tmp_path, count=10, extra_full=1)

    assert manifest.split_counts.total == 10
    assert len(manifest.trajectories) == 11
    unmatched = [record for record in manifest.trajectories if record.mapping_status == "full_only"]
    assert len(unmatched) == 1
    assert unmatched[0].split is None
    assert unmatched[0].exclusion_reason == "no_subtask_group_mapping"


def test_manifest_round_trip_and_rewrite_protection(tmp_path):
    manifest = _manifest(tmp_path)
    path = tmp_path / "temporal_manifest.json"
    temporal_data.save_temporal_manifest(path, manifest)
    temporal_data.save_temporal_manifest(path, manifest)

    loaded = temporal_data.load_temporal_manifest(path)
    assert loaded == manifest
    changed = dataclasses.replace(manifest, source_full_repo_id="different/full")
    with pytest.raises(FileExistsError, match="refusing to rewrite"):
        temporal_data.save_temporal_manifest(path, changed)


def test_manifest_rejects_content_tampering_and_unknown_schema_fields(tmp_path):
    manifest = _manifest(tmp_path)
    value = manifest.to_dict()
    value["fps"] = 31
    with pytest.raises(ValueError, match="timing"):
        temporal_data.TemporalCompletionManifest.from_dict(value)

    value = manifest.to_dict()
    value["unexpected"] = True
    with pytest.raises(ValueError, match="fields do not match schema"):
        temporal_data.TemporalCompletionManifest.from_dict(value)


def test_manifest_rejects_split_reassignment_even_when_counts_are_unchanged(tmp_path):
    manifest = _manifest(tmp_path, count=20)
    train_index = next(index for index, record in enumerate(manifest.trajectories) if record.split == "train")
    test_index = next(index for index, record in enumerate(manifest.trajectories) if record.split == "test")
    records = list(manifest.trajectories)
    records[train_index] = dataclasses.replace(records[train_index], split="test")
    records[test_index] = dataclasses.replace(records[test_index], split="train")
    reassigned = dataclasses.replace(manifest, trajectories=tuple(records))

    with pytest.raises(ValueError, match="fixed-seed"):
        temporal_data.validate_temporal_manifest(reassigned)


def test_manifest_rows_keep_all_source_views_in_their_trajectory_split(tmp_path):
    manifest = _manifest(tmp_path, count=20)
    train_rows = temporal_data.build_manifest_sample_rows(manifest, "train")
    test_rows = temporal_data.build_manifest_sample_rows(manifest, "test")

    train_trajectories = {row.trajectory_id for row in train_rows}
    test_trajectories = {row.trajectory_id for row in test_rows}
    train_sources = {episode_id for row in train_rows for episode_id in row.source_episode_ids}
    test_sources = {episode_id for row in test_rows for episode_id in row.source_episode_ids}
    assert train_trajectories.isdisjoint(test_trajectories)
    assert train_sources.isdisjoint(test_sources)
    assert all(row.prompt_index == row.task_index for row in (*train_rows, *test_rows))
