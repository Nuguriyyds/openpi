from openpi.training import breakfast_progress_split as split


def test_breakfast_progress_split_is_complete_and_grouped():
    total_episodes = split.TOTAL_SOURCE_EPISODES * split.SUBTASKS_PER_SOURCE_EPISODE
    train_episodes = set(split.TRAIN_EPISODE_INDICES)
    test_episodes = set(split.TEST_EPISODE_INDICES)

    assert len(split.TEST_SOURCE_EPISODE_INDICES) == 5
    assert len(train_episodes) == 1500
    assert len(test_episodes) == 20
    assert train_episodes.isdisjoint(test_episodes)
    assert train_episodes | test_episodes == set(range(total_episodes))

    for source_episode_index in split.TEST_SOURCE_EPISODE_INDICES:
        expected_subtasks = {
            source_episode_index * split.SUBTASKS_PER_SOURCE_EPISODE + subtask_index
            for subtask_index in range(split.SUBTASKS_PER_SOURCE_EPISODE)
        }
        assert expected_subtasks <= test_episodes

