"""Fixed train/test split for the breakfast progress dataset."""

TOTAL_SOURCE_EPISODES = 380
SUBTASKS_PER_SOURCE_EPISODE = 4

# Sampled once with random.Random(42).sample(range(380), 5), then sorted and
# committed so every training and evaluation run uses the exact same split.
TEST_SOURCE_EPISODE_INDICES = (12, 57, 140, 327, 379)

TEST_EPISODE_INDICES = tuple(
    source_episode_index * SUBTASKS_PER_SOURCE_EPISODE + subtask_index
    for source_episode_index in TEST_SOURCE_EPISODE_INDICES
    for subtask_index in range(SUBTASKS_PER_SOURCE_EPISODE)
)

_test_episode_indices = frozenset(TEST_EPISODE_INDICES)
TRAIN_EPISODE_INDICES = tuple(
    episode_index
    for episode_index in range(TOTAL_SOURCE_EPISODES * SUBTASKS_PER_SOURCE_EPISODE)
    if episode_index not in _test_episode_indices
)
