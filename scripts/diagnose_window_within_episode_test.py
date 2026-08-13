"""Synthetic tests for ``diagnose_window_within_episode._compute_within_episode_metrics``.

These construct hand-built feature layouts with a known nearest-neighbor
structure (independent of the frozen model / dataset) and assert the metrics
report what the geometry says -- so we know the analysis is correct before
trusting its output on real features.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.diagnose_window_within_episode import _compute_within_episode_metrics


def _metrics(episode_index, target, feature):
    return _compute_within_episode_metrics(
        np.asarray(episode_index, dtype=np.int32),
        np.asarray(target, dtype=np.float64),
        np.asarray(feature, dtype=np.float64),
    )


def test_confused_positive_and_separable_cluster():
    """Episode 0: a positive frame whose nearest neighbor is a negative frame.
    Episode 1: positives and negatives form two separable clusters.

    Built so the cosine-distance nearest-neighbor structure is unambiguous.
    """

    # Episode 0 (3 frames): the positive frame sits next to a negative frame
    # in feature space, far from the other negative frame.
    # Episode 1 (4 frames): two negatives cluster together, two positives
    # cluster together -- every frame's nearest neighbor shares its label.
    episode_index = [0, 0, 0, 1, 1, 1, 1]
    target = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]
    feature = [
        [1.0, 0.0],   # ep0 frame0 (neg)
        [0.0, 1.0],   # ep0 frame1 (neg)
        [0.99, 0.01],  # ep0 frame2 (pos)  -- closest to frame0 (neg)
        [1.0, 0.0],   # ep1 frame0 (neg)
        [0.99, 0.01],  # ep1 frame1 (neg)  -- closest to frame0 (neg)
        [0.0, 1.0],   # ep1 frame2 (pos)  -- closest to frame3 (pos)
        [0.01, 0.99],  # ep1 frame3 (pos)  -- closest to frame2 (pos)
    ]

    metrics = _metrics(episode_index, target, feature)

    assert metrics["frame_count"] == 7
    assert metrics["positive_count"] == 3
    assert metrics["negative_count"] == 4
    # ep0: 3 frames, all mismatched (0/3 same); ep1: 4 frames, all matched (4/4).
    assert metrics["same_label_rate"] == pytest.approx(4 / 7)
    # All three positives have a within-episode neighbor; only ep0's positive
    # is confused (its nearest is a negative frame).
    assert metrics["positive_total"] == 3
    assert metrics["positive_confused"] == 1
    assert len(metrics["worst"]) == 1
    distance, self_index, neighbor_index = metrics["worst"][0]
    assert self_index == 2          # ep0 frame2 (the positive)
    assert neighbor_index == 0      # ep0 frame0 (the negative look-alike)
    # cosine distance between [1,0] and [0.99,0.01] (the latter is NOT unit-length,
    # so the naive 0.01 gap is wrong -- normalize by its L2 norm).
    expected_distance = 1 - 0.99 / np.sqrt(0.99**2 + 0.01**2)
    assert distance == pytest.approx(expected_distance, abs=1e-6)


def test_single_frame_episode_is_skipped():
    """An episode with only one sampled frame cannot have a within-episode
    neighbor and must contribute nothing -- no division-by-zero, no crash."""

    episode_index = [0, 1, 1]
    target = [1.0, 0.0, 1.0]
    feature = [
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    metrics = _metrics(episode_index, target, feature)

    # Episode 0 has a single frame -> skipped. Episode 1's two frames are each
    # other's nearest; the positive (frame 2) is confused (only neighbor is neg).
    assert metrics["same_label_rate"] == pytest.approx(0.0)
    assert metrics["positive_total"] == 1
    assert metrics["positive_confused"] == 1


def test_all_positive_episode_has_no_confused_pairs():
    """If every frame in an episode is positive, none can be confused with a
    negative frame -- ``worst`` stays empty even though same_label_rate is 1.0."""

    episode_index = [0, 0, 0]
    target = [1.0, 1.0, 1.0]
    feature = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ]

    metrics = _metrics(episode_index, target, feature)

    assert metrics["positive_total"] == 3
    assert metrics["positive_confused"] == 0
    assert metrics["worst"] == []
    assert metrics["same_label_rate"] == pytest.approx(1.0)
