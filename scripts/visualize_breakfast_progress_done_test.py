"""Tests for the standalone progress/done visualization."""

# ruff: noqa: SLF001

import numpy as np

from scripts import visualize_breakfast_progress_done as visualize


def test_build_segments_groups_rows_and_preserves_missing_progress_target():
    samples = (
        type(
            "Sample",
            (),
            {"episode_index": 4, "current_sub_task": "load_bread_into_toaster", "query_frame": 10, "label": 0},
        )(),
        type(
            "Sample",
            (),
            {"episode_index": 4, "current_sub_task": "load_bread_into_toaster", "query_frame": 25, "label": 1},
        )(),
    )
    episode = type("Episode", (), {"index": 4, "stage_starts": (10, 25, 40, 55), "terminal_frame": None})()
    dataset = type("Dataset", (), {"samples": samples, "episodes": (episode,)})()
    segments = visualize._build_segments(
        dataset,
        np.asarray([0, 1]),
        np.asarray([0.1, 0.8]),
        np.asarray([0.2, 0.9]),
        np.asarray([0.0, 1.0]),
        np.asarray([True, False]),
        fps=30,
    )
    assert len(segments) == 1
    assert [point.seconds for point in segments[0].points] == [0.0, 0.5]
    assert segments[0].points[-1].progress_target is None


def test_segment_issues_selects_done_error_large_point_error_and_rollback():
    segment = visualize.Segment(
        episode=1,
        task_index=0,
        task_id="task",
        start_frame=0,
        end_frame=30,
        points=(
            visualize.Point(0, 0.0, 0, 0.1, 0.0, 0.1),
            visualize.Point(15, 0.5, 0, 0.2, 0.5, 0.7),
            visualize.Point(30, 1.0, 1, 0.2, 1.0, 0.4),
        ),
    )
    issues = visualize._segment_issues(segment, max_progress_error=0.2, rollback_threshold=0.01)
    assert issues is not None
    assert issues.done_fn == 1
    assert issues.max_abs_error == 0.6
    assert np.isclose(issues.max_rollback, 0.3)
    assert len(issues.reasons) == 3


def test_html_embeds_all_images_and_has_case_and_point_navigation(tmp_path):
    (tmp_path / "frames").mkdir()
    images = []
    points = []
    for index in range(3):
        path = tmp_path / "frames" / f"{index}.jpg"
        path.write_bytes(b"jpeg")
        images.append({"path": f"frames/{index}.jpg", "frame": index, "seconds": index / 2})
        points.append(
            {
                "frame": index,
                "seconds": index / 2,
                "done_target": 0,
                "done_score": 0.1,
                "progress_target": index / 2,
                "progress_score": index / 2,
            }
        )
    pages = visualize._embed_assets(
        [
            {
                "episode": 1,
                "task_index": 2,
                "task_id": "pour",
                "reasons": ["Progress rollback 0.1"],
                "focus_index": 1,
                "boundary_seconds": 1.0,
                "points": points,
                "images": images,
            }
        ],
        tmp_path,
    )
    document = visualize._html_document(pages, checkpoint=visualize.DEFAULT_CHECKPOINT)
    assert "Previous case" in document
    assert "Next case" in document
    assert "prevPoint" in document
    assert "pointSlider" in document
    assert "frames/" not in document
    assert document.count("data:image/jpeg;base64,") == 3
    assert "ArrowLeft" in document
    assert "PageDown" in document
    assert "['INPUT','SELECT','BUTTON'].includes(e.target.tagName)" in document
