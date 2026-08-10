import numpy as np

from scripts import evaluate_progress_head


def test_progress_evaluator_accepts_continuous_targets_and_reports_regression_metrics(tmp_path):
    prediction_file = tmp_path / "predictions.npz"
    np.savez_compressed(
        prediction_file,
        episode_index=np.asarray([2, 2, 2, 5, 5], dtype=np.int32),
        task_index=np.asarray([7, 7, 7, 8, 8], dtype=np.int16),
        frame_index=np.asarray([0, 1, 2, 0, 1], dtype=np.int32),
        logit=np.asarray([-5.0, 0.0, 5.0, -2.0, 2.0], dtype=np.float32),
        target=np.asarray([0.0, 0.5, 1.0, 0.0, 1.0], dtype=np.float32),
        infer_ms=np.asarray([1.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float32),
    )

    metrics = evaluate_progress_head.compute_metrics(prediction_file, huber_delta=0.1)

    assert metrics["overall"]["frame_count"] == 5
    assert metrics["overall"]["mae"] < 0.1
    assert metrics["overall"]["rmse"] < 0.1
    assert metrics["overall"]["pearson"] > 0.9
    assert metrics["overall"]["spearman"] > 0.9
    assert metrics["overall"]["prediction_min"] >= 0.0
    assert metrics["overall"]["prediction_max"] <= 1.0
    assert metrics["overall"]["early_mae"] >= 0.0
    assert metrics["overall"]["late_mae"] >= 0.0
    assert [row["episode_index"] for row in metrics["episodes"]] == [2, 5]
    assert {row["task_index"] for row in metrics["episodes"]} == {7, 8}
