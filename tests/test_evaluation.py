import numpy as np

from fraud_detection.evaluation import (
    compute_metrics,
    precision_recall_points,
    select_threshold,
)


def test_compute_metrics_on_known_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = compute_metrics(y, scores, threshold=0.5)
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["pr_auc"] > 0.99


def test_select_threshold_meets_target_recall():
    rng = np.random.default_rng(0)
    y = np.array([0] * 900 + [1] * 100)
    scores = np.concatenate([
        rng.uniform(0.0, 0.6, size=900),
        rng.uniform(0.4, 1.0, size=100),
    ])
    info = select_threshold(y, scores, target_recall=0.8, beta=2.0)
    predictions = (scores >= info["threshold"]).astype(int)
    recall = predictions[y == 1].mean()
    assert recall >= 0.8


def test_precision_recall_points_non_empty():
    y = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.7, 0.3, 0.6])
    points = precision_recall_points(y, scores)
    assert len(points) > 0
    assert set(points[0].keys()) == {"threshold", "precision", "recall"}
