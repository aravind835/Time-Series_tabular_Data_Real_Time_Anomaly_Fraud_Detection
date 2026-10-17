from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    beta: float = 1.0,
) -> Dict[str, float]:
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)

    positives = int(labels.sum())
    both_classes = len(np.unique(labels)) > 1

    pr_auc = float(average_precision_score(labels, scores)) if positives > 0 else float("nan")
    roc_auc = float(roc_auc_score(labels, scores)) if both_classes else float("nan")

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    beta_squared = beta * beta
    fbeta = _safe_ratio(
        (1 + beta_squared) * precision * recall, beta_squared * precision + recall
    )

    return {
        "threshold": float(threshold),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fbeta": fbeta,
        "beta": float(beta),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "positives": positives,
        "total": int(labels.shape[0]),
        "alert_rate": float(predictions.mean()),
    }


def precision_recall_points(
    y_true: np.ndarray, scores: np.ndarray, max_points: int = 200
) -> List[Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision = precision[:-1]
    recall = recall[:-1]
    count = len(thresholds)
    if count == 0:
        return []
    step = max(1, count // max_points)
    points = []
    for index in range(0, count, step):
        points.append(
            {
                "threshold": float(thresholds[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
            }
        )
    return points
