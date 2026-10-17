from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import precision_recall_curve


def select_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float = 0.85,
    beta: float = 1.0,
) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision = precision[:-1]
    recall = recall[:-1]

    if len(thresholds) == 0:
        return {
            "threshold": 0.5,
            "threshold_target_recall": 0.5,
            "threshold_best_fbeta": 0.5,
            "target_recall": float(target_recall),
            "beta": float(beta),
            "precision_at_threshold": 0.0,
            "recall_at_threshold": 0.0,
        }

    beta_squared = beta * beta
    fbeta = (1 + beta_squared) * precision * recall / (
        beta_squared * precision + recall + 1e-12
    )
    best_fbeta_index = int(np.argmax(fbeta))
    threshold_best_fbeta = float(thresholds[best_fbeta_index])

    meets_recall = recall >= target_recall
    if np.any(meets_recall):
        candidate_indices = np.where(meets_recall)[0]
        best_index = candidate_indices[int(np.argmax(precision[candidate_indices]))]
    else:
        best_index = 0
    threshold_target_recall = float(thresholds[best_index])

    return {
        "threshold": threshold_target_recall,
        "threshold_target_recall": threshold_target_recall,
        "threshold_best_fbeta": threshold_best_fbeta,
        "target_recall": float(target_recall),
        "beta": float(beta),
        "precision_at_threshold": float(precision[best_index]),
        "recall_at_threshold": float(recall[best_index]),
    }
