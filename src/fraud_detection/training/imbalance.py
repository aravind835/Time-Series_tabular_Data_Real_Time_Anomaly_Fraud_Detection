from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from fraud_detection.config import ImbalanceConfig


def scale_pos_weight(y: np.ndarray) -> float:
    positives = float(np.sum(y == 1))
    negatives = float(np.sum(y == 0))
    if positives == 0:
        return 1.0
    return negatives / positives


def class_weights(y: np.ndarray) -> Dict[int, float]:
    total = float(len(y))
    positives = float(np.sum(y == 1))
    negatives = total - positives
    if positives == 0 or negatives == 0:
        return {0: 1.0, 1: 1.0}
    return {
        0: total / (2.0 * negatives),
        1: total / (2.0 * positives),
    }


def resample(
    X: np.ndarray,
    y: np.ndarray,
    config: ImbalanceConfig,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if config.method == "none":
        return X, y
    n_minority = int(np.sum(y == 1))
    if n_minority < 2:
        return X, y
    k_neighbors = int(min(config.k_neighbors, n_minority - 1))
    if config.method == "smote":
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(
            sampling_strategy=config.sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=seed,
        )
    elif config.method == "smote_tomek":
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE

        sampler = SMOTETomek(
            smote=SMOTE(
                sampling_strategy=config.sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=seed,
            ),
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown imbalance method: {config.method}")
    return sampler.fit_resample(X, y)
