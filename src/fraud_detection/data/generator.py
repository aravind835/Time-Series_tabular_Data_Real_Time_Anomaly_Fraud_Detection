from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

TRANSACTION_ID = "transaction_id"
TIMESTAMP = "timestamp"
FRAUD_LABEL = "is_fraud"

_BASE_EPOCH = 1_700_000_000
_SPAN_SECONDS = 30 * 24 * 3600


def feature_columns(n_features: int) -> List[str]:
    return ["amount", "hour", "day_of_week"] + [f"f{i}" for i in range(n_features)]


def _hour_weights() -> Tuple[np.ndarray, np.ndarray]:
    hours = np.arange(24)
    legit = np.exp(-0.5 * ((hours - 14.0) / 5.0) ** 2) + 0.15
    circular = np.minimum(np.abs(hours - 2), 24 - np.abs(hours - 2))
    fraud = np.exp(-0.5 * (circular / 3.0) ** 2) + 0.05
    return legit / legit.sum(), fraud / fraud.sum()


def generate_transactions(
    n_samples: int,
    n_features: int,
    n_informative: int,
    fraud_rate: float,
    drift_strength: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_informative = int(np.clip(n_informative, 1, n_features))
    n_fraud = max(1, int(round(n_samples * fraud_rate)))

    labels = np.zeros(n_samples, dtype=np.int64)
    fraud_idx = rng.choice(n_samples, size=n_fraud, replace=False)
    labels[fraud_idx] = 1
    is_fraud = labels == 1

    time_fraction = np.linspace(0.0, 1.0, n_samples)

    latent = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    drift_direction = rng.normal(size=n_informative)
    for j in range(n_informative):
        latent[:, j] += drift_strength * time_fraction * drift_direction[j]

    separation = np.zeros(n_features)
    separation[:n_informative] = rng.uniform(1.0, 2.0, size=n_informative) * rng.choice(
        [-1.0, 1.0], size=n_informative
    )
    latent[is_fraud] += separation
    latent[is_fraud] += rng.normal(0.0, 0.6, size=(n_fraud, n_features))

    amount = rng.lognormal(mean=3.0, sigma=1.0, size=n_samples)
    amount[is_fraud] = rng.lognormal(mean=3.8, sigma=1.1, size=n_fraud)
    amount = np.round(amount, 2)

    legit_weights, fraud_weights = _hour_weights()
    hour = np.empty(n_samples, dtype=np.int64)
    hour[~is_fraud] = rng.choice(24, size=(~is_fraud).sum(), p=legit_weights)
    hour[is_fraud] = rng.choice(24, size=n_fraud, p=fraud_weights)

    gaps = rng.exponential(_SPAN_SECONDS / n_samples, size=n_samples)
    timestamps = _BASE_EPOCH + np.cumsum(gaps)
    day_of_week = ((timestamps // 86400).astype(np.int64)) % 7

    frame = pd.DataFrame(
        {
            TRANSACTION_ID: np.arange(n_samples, dtype=np.int64),
            TIMESTAMP: timestamps,
            "amount": amount,
            "hour": hour,
            "day_of_week": day_of_week,
        }
    )
    for i in range(n_features):
        frame[f"f{i}"] = latent[:, i]
    frame[FRAUD_LABEL] = labels
    return frame
