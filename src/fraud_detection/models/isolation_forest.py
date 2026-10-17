from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from fraud_detection.config import IsolationForestConfig
from fraud_detection.models.base import BaseModel, apply_normalizer, fit_normalizer
from fraud_detection.utils import ensure_dir


class IsolationForestModel(BaseModel):
    name = "isolation_forest"
    supervised = False

    def __init__(self, config: IsolationForestConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self.model = IsolationForest(
            n_estimators=config.n_estimators,
            max_samples=config.max_samples,
            contamination=config.contamination,
            random_state=seed,
            n_jobs=-1,
        )
        self.low = 0.0
        self.high = 1.0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IsolationForestModel":
        self.model.fit(X)
        raw = -self.model.score_samples(X)
        self.low, self.high = fit_normalizer(raw)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        raw = -self.model.score_samples(X)
        return apply_normalizer(raw, self.low, self.high)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        ensure_dir(target.parent)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: str | Path) -> "IsolationForestModel":
        return joblib.load(path)
