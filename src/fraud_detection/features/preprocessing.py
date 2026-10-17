from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

from fraud_detection.utils import ensure_dir


class Preprocessor:
    def __init__(
        self,
        feature_names: List[str],
        scaler: str = "standard",
        log_amount: bool = True,
    ) -> None:
        self.feature_names = list(feature_names)
        self.scaler_kind = scaler
        self.log_amount = log_amount
        self.amount_index = (
            self.feature_names.index("amount")
            if log_amount and "amount" in self.feature_names
            else None
        )
        self.scaler = self._build_scaler(scaler)
        self.fitted = False

    @staticmethod
    def _build_scaler(kind: str):
        if kind == "robust":
            return RobustScaler()
        if kind == "standard":
            return StandardScaler()
        raise ValueError(f"unknown scaler: {kind}")

    def _apply_log(self, X: np.ndarray) -> np.ndarray:
        if self.amount_index is None:
            return X
        transformed = np.asarray(X, dtype=np.float64).copy()
        column = transformed[:, self.amount_index]
        transformed[:, self.amount_index] = np.log1p(np.clip(column, a_min=0.0, a_max=None))
        return transformed

    def fit(self, X: np.ndarray) -> "Preprocessor":
        self.scaler.fit(self._apply_log(X))
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return self.scaler.transform(self._apply_log(matrix))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        ensure_dir(target.parent)
        joblib.dump(self, target)

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        return joblib.load(path)
