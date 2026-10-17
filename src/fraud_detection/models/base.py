from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class BaseModel(ABC):
    name: str = "base"
    supervised: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BaseModel":
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseModel":
        ...


def fit_normalizer(
    scores: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0
) -> Tuple[float, float]:
    low = float(np.percentile(scores, low_percentile))
    high = float(np.percentile(scores, high_percentile))
    if high <= low:
        high = low + 1e-9
    return low, high


def apply_normalizer(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((scores - low) / (high - low), 0.0, 1.0)
