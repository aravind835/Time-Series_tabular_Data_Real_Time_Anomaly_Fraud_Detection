from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, List, Optional, Sequence

import numpy as np

from fraud_detection.artifacts import ArtifactPaths, load_bundle
from fraud_detection.utils import read_json


class LatencyTracker:
    def __init__(self, maxlen: int = 5000) -> None:
        self._samples: Deque[float] = deque(maxlen=maxlen)
        self._lock = Lock()
        self.total = 0

    def record(self, milliseconds: float) -> None:
        with self._lock:
            self._samples.append(milliseconds)
            self.total += 1

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            values = np.array(self._samples, dtype=np.float64)
            total = self.total
        if values.size == 0:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0}
        return {
            "count": total,
            "window": int(values.size),
            "mean_ms": float(values.mean()),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "max_ms": float(values.max()),
        }


class Predictor:
    def __init__(self, artifacts_dir: str | Path, model_name: Optional[str] = None) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        metadata = read_json(ArtifactPaths(self.artifacts_dir).metadata)
        self.model_name = model_name or metadata["default_model"]
        preprocessor, model, threshold, meta = load_bundle(self.artifacts_dir, self.model_name)
        self.preprocessor = preprocessor
        self.model = model
        self.threshold = threshold
        self.feature_names: List[str] = meta["feature_names"]
        self.metadata = meta
        self.latency = LatencyTracker()
        self.alerts = 0

    def _vectorize(self, record: Dict[str, float]) -> np.ndarray:
        return np.array(
            [float(record.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float64,
        )

    def _vectorize_many(self, records: Sequence[Dict[str, float]]) -> np.ndarray:
        return np.array([self._vectorize(record) for record in records], dtype=np.float64)

    def predict(self, record: Dict[str, float]) -> Dict[str, object]:
        start = time.perf_counter()
        features = self.preprocessor.transform(self._vectorize(record).reshape(1, -1))
        score = float(self.model.score(features)[0])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.latency.record(elapsed_ms)
        is_fraud = score >= self.threshold
        if is_fraud:
            self.alerts += 1
        return {
            "model": self.model_name,
            "score": score,
            "threshold": self.threshold,
            "is_fraud": bool(is_fraud),
            "latency_ms": elapsed_ms,
        }

    def predict_batch(self, records: Sequence[Dict[str, float]]) -> List[Dict[str, object]]:
        if not records:
            return []
        start = time.perf_counter()
        features = self.preprocessor.transform(self._vectorize_many(records))
        scores = self.model.score(features)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_record_ms = elapsed_ms / len(records)
        results = []
        for score in scores:
            self.latency.record(per_record_ms)
            is_fraud = float(score) >= self.threshold
            if is_fraud:
                self.alerts += 1
            results.append(
                {
                    "model": self.model_name,
                    "score": float(score),
                    "threshold": self.threshold,
                    "is_fraud": bool(is_fraud),
                    "latency_ms": per_record_ms,
                }
            )
        return results

    def metrics(self) -> Dict[str, object]:
        snapshot = self.latency.snapshot()
        return {
            "model": self.model_name,
            "threshold": self.threshold,
            "alerts": self.alerts,
            "latency": snapshot,
        }

    def info(self) -> Dict[str, object]:
        return {
            "model": self.model_name,
            "threshold": self.threshold,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names),
            "available_models": self.metadata.get("models", []),
        }
