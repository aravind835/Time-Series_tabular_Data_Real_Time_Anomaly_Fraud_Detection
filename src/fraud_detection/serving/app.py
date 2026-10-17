from __future__ import annotations

import os
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from fraud_detection.serving.predictor import Predictor


class Transaction(BaseModel):
    features: Dict[str, float] = Field(default_factory=dict)


class BatchRequest(BaseModel):
    transactions: List[Dict[str, float]] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    model: str
    score: float
    threshold: float
    is_fraud: bool
    latency_ms: float


def create_app(artifacts_dir: str, model_name: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Real-Time Fraud Detection", version="0.1.0")
    predictor = Predictor(artifacts_dir, model_name)

    @app.get("/health")
    def health() -> Dict[str, object]:
        return {"status": "ok", "model": predictor.model_name}

    @app.get("/info")
    def info() -> Dict[str, object]:
        return predictor.info()

    @app.get("/metrics")
    def metrics() -> Dict[str, object]:
        return predictor.metrics()

    @app.post("/predict", response_model=PredictionResponse)
    def predict(transaction: Transaction) -> Dict[str, object]:
        return predictor.predict(transaction.features)

    @app.post("/predict/batch", response_model=List[PredictionResponse])
    def predict_batch(request: BatchRequest) -> List[Dict[str, object]]:
        return predictor.predict_batch(request.transactions)

    return app


def create_app_from_env() -> FastAPI:
    artifacts_dir = os.getenv("FRAUD_ARTIFACTS_DIR", "artifacts")
    model_name = os.getenv("FRAUD_MODEL") or None
    return create_app(artifacts_dir, model_name)


def __getattr__(name: str) -> FastAPI:
    if name == "app":
        return create_app_from_env()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
