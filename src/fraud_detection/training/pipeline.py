from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np

from fraud_detection.artifacts import ArtifactPaths
from fraud_detection.config import Config
from fraud_detection.data.loader import DataSplit, load_or_generate, split_dataframe
from fraud_detection.evaluation import (
    compute_metrics,
    precision_recall_points,
    select_threshold,
)
from fraud_detection.features import Preprocessor
from fraud_detection.models import (
    AutoencoderModel,
    BaseModel,
    IsolationForestModel,
    XGBoostModel,
)
from fraud_detection.training.imbalance import resample
from fraud_detection.utils import ensure_dir, set_seed, write_json


def _build_models(config: Config) -> Dict[str, BaseModel]:
    return {
        IsolationForestModel.name: IsolationForestModel(
            config.models.isolation_forest, config.seed
        ),
        XGBoostModel.name: XGBoostModel(config.models.xgboost, config.seed),
        AutoencoderModel.name: AutoencoderModel(config.models.autoencoder, config.seed),
    }


def _fit_model(
    model: BaseModel,
    split: DataSplit,
    X_train: np.ndarray,
    X_resampled: np.ndarray,
    y_resampled: np.ndarray,
    X_val: np.ndarray,
) -> None:
    if isinstance(model, XGBoostModel):
        model.fit(X_resampled, y_resampled, eval_set=[(X_val, split.y_val)])
    elif isinstance(model, AutoencoderModel):
        model.fit(X_train, split.y_train)
    else:
        model.fit(X_train)


def _latency_profile(model: BaseModel, sample: np.ndarray, iterations: int = 300) -> Dict[str, float]:
    row = sample[:1]
    for _ in range(10):
        model.score(row)
    durations = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        start = time.perf_counter()
        model.score(row)
        durations[index] = (time.perf_counter() - start) * 1000.0
    return {
        "p50_ms": float(np.percentile(durations, 50)),
        "p95_ms": float(np.percentile(durations, 95)),
        "p99_ms": float(np.percentile(durations, 99)),
    }


def train_and_save(config: Config, artifacts_dir: str | Path | None = None) -> Dict:
    set_seed(config.seed)
    paths = ArtifactPaths(ensure_dir(artifacts_dir or config.serving.artifacts_dir))

    frame = load_or_generate(config)
    split = split_dataframe(frame, config)

    preprocessor = Preprocessor(
        split.feature_names,
        scaler=config.preprocessing.scaler,
        log_amount=config.preprocessing.log_amount,
    )
    X_train = preprocessor.fit_transform(split.X_train)
    X_val = preprocessor.transform(split.X_val)
    X_test = preprocessor.transform(split.X_test)
    preprocessor.save(paths.preprocessor)

    X_resampled, y_resampled = resample(X_train, split.y_train, config.imbalance, config.seed)

    models = _build_models(config)
    thresholds: Dict[str, Dict] = {}
    metrics: Dict[str, Dict] = {}
    curves: Dict[str, list] = {}

    for name, model in models.items():
        start = time.perf_counter()
        _fit_model(model, split, X_train, X_resampled, y_resampled, X_val)
        train_seconds = time.perf_counter() - start

        val_scores = model.score(X_val)
        threshold_info = select_threshold(
            split.y_val,
            val_scores,
            target_recall=config.evaluation.target_recall,
            beta=config.evaluation.beta,
        )
        test_scores = model.score(X_test)
        model_metrics = compute_metrics(
            split.y_test,
            test_scores,
            threshold_info["threshold"],
            beta=config.evaluation.beta,
        )
        model_metrics["train_seconds"] = float(train_seconds)
        model_metrics["latency"] = _latency_profile(model, X_test)

        model.save(paths.model(name))
        thresholds[name] = threshold_info
        metrics[name] = model_metrics
        curves[name] = precision_recall_points(split.y_test, test_scores)

        print(
            f"[{name}] PR-AUC={model_metrics['pr_auc']:.4f} "
            f"precision={model_metrics['precision']:.3f} "
            f"recall={model_metrics['recall']:.3f} "
            f"threshold={threshold_info['threshold']:.4f} "
            f"p95={model_metrics['latency']['p95_ms']:.3f}ms"
        )

    metadata = {
        "seed": config.seed,
        "feature_names": split.feature_names,
        "n_features": len(split.feature_names),
        "default_model": config.serving.model,
        "models": list(models.keys()),
        "data_summary": split.summary(),
        "imbalance": {
            "method": config.imbalance.method,
            "sampling_strategy": config.imbalance.sampling_strategy,
            "train_size_after_resample": int(X_resampled.shape[0]),
            "train_positives_after_resample": int(np.sum(y_resampled == 1)),
        },
        "target_recall": config.evaluation.target_recall,
        "beta": config.evaluation.beta,
    }

    write_json(paths.thresholds, thresholds)
    write_json(paths.metrics, {"metrics": metrics, "curves": curves})
    write_json(paths.metadata, metadata)

    return {"metrics": metrics, "thresholds": thresholds, "metadata": metadata}
