from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import Config
from fraud_detection.data.generator import (
    FRAUD_LABEL,
    TIMESTAMP,
    feature_columns,
    generate_transactions,
)
from fraud_detection.utils import ensure_dir


@dataclass
class DataSplit:
    feature_names: List[str]
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame

    def summary(self) -> dict:
        return {
            "features": len(self.feature_names),
            "train": _counts(self.y_train),
            "val": _counts(self.y_val),
            "test": _counts(self.y_test),
        }


def _counts(y: np.ndarray) -> dict:
    positives = int(y.sum())
    total = int(y.shape[0])
    rate = positives / total if total else 0.0
    return {"total": total, "positives": positives, "positive_rate": rate}


def load_or_generate(config: Config) -> pd.DataFrame:
    output_dir = ensure_dir(config.data.output_dir)
    path = output_dir / config.data.filename
    if path.exists():
        return pd.read_parquet(path)
    frame = generate_transactions(
        n_samples=config.data.n_samples,
        n_features=config.data.n_features,
        n_informative=config.data.n_informative,
        fraud_rate=config.data.fraud_rate,
        drift_strength=config.data.drift_strength,
        seed=config.seed,
    )
    frame.to_parquet(path, index=False)
    return frame


def split_dataframe(frame: pd.DataFrame, config: Config) -> DataSplit:
    names = feature_columns(config.data.n_features)
    if config.split.time_ordered:
        train_frame, val_frame, test_frame = _time_ordered_split(frame, config)
    else:
        train_frame, val_frame, test_frame = _stratified_split(frame, config)
    return DataSplit(
        feature_names=names,
        X_train=train_frame[names].to_numpy(dtype=np.float64),
        y_train=train_frame[FRAUD_LABEL].to_numpy(dtype=np.int64),
        X_val=val_frame[names].to_numpy(dtype=np.float64),
        y_val=val_frame[FRAUD_LABEL].to_numpy(dtype=np.int64),
        X_test=test_frame[names].to_numpy(dtype=np.float64),
        y_test=test_frame[FRAUD_LABEL].to_numpy(dtype=np.int64),
        train_frame=train_frame.reset_index(drop=True),
        val_frame=val_frame.reset_index(drop=True),
        test_frame=test_frame.reset_index(drop=True),
    )


def _time_ordered_split(frame: pd.DataFrame, config: Config):
    ordered = frame.sort_values(TIMESTAMP).reset_index(drop=True)
    n = len(ordered)
    test_start = int(n * (1.0 - config.split.test_size))
    val_start = int(test_start * (1.0 - config.split.val_size))
    train_frame = ordered.iloc[:val_start]
    val_frame = ordered.iloc[val_start:test_start]
    test_frame = ordered.iloc[test_start:]
    return train_frame, val_frame, test_frame


def _stratified_split(frame: pd.DataFrame, config: Config):
    labels = frame[FRAUD_LABEL]
    train_val, test_frame = train_test_split(
        frame,
        test_size=config.split.test_size,
        stratify=labels,
        random_state=config.seed,
    )
    val_fraction = config.split.val_size
    train_frame, val_frame = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val[FRAUD_LABEL],
        random_state=config.seed,
    )
    return train_frame, val_frame, test_frame
