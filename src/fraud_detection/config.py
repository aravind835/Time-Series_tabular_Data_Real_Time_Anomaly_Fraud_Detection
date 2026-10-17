from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class DataConfig:
    n_samples: int = 200000
    n_features: int = 20
    n_informative: int = 8
    fraud_rate: float = 0.001
    drift_strength: float = 0.6
    output_dir: str = "data"
    filename: str = "transactions.parquet"


@dataclass
class SplitConfig:
    test_size: float = 0.2
    val_size: float = 0.1
    time_ordered: bool = True


@dataclass
class PreprocessingConfig:
    scaler: str = "standard"
    log_amount: bool = True


@dataclass
class ImbalanceConfig:
    method: str = "smote"
    sampling_strategy: float = 0.05
    k_neighbors: int = 5


@dataclass
class IsolationForestConfig:
    n_estimators: int = 200
    max_samples: float = 256
    contamination: float = 0.001


@dataclass
class XGBoostConfig:
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 5.0
    reg_lambda: float = 1.0
    tree_method: str = "hist"


@dataclass
class AutoencoderConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [32, 16, 8])
    epochs: int = 25
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 0.00001


@dataclass
class ModelsConfig:
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)


@dataclass
class EvaluationConfig:
    target_recall: float = 0.85
    beta: float = 2.0
    reports_dir: str = "reports"


@dataclass
class ServingConfig:
    model: str = "xgboost"
    host: str = "127.0.0.1"
    port: int = 8000
    artifacts_dir: str = "artifacts"


@dataclass
class StreamingConfig:
    rate_per_second: int = 200
    duration_seconds: int = 15
    model: str = "xgboost"


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    imbalance: ImbalanceConfig = field(default_factory=ImbalanceConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)


def _from_dict(cls: type, values: Dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        return values
    kwargs: Dict[str, Any] = {}
    valid = {f.name: f for f in fields(cls)}
    for key, value in values.items():
        if key not in valid:
            continue
        field_type = valid[key].type
        if isinstance(value, dict) and is_dataclass(_resolve(field_type)):
            kwargs[key] = _from_dict(_resolve(field_type), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _resolve(field_type: Any) -> Any:
    if isinstance(field_type, str):
        return globals().get(field_type, object)
    return field_type


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return _from_dict(Config, data)
