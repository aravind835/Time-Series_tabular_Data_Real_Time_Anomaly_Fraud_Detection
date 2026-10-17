from typing import Dict, Type

from fraud_detection.models.autoencoder import AutoencoderModel
from fraud_detection.models.base import BaseModel
from fraud_detection.models.isolation_forest import IsolationForestModel
from fraud_detection.models.xgboost_model import XGBoostModel

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    IsolationForestModel.name: IsolationForestModel,
    XGBoostModel.name: XGBoostModel,
    AutoencoderModel.name: AutoencoderModel,
}

__all__ = [
    "BaseModel",
    "IsolationForestModel",
    "XGBoostModel",
    "AutoencoderModel",
    "MODEL_REGISTRY",
]
