from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from xgboost import XGBClassifier

from fraud_detection.config import XGBoostConfig
from fraud_detection.models.base import BaseModel
from fraud_detection.training.imbalance import scale_pos_weight
from fraud_detection.utils import ensure_dir

EvalSet = Optional[List[Tuple[np.ndarray, np.ndarray]]]


class XGBoostModel(BaseModel):
    name = "xgboost"
    supervised = True

    def __init__(self, config: XGBoostConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self.model = XGBClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            min_child_weight=config.min_child_weight,
            reg_lambda=config.reg_lambda,
            tree_method=config.tree_method,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=seed,
            n_jobs=-1,
        )

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        eval_set: EvalSet = None,
    ) -> "XGBoostModel":
        if y is None:
            raise ValueError("XGBoostModel requires labels")
        self.model.set_params(scale_pos_weight=scale_pos_weight(y))
        self.model.fit(X, y, eval_set=eval_set, verbose=False)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self) -> np.ndarray:
        return self.model.feature_importances_

    def save(self, path: str | Path) -> None:
        target = Path(path)
        ensure_dir(target.parent)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostModel":
        return joblib.load(path)
