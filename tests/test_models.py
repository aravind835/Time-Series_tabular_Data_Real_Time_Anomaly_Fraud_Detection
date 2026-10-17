import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fraud_detection.config import (
    AutoencoderConfig,
    IsolationForestConfig,
    XGBoostConfig,
)
from fraud_detection.models import (
    AutoencoderModel,
    IsolationForestModel,
    XGBoostModel,
)


def _separable_dataset(seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(0.0, 1.0, size=(1500, 6))
    anomalies = rng.normal(5.0, 1.0, size=(60, 6))
    X = np.vstack([normal, anomalies])
    y = np.array([0] * 1500 + [1] * 60)
    order = rng.permutation(len(y))
    return X[order], y[order]


def test_xgboost_separates_and_roundtrips(tmp_path):
    X, y = _separable_dataset()
    model = XGBoostModel(XGBoostConfig(n_estimators=60), seed=0).fit(X, y)
    scores = model.score(X)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert average_precision_score(y, scores) > 0.9
    path = tmp_path / "xgb.joblib"
    model.save(path)
    reloaded = XGBoostModel.load(path)
    assert np.allclose(reloaded.score(X), scores)


def test_isolation_forest_scores_range_and_roundtrip(tmp_path):
    X, y = _separable_dataset()
    model = IsolationForestModel(IsolationForestConfig(n_estimators=60), seed=0).fit(X)
    scores = model.score(X)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert average_precision_score(y, scores) > y.mean()
    path = tmp_path / "iso.joblib"
    model.save(path)
    reloaded = IsolationForestModel.load(path)
    assert reloaded.score(X).shape == (len(y),)


def test_autoencoder_scores_range_and_roundtrip(tmp_path):
    X, y = _separable_dataset()
    model = AutoencoderModel(AutoencoderConfig(hidden_dims=[8, 4], epochs=5), seed=0).fit(X, y)
    scores = model.score(X)
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    path = tmp_path / "ae.pt"
    model.save(path)
    reloaded = AutoencoderModel.load(path)
    assert np.allclose(reloaded.score(X), scores, atol=1e-5)


def test_xgboost_requires_labels():
    X, _ = _separable_dataset()
    with pytest.raises(ValueError):
        XGBoostModel(XGBoostConfig(n_estimators=10), seed=0).fit(X)
