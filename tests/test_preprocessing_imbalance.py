import numpy as np

from fraud_detection.config import ImbalanceConfig
from fraud_detection.features import Preprocessor
from fraud_detection.training import class_weights, resample, scale_pos_weight


def test_standard_scaler_standardizes():
    names = ["amount", "hour", "day_of_week", "f0", "f1"]
    rng = np.random.default_rng(0)
    X = rng.normal(5.0, 3.0, size=(500, len(names)))
    X[:, 0] = np.abs(X[:, 0]) + 1.0
    pre = Preprocessor(names, scaler="standard", log_amount=True)
    transformed = pre.fit_transform(X)
    assert transformed.shape == X.shape
    assert abs(float(transformed.mean())) < 0.1
    assert abs(float(transformed.std()) - 1.0) < 0.1


def test_single_row_transform():
    names = ["amount", "f0"]
    pre = Preprocessor(names)
    pre.fit(np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]]))
    out = pre.transform(np.array([15.0, 1.5]))
    assert out.shape == (1, 2)


def test_scale_pos_weight_and_class_weights():
    y = np.array([0] * 990 + [1] * 10)
    assert scale_pos_weight(y) == 99.0
    weights = class_weights(y)
    assert weights[1] > weights[0]


def test_smote_increases_minority():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 8))
    y = np.zeros(2000, dtype=int)
    y[:20] = 1
    config = ImbalanceConfig(method="smote", sampling_strategy=0.1, k_neighbors=5)
    X_res, y_res = resample(X, y, config, seed=42)
    assert int(y_res.sum()) > int(y.sum())
    assert X_res.shape[0] == y_res.shape[0]


def test_resample_skips_when_minority_too_small():
    X = np.zeros((100, 4))
    y = np.zeros(100, dtype=int)
    y[0] = 1
    config = ImbalanceConfig(method="smote")
    X_res, y_res = resample(X, y, config, seed=0)
    assert X_res.shape[0] == 100
