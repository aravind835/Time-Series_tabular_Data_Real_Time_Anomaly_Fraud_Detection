import numpy as np

from fraud_detection.config import Config
from fraud_detection.data import (
    FRAUD_LABEL,
    TIMESTAMP,
    feature_columns,
    generate_transactions,
)
from fraud_detection.data.loader import split_dataframe


def test_generator_shape_and_fraud_rate():
    frame = generate_transactions(20000, 16, 8, 0.001, 0.6, 42)
    assert len(frame) == 20000
    assert frame[FRAUD_LABEL].sum() == 20
    for column in feature_columns(16):
        assert column in frame.columns


def test_generator_is_deterministic():
    first = generate_transactions(5000, 12, 6, 0.01, 0.5, 7)
    second = generate_transactions(5000, 12, 6, 0.01, 0.5, 7)
    assert first.equals(second)


def test_fraud_signal_present():
    frame = generate_transactions(20000, 16, 8, 0.01, 0.6, 1)
    fraud_amount = frame.loc[frame[FRAUD_LABEL] == 1, "amount"].median()
    legit_amount = frame.loc[frame[FRAUD_LABEL] == 0, "amount"].median()
    assert fraud_amount > legit_amount


def test_time_ordered_split_is_monotonic_and_disjoint():
    config = Config()
    config.data.n_features = 16
    frame = generate_transactions(20000, 16, 8, 0.01, 0.6, 3)
    split = split_dataframe(frame, config)
    assert np.all(np.diff(split.train_frame[TIMESTAMP].to_numpy()) >= 0)
    train_max = split.train_frame[TIMESTAMP].max()
    test_min = split.test_frame[TIMESTAMP].min()
    assert train_max <= test_min
    total = len(split.y_train) + len(split.y_val) + len(split.y_test)
    assert total == len(frame)
