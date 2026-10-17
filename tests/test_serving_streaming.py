import numpy as np
from fastapi.testclient import TestClient

from fraud_detection.data import feature_columns, generate_transactions
from fraud_detection.serving.app import create_app
from fraud_detection.serving.predictor import Predictor
from fraud_detection.streaming import records_from_frame, run_stream


def _records(n_features):
    frame = generate_transactions(200, n_features, 6, 0.1, 0.6, 5)
    names = feature_columns(n_features)
    return frame[names].to_dict(orient="records"), frame["is_fraud"].tolist()


def test_predictor_single_and_batch(trained):
    predictor = Predictor(trained["artifacts_dir"], "xgboost")
    records, _ = _records(trained["config"].data.n_features)
    single = predictor.predict(records[0])
    assert 0.0 <= single["score"] <= 1.0
    assert isinstance(single["is_fraud"], bool)
    batch = predictor.predict_batch(records[:20])
    assert len(batch) == 20
    assert predictor.metrics()["latency"]["count"] == 21


def test_http_endpoints(trained):
    app = create_app(trained["artifacts_dir"], "xgboost")
    client = TestClient(app)
    records, _ = _records(trained["config"].data.n_features)
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/predict", json={"features": records[0]}).json()
    assert set(response.keys()) == {"model", "score", "threshold", "is_fraud", "latency_ms"}
    batch = client.post("/predict/batch", json={"transactions": records[:5]}).json()
    assert len(batch) == 5
    assert client.get("/metrics").json()["latency"]["count"] >= 6


def test_stream_counts_are_consistent(trained):
    predictor = Predictor(trained["artifacts_dir"], "xgboost")
    records, labels = _records(trained["config"].data.n_features)
    summary = run_stream(predictor, records, labels, rate_per_second=0, duration_seconds=None)
    assert summary.processed == len(records)
    assert summary.frauds == int(np.sum(labels))
    assert summary.true_positives + summary.false_negatives == summary.frauds
    assert summary.alerts == summary.true_positives + summary.false_positives
