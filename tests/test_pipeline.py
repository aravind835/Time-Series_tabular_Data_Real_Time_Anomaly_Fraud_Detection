from pathlib import Path

from fraud_detection.artifacts import (
    METADATA_FILE,
    METRICS_FILE,
    THRESHOLDS_FILE,
    load_bundle,
)
from fraud_detection.config import load_config


def test_config_roundtrip(tmp_path):
    text = "seed: 7\ndata:\n  n_features: 5\nmodels:\n  xgboost:\n    max_depth: 3\n"
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    config = load_config(str(path))
    assert config.seed == 7
    assert config.data.n_features == 5
    assert config.models.xgboost.max_depth == 3


def test_training_produces_all_artifacts(trained):
    artifacts_dir = Path(trained["artifacts_dir"])
    for filename in (METADATA_FILE, METRICS_FILE, THRESHOLDS_FILE):
        assert (artifacts_dir / filename).exists()
    expected_features = trained["config"].data.n_features + 3
    for name in ("isolation_forest", "xgboost", "autoencoder"):
        preprocessor, model, threshold, metadata = load_bundle(artifacts_dir, name)
        assert 0.0 <= threshold <= 1.0
        assert metadata["n_features"] == expected_features
        assert model.name == name


def test_metrics_beat_random_baseline(trained):
    metrics = trained["results"]["metrics"]
    xgb = metrics["xgboost"]
    assert xgb["pr_auc"] > 0.1
    assert xgb["recall"] > 0.0
