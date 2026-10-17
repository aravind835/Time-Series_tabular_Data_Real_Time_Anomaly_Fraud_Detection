import sys
import warnings
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def small_config():
    from fraud_detection.config import Config

    config = Config()
    config.data.n_samples = 8000
    config.data.n_features = 12
    config.data.fraud_rate = 0.01
    config.models.autoencoder.epochs = 3
    config.models.isolation_forest.n_estimators = 60
    config.models.xgboost.n_estimators = 80
    return config


@pytest.fixture(scope="session")
def trained(tmp_path_factory, small_config):
    from fraud_detection.training import train_and_save

    config = small_config
    config.data.output_dir = str(tmp_path_factory.mktemp("data"))
    artifacts_dir = str(tmp_path_factory.mktemp("artifacts"))
    results = train_and_save(config, artifacts_dir=artifacts_dir)
    return {"config": config, "artifacts_dir": artifacts_dir, "results": results}
