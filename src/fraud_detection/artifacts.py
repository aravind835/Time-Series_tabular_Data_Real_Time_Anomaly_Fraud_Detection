from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

from fraud_detection.features import Preprocessor
from fraud_detection.utils import read_json

if TYPE_CHECKING:
    from fraud_detection.models import BaseModel

MODEL_FILENAMES: Dict[str, str] = {
    "isolation_forest": "isolation_forest.joblib",
    "xgboost": "xgboost.joblib",
    "autoencoder": "autoencoder.pt",
}
PREPROCESSOR_FILE = "preprocessor.joblib"
THRESHOLDS_FILE = "thresholds.json"
METADATA_FILE = "metadata.json"
METRICS_FILE = "metrics.json"


@dataclass
class ArtifactPaths:
    root: Path

    @property
    def preprocessor(self) -> Path:
        return self.root / PREPROCESSOR_FILE

    @property
    def thresholds(self) -> Path:
        return self.root / THRESHOLDS_FILE

    @property
    def metadata(self) -> Path:
        return self.root / METADATA_FILE

    @property
    def metrics(self) -> Path:
        return self.root / METRICS_FILE

    def model(self, name: str) -> Path:
        return self.root / MODEL_FILENAMES[name]


def load_model(artifacts_dir: str | Path, name: str) -> "BaseModel":
    from fraud_detection.models import MODEL_REGISTRY

    paths = ArtifactPaths(Path(artifacts_dir))
    return MODEL_REGISTRY[name].load(paths.model(name))


def load_bundle(
    artifacts_dir: str | Path, name: str
) -> Tuple[Preprocessor, "BaseModel", float, Dict]:
    paths = ArtifactPaths(Path(artifacts_dir))
    preprocessor = Preprocessor.load(paths.preprocessor)
    model = load_model(artifacts_dir, name)
    thresholds = read_json(paths.thresholds)
    metadata = read_json(paths.metadata)
    threshold = float(thresholds[name]["threshold"])
    return preprocessor, model, threshold, metadata
