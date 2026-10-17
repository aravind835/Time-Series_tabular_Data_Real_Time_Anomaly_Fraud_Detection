from typing import Any

from fraud_detection.training.imbalance import (
    class_weights,
    resample,
    scale_pos_weight,
)

__all__ = ["class_weights", "resample", "scale_pos_weight", "train_and_save"]


def __getattr__(name: str) -> Any:
    if name == "train_and_save":
        from fraud_detection.training.pipeline import train_and_save

        return train_and_save
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
