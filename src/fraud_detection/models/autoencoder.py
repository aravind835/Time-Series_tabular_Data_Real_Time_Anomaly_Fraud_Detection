from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fraud_detection.config import AutoencoderConfig
from fraud_detection.models.base import BaseModel, apply_normalizer, fit_normalizer
from fraud_detection.utils import ensure_dir


class _AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        encoder_layers: List[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            encoder_layers.append(nn.Linear(previous, width))
            encoder_layers.append(nn.ReLU())
            previous = width
        decoder_layers: List[nn.Module] = []
        for width in reversed(hidden_dims[:-1]):
            decoder_layers.append(nn.Linear(previous, width))
            decoder_layers.append(nn.ReLU())
            previous = width
        decoder_layers.append(nn.Linear(previous, input_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderModel(BaseModel):
    name = "autoencoder"
    supervised = False

    def __init__(self, config: AutoencoderConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self.input_dim: Optional[int] = None
        self.net: Optional[_AutoencoderNet] = None
        self.device = torch.device("cpu")
        self.low = 0.0
        self.high = 1.0

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AutoencoderModel":
        torch.manual_seed(self.seed)
        normal = X if y is None else X[y == 0]
        self.input_dim = int(normal.shape[1])
        self.net = _AutoencoderNet(self.input_dim, list(self.config.hidden_dims)).to(self.device)

        tensor = torch.tensor(normal, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.MSELoss()

        self.net.train()
        for _ in range(self.config.epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                reconstruction = self.net(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                optimizer.step()

        errors = self._reconstruction_error(normal)
        self.low, self.high = fit_normalizer(errors)
        return self

    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("model is not fitted")
        self.net.eval()
        tensor = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        with torch.no_grad():
            reconstruction = self.net(tensor)
            errors = torch.mean((reconstruction - tensor) ** 2, dim=1)
        return errors.cpu().numpy()

    def score(self, X: np.ndarray) -> np.ndarray:
        errors = self._reconstruction_error(np.asarray(X, dtype=np.float32))
        return apply_normalizer(errors, self.low, self.high)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        ensure_dir(target.parent)
        payload = {
            "state_dict": self.net.state_dict() if self.net is not None else None,
            "input_dim": self.input_dim,
            "hidden_dims": list(self.config.hidden_dims),
            "seed": self.seed,
            "low": self.low,
            "high": self.high,
            "config": self.config,
        }
        torch.save(payload, target)

    @classmethod
    def load(cls, path: str | Path) -> "AutoencoderModel":
        payload = torch.load(path, weights_only=False)
        model = cls(payload["config"], payload["seed"])
        model.input_dim = payload["input_dim"]
        model.net = _AutoencoderNet(payload["input_dim"], payload["hidden_dims"])
        model.net.load_state_dict(payload["state_dict"])
        model.net.to(model.device)
        model.low = payload["low"]
        model.high = payload["high"]
        return model
