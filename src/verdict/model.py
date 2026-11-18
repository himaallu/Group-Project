"""Verdict head: a small MLP over the features from ``features.py`` (H3, CLAUDE.md §3).

What it does
    ``VerdictHead``: input -> 64 -> 3 with ReLU and dropout 0.2. Outputs logits in
    the order ``supported, contradicted, unverified`` (= LABELS). Features are
    standardised with the training mean / sd, which are saved with the weights.
    ``save_head`` / ``load_head`` write and read ``models/<name>/model.pt`` +
    ``config.json``; ``predict`` turns a feature table into logits.

Inputs / outputs
    Feature DataFrame (columns ``config["features"]``) -> numpy logits (n, 3).

How to run
    from src.verdict.model import load_head, predict
    model, config = load_head("models/verdict_v0_s42")
    logits = predict(model, config, feature_table)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HIDDEN = 64
DROPOUT = 0.2


class VerdictHead(nn.Module):
    """MLP: n_features -> 64 -> 3 (ReLU, dropout 0.2)."""

    def __init__(self, n_features: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        """Build the two linear layers."""
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (batch, 3) for standardised features (batch, n_features)."""
        return self.net(x)


def standardise(table: pd.DataFrame, features: list[str], mean: list[float],
                sd: list[float]) -> torch.Tensor:
    """(x - mean) / sd as a float tensor, columns in ``features`` order."""
    x = table[features].to_numpy(dtype=np.float32)
    return torch.from_numpy((x - np.asarray(mean, np.float32)) / np.asarray(sd, np.float32))


def fit_scaler(table: pd.DataFrame, features: list[str]) -> tuple[list[float], list[float]]:
    """Training-set mean and sd per feature (sd 1.0 for constant features)."""
    x = table[features].to_numpy(dtype=float)
    sd = x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return x.mean(axis=0).tolist(), sd.tolist()


def predict(model: VerdictHead, config: dict, table: pd.DataFrame) -> np.ndarray:
    """Logits (n, 3) for a feature table, with dropout off."""
    model.eval()
    with torch.no_grad():
        x = standardise(table, config["features"], config["mean"], config["sd"])
        return model(x).numpy()


def save_head(model: VerdictHead, config: dict, out_dir: str | Path) -> None:
    """Write ``model.pt`` (weights) and ``config.json`` (features, scaler, settings)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def load_head(model_dir: str | Path) -> tuple[VerdictHead, dict]:
    """Read a head saved by ``save_head``; returns (model in eval mode, config)."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model = VerdictHead(len(config["features"]))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    return model.eval(), config
