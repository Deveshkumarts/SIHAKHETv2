"""
LSTM motion model for debris tracking.

Input  : the last `seq_len` observations of a track, each as
         [dx, dy, w, h, conf, dt, vx, vy, heading]   (positions relative to the most recent observation)
Output : predicted positions for the next `horizon` frames, relative to the most recent observation.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

FEATURES = 9
POS_SCALE = 100.0        # px -> ~unit range
SIZE_SCALE = 100.0


class TrackLSTM(nn.Module):
    def __init__(self, horizon: int = 5, hidden: int = 64, layers: int = 1, in_dim: int = FEATURES):
        super().__init__()
        self.horizon, self.hidden, self.layers, self.in_dim = horizon, hidden, layers, in_dim
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, horizon * 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # (B,T,F) -> (B,H,2), normalised units
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).view(-1, self.horizon, 2)


def make_features(history: Sequence[Dict[str, float]], seq_len: int) -> np.ndarray:
    """history: chronological observations {x,y,w,h,conf,t}. Returns (seq_len, FEATURES) float32.
    Short histories are padded by repeating the earliest observation (zero motion)."""
    obs = list(history)[-seq_len:]
    while len(obs) < seq_len:
        obs.insert(0, obs[0])
    last = obs[-1]
    feats = np.zeros((seq_len, FEATURES), np.float32)
    for i, o in enumerate(obs):
        prev = obs[i - 1] if i > 0 else o
        dt = max(o["t"] - prev["t"], 0.0)
        vx = (o["x"] - prev["x"]) / dt if dt > 0 else 0.0
        vy = (o["y"] - prev["y"]) / dt if dt > 0 else 0.0
        feats[i] = [(o["x"] - last["x"]) / POS_SCALE, (o["y"] - last["y"]) / POS_SCALE,
                    o["w"] / SIZE_SCALE, o["h"] / SIZE_SCALE, o["conf"], dt,
                    vx / POS_SCALE, vy / POS_SCALE, np.arctan2(vy, vx) / np.pi if (vx or vy) else 0.0]
    return feats


def constant_velocity_forecast(history: Sequence[Dict[str, float]], horizon: int, window: int = 3) -> np.ndarray:
    """Baseline: extrapolate mean velocity over the last `window` steps. Returns (horizon,2) relative displacements."""
    obs = list(history)[-(window + 1):]
    if len(obs) < 2:
        return np.zeros((horizon, 2), np.float32)
    dt = obs[-1]["t"] - obs[0]["t"]
    v = np.array([obs[-1]["x"] - obs[0]["x"], obs[-1]["y"] - obs[0]["y"]]) / dt if dt > 0 else np.zeros(2)
    step = (obs[-1]["t"] - obs[-2]["t"]) or 1.0
    return np.stack([v * step * (k + 1) for k in range(horizon)]).astype(np.float32)
