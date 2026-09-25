"""
LSTM debris tracker: Hungarian association on LSTM-predicted positions, persistent IDs,
future-position prediction and trajectories. Falls back to a constant-velocity predictor when no
LSTM weights are available (and says so via `predictor_name`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from backend.pipeline.base import file_version
from models.lstm.model import TrackLSTM, constant_velocity_forecast, make_features


@dataclass
class Track:
    id: int
    class_name: str
    history: List[Dict[str, float]] = field(default_factory=list)
    hits: int = 1
    misses: int = 0
    last_t: float = 0.0
    payload: Optional[Any] = None


class LSTMTracker:
    def __init__(self, weights: str | Path | None = None, seq_len: int = 8, horizon: int = 5, max_age: int = 5,
                 match_dist_px: float = 60.0, min_hits: int = 2, device: str = "cpu", predictor: str = "auto",
                 onnx: str | Path | None = None):
        self.seq_len, self.horizon, self.max_age = seq_len, horizon, max_age
        self.match_dist_px, self.min_hits = match_dist_px, min_hits
        self.device = torch.device(device)
        self.model: Optional[TrackLSTM] = None
        self.version = file_version(weights, "lstm-tracker")
        if predictor != "cv" and weights and Path(weights).exists():
            ckpt = torch.load(str(weights), map_location=self.device, weights_only=True)
            self.seq_len, self.horizon = int(ckpt["seq_len"]), int(ckpt["horizon"])
            self.model = TrackLSTM(self.horizon, int(ckpt["hidden"])).to(self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
        self.session = None
        if self.model is not None and onnx and Path(onnx).exists():
            from backend.pipeline.onnx_backend import make_session
            self.session = make_session(onnx)
        self.tracks: List[Track] = []
        self._next_id = 1
        self.frame_index = 0

    @property
    def predictor_name(self) -> str:
        return "lstm" if self.model is not None else "constant_velocity"

    def reset(self) -> None:
        self.tracks, self._next_id, self.frame_index = [], 1, 0

    # ---- prediction ----
    def forecast(self, tracks: List[Track]) -> np.ndarray:
        """(N, horizon, 2) predicted displacement from each track's latest observation, in px."""
        if not tracks:
            return np.zeros((0, self.horizon, 2), np.float32)
        if self.model is None:
            return np.stack([constant_velocity_forecast(t.history, self.horizon) for t in tracks])
        feats = np.stack([make_features(t.history, self.seq_len) for t in tracks])
        if self.session is not None:
            return self.session.run(None, {self.session.get_inputs()[0].name: feats})[0] * 100.0
        x = torch.from_numpy(feats).to(self.device)
        with torch.no_grad():
            return self.model(x).cpu().numpy() * 100.0

    # ---- one frame ----
    def update(self, detections: List[Dict[str, Any]], t: Optional[float] = None) -> List[Dict[str, Any]]:
        t = float(self.frame_index if t is None else t)
        self.frame_index += 1
        fc = self.forecast(self.tracks)
        pred_now = []
        for tr, f in zip(self.tracks, fc):
            k = min(tr.misses, self.horizon - 1)                    # frames since last observation -> horizon step
            last = tr.history[-1]
            pred_now.append((last["x"] + f[k, 0], last["y"] + f[k, 1]))

        matched_t, matched_d = set(), set()
        if self.tracks and detections:
            cost = np.full((len(self.tracks), len(detections)), 1e6)
            for i, (px, py) in enumerate(pred_now):
                for j, d in enumerate(detections):
                    dist = float(np.hypot(px - d["x"], py - d["y"]))
                    if dist <= self.match_dist_px:
                        same = d.get("class_name") == self.tracks[i].class_name or "Unknown" in str(d.get("class_name"))
                        cost[i, j] = dist + (0.0 if same else 15.0)
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] < 1e5:
                    tr, d = self.tracks[i], detections[j]
                    tr.history.append({"x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"], "conf": d["conf"], "t": t})
                    tr.history = tr.history[-(self.seq_len + self.horizon):]
                    tr.hits += 1
                    tr.misses = 0
                    tr.last_t = t
                    tr.payload = d.get("payload")
                    matched_t.add(i)
                    matched_d.add(j)
                    d["track_id"] = tr.id

        for i, tr in enumerate(self.tracks):
            if i not in matched_t:
                tr.misses += 1
        for j, d in enumerate(detections):
            if j not in matched_d:
                tr = Track(self._next_id, d.get("class_name", "object"),
                           [{"x": d["x"], "y": d["y"], "w": d["w"], "h": d["h"], "conf": d["conf"], "t": t}], last_t=t,
                           payload=d.get("payload"))
                d["track_id"] = tr.id
                self._next_id += 1
                self.tracks.append(tr)
        self.tracks = [tr for tr in self.tracks if tr.misses <= self.max_age]
        return self.snapshot()

    def snapshot(self) -> List[Dict[str, Any]]:
        fc = self.forecast(self.tracks)
        out = []
        for tr, f in zip(self.tracks, fc):
            last = tr.history[-1]
            v = (0.0, 0.0)
            if len(tr.history) >= 2:
                a, b = tr.history[-2], tr.history[-1]
                dt = (b["t"] - a["t"]) or 1.0
                v = ((b["x"] - a["x"]) / dt, (b["y"] - a["y"]) / dt)
            out.append({
                "track_id": tr.id, "class_name": tr.class_name, "x": float(last["x"]), "y": float(last["y"]), "w": float(last["w"]),
                "h": float(last["h"]), "confidence": float(last["conf"]), "state": "confirmed" if tr.hits >= self.min_hits else "tentative",
                "hits": tr.hits, "misses": tr.misses, "velocity": [float(v[0]), float(v[1])],
                "trajectory": [[float(o["x"]), float(o["y"])] for o in tr.history],
                "predicted": [[float(last["x"] + d[0]), float(last["y"] + d[1])] for d in f],
                "predictor": self.predictor_name, "payload": tr.payload,
            })
        return out
