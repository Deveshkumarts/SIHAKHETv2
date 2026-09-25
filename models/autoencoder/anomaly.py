"""
Autoencoder anomaly scoring at inference:  ROI -> encoder -> latent -> decoder -> reconstruction
-> reconstruction error -> anomaly score -> threshold -> normal / unknown-anomaly.

The threshold is NOT hard-coded: it is loaded from the calibration file written by train.py,
which selects it on held-out validation data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from backend.pipeline.base import file_version
from models.autoencoder.conv_ae import PatchConvAE

SCORE_SHARPNESS = 3.0


def error_to_score(err: float, threshold: float, k: float = SCORE_SHARPNESS) -> float:
    """Map reconstruction error to (0,1) with score == 0.5 exactly at the validated threshold."""
    if threshold <= 0:
        return 0.0
    return float(1.0 / (1.0 + (threshold / max(err, 1e-12)) ** k))


@torch.no_grad()
def reconstruct(model: PatchConvAE, patches: np.ndarray, device, batch: int = 256):
    """patches: (N,1,P,P) float32 in [0,1] -> (recon, mse[N], mae[N])."""
    recons = []
    for i in range(0, len(patches), batch):
        x = torch.from_numpy(patches[i:i + batch]).to(device)
        recons.append(model(x).cpu().numpy())
    recon = np.concatenate(recons) if recons else np.zeros_like(patches)
    diff = patches - recon
    return recon, (diff ** 2).reshape(len(patches), -1).mean(1), np.abs(diff).reshape(len(patches), -1).mean(1)


class AnomalyScorer:
    def __init__(self, weights: str | Path, calibration: str | Path, device: str = "auto", onnx: str | Path | None = None):
        self.weights, self.calibration_path = Path(weights), Path(calibration)
        self.session = None                                  # ONNX Runtime session when an .onnx model is supplied
        self.backend = "pytorch"
        self.device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else
                                   ("cpu" if device == "auto" else device))
        self.model: Optional[PatchConvAE] = None
        self.calibration: Dict[str, Any] = {}
        self.threshold = 0.0
        self.patch = 64
        if self.weights.exists() and self.calibration_path.exists():
            self.calibration = json.loads(self.calibration_path.read_text())
            ckpt = torch.load(str(self.weights), map_location=self.device, weights_only=True)
            self.patch, latent = int(ckpt["patch"]), int(ckpt["latent"])
            self.model = PatchConvAE(self.patch, latent).to(self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.threshold = float(self.calibration["threshold"])
            if onnx and Path(onnx).exists():
                from backend.pipeline.onnx_backend import make_session
                self.session, self.backend = make_session(onnx), "onnx"
        self.version = file_version(self.weights, "conv-ae")

    @property
    def available(self) -> bool:
        return self.model is not None

    def _prep(self, roi: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        s = self.patch
        g = cv2.resize(g, (s, s), interpolation=cv2.INTER_AREA if min(g.shape) >= s else cv2.INTER_CUBIC)
        return g.astype(np.float32) / 255.0

    def score_rois(self, rois: List[np.ndarray]) -> List[Dict[str, Any]]:
        if not self.available or not rois:
            return [{"available": False} for _ in rois]
        arr = np.stack([self._prep(r) for r in rois])[:, None]
        if self.session is not None:
            recon = self.session.run(None, {self.session.get_inputs()[0].name: arr})[0]
            diff = arr - recon
            mse, mae = (diff ** 2).reshape(len(arr), -1).mean(1), np.abs(diff).reshape(len(arr), -1).mean(1)
        else:
            recon, mse, mae = reconstruct(self.model, arr, self.device)
        return [{"available": True, "reconstruction_error": float(m), "reconstruction_mae": float(a),
                 "anomaly_score": error_to_score(float(m), self.threshold), "threshold": self.threshold,
                 "is_anomaly": bool(m > self.threshold)} for m, a in zip(mse, mae)]

    def score_box(self, image_bgr: np.ndarray, bbox, pad: int = 8) -> Dict[str, Any]:
        h, w = image_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        roi = image_bgr[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
        if roi.size == 0:
            return {"available": False}
        return self.score_rois([roi])[0]
