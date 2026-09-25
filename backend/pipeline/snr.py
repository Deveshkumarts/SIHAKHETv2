"""Stage 5 - SNR analysis: global SNR (dB) plus a per-tile quality map used to weight evidence downstream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

from utils.sonar_calibration import QualityMetrics, compute_snr_index


@dataclass
class SNRResult:
    metrics: QualityMetrics
    tile_snr_db: np.ndarray            # (rows, cols) SNR per tile
    tile_size: int

    @property
    def snr_db(self) -> float:
        return float(self.metrics.snr_db)

    def tile_quality_at(self, x: float, y: float) -> float:
        """Tile SNR (dB) at a pixel position."""
        r = int(np.clip(y // self.tile_size, 0, self.tile_snr_db.shape[0] - 1))
        c = int(np.clip(x // self.tile_size, 0, self.tile_snr_db.shape[1] - 1))
        return float(self.tile_snr_db[r, c])

    def summary(self) -> Dict[str, float]:
        return {"snr_db": round(self.snr_db, 2), "dynamic_range_db": round(float(self.metrics.dynamic_range_db), 2),
                "tile_snr_min": round(float(self.tile_snr_db.min()), 2),
                "tile_snr_mean": round(float(self.tile_snr_db.mean()), 2)}


def run_snr(image_bgr: np.ndarray, cfg) -> SNRResult:
    tile = int(cfg.snr.tile_size)
    metrics = compute_snr_index(image_bgr)
    h, w = image_bgr.shape[:2]
    rows, cols = max(1, -(-h // tile)), max(1, -(-w // tile))
    tiles = np.zeros((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            patch = image_bgr[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
            tiles[r, c] = compute_snr_index(patch).snr_db if patch.size else 0.0
    return SNRResult(metrics=metrics, tile_snr_db=tiles, tile_size=tile)
