"""Stage 4 - enhancement chain exactly as specified: Median -> Bilateral -> CLAHE (each timed separately)."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from utils.sonar_preprocess import apply_bilateral_denoise, apply_median_filter
from backend.pipeline.base import RunLog


def apply_clahe_luminance(image_bgr: np.ndarray, clip: float, tile_px: int = 48) -> np.ndarray:
    """CLAHE on the LAB luminance channel with a size-aware tile grid (near-global on tiny crops)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    h, w = l.shape
    grid = (max(2, min(12, w // tile_px)), max(2, min(12, h // tile_px)))
    l2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def run_preprocessing(image_bgr: np.ndarray, cfg, runlog: Optional[RunLog] = None) -> np.ndarray:
    p = cfg.preprocessing
    stages = [
        ("median", lambda im: apply_median_filter(im, ksize=p.median_ksize)),
        ("bilateral", lambda im: apply_bilateral_denoise(im, d=p.bilateral_d, sigma_color=p.bilateral_sigma,
                                                         sigma_space=p.bilateral_sigma)),
        ("clahe", lambda im: apply_clahe_luminance(im, clip=p.clahe_clip)),
    ]
    out = image_bgr
    for name, fn in stages:
        if runlog is not None:
            with runlog.stage(name) as s:
                out = fn(out)
                s["mean_intensity"] = round(float(out.mean()), 2)
        else:
            out = fn(out)
    return out
