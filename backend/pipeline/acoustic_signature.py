"""
Acoustic signature analysis (acoustic-impedance proxy) for object verification and confirmation.

What side-scan backscatter can and cannot tell you
--------------------------------------------------
Backscatter intensity from an object rises with the impedance contrast between it and the water/seabed
(reflection coefficient R = (Z2 - Z1) / (Z2 + Z1)), but also with surface roughness, orientation and grazing angle.
Uncalibrated 8-bit imagery therefore cannot give ABSOLUTE impedance. What this module computes are
RELATIVE, image-derived quantities that are physically meaningful and directly comparable between objects:

  impedance_contrast_idx  Michelson-style contrast (I_obj - I_bg) / (I_obj + I_bg) in [-1, 1] - the same algebraic form as
                          the reflection coefficient, evaluated on backscatter amplitude (relative, not absolute Z)
  contrast_db / peak_db   object mean / 95th-percentile amplitude vs the local seabed, in dB (20 log10)
  shadow_depth            1 - (darkest adjacent region / seabed); a tall, acoustically hard object shadows strongly
  highlight_shadow_ratio  object amplitude / shadow amplitude
  shadow_length_px        how far the shadow extends; with sonar geometry it gives object height (height_m)
  edge_sharpness, texture_cv, entropy, area, elongation, fill_ratio

These features feed a class-conditional model (scripts/train_signature.py) that answers: "is this acoustic signature
consistent with the class the detector claimed?"  It VERIFIES and CONFIRMS; it never vetoes an object on its own.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

FEATURE_NAMES = ["impedance_contrast_idx", "contrast_db", "peak_db", "shadow_depth", "highlight_shadow_ratio",
                 "shadow_length_rel", "edge_sharpness", "texture_cv", "entropy", "log_area", "elongation", "fill_ratio"]


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def box_mask(shape, bbox) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1
    return m.astype(bool)


def _shadow(gray: np.ndarray, bbox, bg: float, waterfall: bool, max_ratio: float = 2.0) -> Tuple[float, float, int]:
    """(darkest side mean / bg, side index, shadow length px). Waterfall: only the side away from the nadir counts;
    plain images have no nadir so all four sides are searched."""
    h, w = gray.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = max(2, x2 - x1), max(2, y2 - y1)
    lx, ly = int(bw * max_ratio), int(bh * max_ratio)
    sides = {"left": gray[y1:y2, max(0, x1 - lx):x1], "right": gray[y1:y2, x2:min(w, x2 + lx)],
             "up": gray[max(0, y1 - ly):y1, x1:x2], "down": gray[y2:min(h, y2 + ly), x1:x2]}
    if waterfall:
        keep = "left" if (x1 + x2) / 2 < w / 2 else "right"
        sides = {keep: sides[keep]}
    best_ratio, best_side, best_len = 1.0, 0, 0
    for i, (name, p) in enumerate(sides.items()):
        if p.size == 0:
            continue
        r = float(p.mean()) / max(1.0, bg)
        if r < best_ratio:
            best_ratio, best_side = r, i
            prof = p.mean(axis=0) if name in ("left", "right") else p.mean(axis=1)
            prof = prof[::-1] if name in ("left", "up") else prof          # order: away from the object
            dark = prof < 0.7 * bg
            best_len = int(np.argmin(dark)) if not dark.all() and dark.any() and dark[0] else (len(prof) if dark.all() else 0)
    return best_ratio, best_side, best_len


def extract_signature(image: np.ndarray, bbox, mask: Optional[np.ndarray] = None, waterfall: bool = False,
                      geometry: Optional[Dict[str, float]] = None) -> Optional[Dict[str, float]]:
    """Signature features for one object. `mask` (bool HxW) is the instance mask if available, else the box is used.
    geometry = {"altitude_m", "range_per_px_m"} enables the shadow-based height estimate (waterfalls only)."""
    g = _gray(image).astype(np.float32)
    h, w = g.shape
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    m = (mask.astype(bool) & box_mask(g.shape, (x1, y1, x2, y2))) if mask is not None and mask.any() else box_mask(g.shape, (x1, y1, x2, y2))
    if m.sum() < 6:
        m = box_mask(g.shape, (x1, y1, x2, y2))

    pad = max(4, int(0.6 * max(x2 - x1, y2 - y1)))
    outer = np.zeros_like(m, np.uint8)
    outer[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)] = 1
    ring = outer.astype(bool) & ~cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
    if ring.sum() < 8:
        ring = ~m
    bg = float(g[ring].mean())
    obj = g[m]
    mean_o, p95 = float(obj.mean()), float(np.percentile(obj, 95))
    eps = 1.0

    ratio, _, sh_len = _shadow(g, (x1, y1, x2, y2), bg, waterfall)
    shadow_amp = max(eps, ratio * bg)

    edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    edge_sharp = float(np.hypot(gx, gy)[edge].mean() / max(eps, bg)) if edge.any() else 0.0

    hist = np.bincount(obj.astype(np.uint8), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    ys, xs = np.where(m)
    cov = np.cov(np.vstack([xs, ys])) if len(xs) > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    size = max(x2 - x1, y2 - y1)

    f = {
        "impedance_contrast_idx": float((mean_o - bg) / (mean_o + bg + eps)),
        "contrast_db": float(20 * np.log10((mean_o + eps) / (bg + eps))),
        "peak_db": float(20 * np.log10((p95 + eps) / (bg + eps))),
        "shadow_depth": float(np.clip(1.0 - ratio, 0.0, 1.0)),
        "highlight_shadow_ratio": float(np.clip(mean_o / shadow_amp, 0.0, 50.0)),
        "shadow_length_rel": float(sh_len / max(1, size)),
        "edge_sharpness": edge_sharp,
        "texture_cv": float(obj.std() / max(eps, mean_o)),
        "entropy": float(-(p * np.log2(p)).sum()),
        "log_area": float(np.log1p(m.sum())),
        "elongation": float(np.sqrt(ev[0] / max(ev[1], 1e-6))),
        "fill_ratio": float(m.sum() / max(1, (x2 - x1) * (y2 - y1))),
    }
    if waterfall and geometry and sh_len > 0:
        # classic side-scan relation: h = L_shadow * H / (R_ground_to_shadow_end); small-angle form with slant range R
        r_m = geometry.get("range_to_object_m", 0.0)
        l_m = sh_len * geometry.get("range_per_px_m", 0.0)
        H = geometry.get("altitude_m", 0.0)
        if r_m > 0 and H > 0:
            f["height_m"] = float(l_m * H / (r_m + l_m))
    return f


def to_vector(f: Dict[str, float]) -> np.ndarray:
    return np.array([f[n] for n in FEATURE_NAMES], np.float64)


# ------------------------------------------------------------------ material families (assumption, stated)
# Assigned from the class NAMES by me; the dataset has no material labels. Used only to test whether the
# impedance-contrast physics is visible in the data, never as ground truth for the detector.
FAMILY = {
    "metal": ["can", "chain", "hook", "metal-bottle", "metal-box", "valve", "wrench", "propeller", "pipeline or cable"],
    "glass": ["glass-bottle", "brown-glass-bottle", "glass-jar", "potion-glass-bottle"],
    "plastic": ["plastic-bidon", "plastic-bottle", "plastic-pipe", "plastic-propeller", "drink-carton", "drink-sachet",
                "shampoo-bottle", "standing-bottle", "bottle"],
    "rubber": ["tire", "small-tire", "large-tire"],
    "structure": ["Shipwrecks", "rotating-platform"],
}
CLASS_TO_FAMILY = {c: fam for fam, cs in FAMILY.items() for c in cs}
