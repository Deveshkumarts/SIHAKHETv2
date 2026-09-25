"""
Stage 9 - decision / fusion layer:  KNOWN DEBRIS | UNKNOWN ANOMALY | NOISE.

Evidence per object (each in [0,1], absent evidence is simply left out and the weights renormalised):
  cfar     - SA-CFAR candidate contrast
  yolo     - YOLO11-Seg confidence
  anomaly  - autoencoder anomaly score (0.5 == the validation-selected threshold)
  snr      - SNR / tile-quality reliability
  shadow   - optional acoustic-shadow consistency (weight 0 by default)

Rules (in order):
  KNOWN    : YOLO present and conf >= known_min_yolo_conf
  UNKNOWN  : autoencoder flags an anomaly (score >= unknown_min_anomaly_score), OR the combined evidence is
             above noise_max_fused
  NOISE    : combined evidence is weak
An object is never rejected solely because its anomaly score is low.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from utils.decision_gate import compute_box_iou, verify_acoustic_shadow

KNOWN, UNKNOWN, NOISE = "KNOWN_DEBRIS", "UNKNOWN_ANOMALY", "NOISE"


@dataclass
class FusedObject:
    category: str
    class_name: str
    detection_type: str                     # 'known' | 'unknown' | 'noise'
    confidence: float                       # fused confidence [0,1]
    bbox: List[float]
    evidence: Dict[str, Optional[float]]
    reasons: List[str] = field(default_factory=list)
    yolo: Optional[Dict[str, Any]] = None
    cfar: Optional[Dict[str, Any]] = None
    anomaly: Optional[Dict[str, Any]] = None
    tile_snr_db: Optional[float] = None
    has_shadow: Optional[bool] = None
    centroid: Optional[List[float]] = None
    mask_area_px: int = 0
    mask_polygon: List[List[float]] = field(default_factory=list)
    # filled by later stages (tracking / geolocation)
    features: Dict[str, float] = field(default_factory=dict)
    signature: Optional[Dict[str, Any]] = None              # acoustic signature + verdict (backend/pipeline/signature_verifier.py)
    frame_id: int = 0
    track_id: Optional[int] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    uncertainty_m: Optional[float] = None
    uncertainty_minor_m: Optional[float] = None
    ground_range_m: Optional[float] = None
    channel: Optional[str] = None
    timestamp: Optional[float] = None


def shadow_any_direction(image_bgr: np.ndarray, bbox, search_ratio: float = 1.8, min_drop: float = 0.70):
    """Shadow check for inputs WITHOUT waterfall geometry (plain images / cropped objects): there is no nadir, so
    the shadow can fall on any side. Returns (has_shadow, darkest side-patch / ambient ratio)."""
    import cv2
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    h, w = gray.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = max(2, x2 - x1), max(2, y2 - y1)
    pad = max(4, int(0.5 * max(bw, bh)))
    amb = gray[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
    ambient = float(amb.mean()) if amb.size else 100.0
    lx, ly = int(bw * search_ratio), int(bh * search_ratio)
    sides = [gray[y1:y2, max(0, x1 - lx):x1], gray[y1:y2, x2:min(w, x2 + lx)],
             gray[max(0, y1 - ly):y1, x1:x2], gray[y2:min(h, y2 + ly), x1:x2]]
    ratios = [float(p.mean()) / max(1.0, ambient) for p in sides if p.size]
    if not ratios:
        return False, 1.0
    r = min(ratios)
    return r <= min_drop, round(r, 3)


FEATURE_NAMES = ["cfar", "anomaly", "recon_ratio", "snr", "shadow", "log_area", "contrast"]


class LearnedFusion:
    """Logistic model over the evidence features. Trained on real labelled frames by scripts/train_fusion.py;
    coefficients, scaling and the validated decision threshold live in a small JSON file."""

    def __init__(self, path):
        import json
        d = json.loads(open(path, encoding="utf-8").read())
        self.names, self.mean, self.std = d["features"], np.array(d["mean"]), np.array(d["std"])
        self.coef, self.intercept, self.threshold = np.array(d["coef"]), float(d["intercept"]), float(d["threshold"])
        self.meta = {k: d.get(k) for k in ("trained_on", "n_candidates", "n_positive", "cv")}

    def predict(self, feats: Dict[str, float]) -> float:
        x = (np.array([feats.get(n, 0.0) for n in self.names]) - self.mean) / np.where(self.std > 0, self.std, 1.0)
        return float(1.0 / (1.0 + np.exp(-(float(x @ self.coef) + self.intercept))))


_LEARNED_CACHE: Dict[str, Optional[LearnedFusion]] = {}


def load_learned_fusion(cfg) -> Optional[LearnedFusion]:
    rel = cfg.get("fusion.learned_model")
    if not rel:
        return None
    path = cfg.path("fusion.learned_model")
    key = str(path)
    if key not in _LEARNED_CACHE:
        _LEARNED_CACHE[key] = LearnedFusion(path) if path.exists() else None
    return _LEARNED_CACHE[key]


def context_crop(image_bgr: np.ndarray, bbox, pad: int, context_px: int) -> np.ndarray:
    """Square context crop centred on the object, side = max(context_px, object + 2*pad), slid (not squeezed) to stay
    inside the image so aspect is preserved. context_px=0 -> tight box + pad. Tiny CFAR specks judged in a tight
    crop are just blur; the autoencoder was trained on object-scale crops, so it needs object-scale context."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if context_px <= 0:
        return image_bgr[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
    side = min(max(context_px, max(x2 - x1, y2 - y1) + 2 * pad), min(h, w))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    return image_bgr[y0:y0 + side, x0:x0 + side]


def _center(b):
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _inside(pt, b) -> bool:
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def _match(a, b) -> bool:
    return compute_box_iou(a, b) >= 0.05 or _inside(_center(a), b) or _inside(_center(b), a)


def cfar_evidence(cand: Dict[str, Any]) -> float:
    return float(np.clip((float(cand.get("contrast_ratio", 1.0)) - 1.0) / 3.0, 0.0, 1.0))


def snr_evidence(tile_snr_db: Optional[float], floor: float = 3.0, ceil: float = 20.0) -> Optional[float]:
    if tile_snr_db is None:
        return None
    return float(np.clip((tile_snr_db - floor) / max(1e-6, ceil - floor), 0.0, 1.0))


def fuse_evidence(evidence: Dict[str, Optional[float]], weights: Dict[str, float]) -> float:
    num = den = 0.0
    for k, v in evidence.items():
        w = float(weights.get(k, 0.0))
        if v is None or w <= 0:
            continue
        num += w * v
        den += w
    return float(num / den) if den > 0 else 0.0


def run_fusion(image_bgr: np.ndarray, yolo_dets: List[Dict[str, Any]], cfar_cands: List[Dict[str, Any]],
               anomaly_scorer, snr_result, cfg, waterfall: bool = True, ae_image: Optional[np.ndarray] = None) -> List[FusedObject]:
    f = cfg.fusion
    weights = {"cfar": f.w_cfar, "yolo": f.w_yolo, "anomaly": f.w_anomaly, "snr": f.w_snr, "shadow": f.w_shadow}
    pad = int(cfg.sa_cfar.candidate_pad_px)
    learned = load_learned_fusion(cfg)

    # 1. object set: every YOLO detection, plus every CFAR candidate that no YOLO detection explains
    objects: List[Dict[str, Any]] = []
    used_cfar = set()
    for y in yolo_dets:
        hit = None
        for i, c in enumerate(cfar_cands):
            if _match(y["bbox"], c["bbox"]):
                hit = i if hit is None or compute_box_iou(y["bbox"], c["bbox"]) > compute_box_iou(y["bbox"], cfar_cands[hit]["bbox"]) else hit
        if hit is not None:
            used_cfar.add(hit)
        objects.append({"bbox": y["bbox"], "yolo": y, "cfar": cfar_cands[hit] if hit is not None else None})
    for i, c in enumerate(cfar_cands):
        if i not in used_cfar:
            objects.append({"bbox": [float(v) for v in c["bbox"]], "yolo": None, "cfar": c})

    # 2. autoencoder on every object ROI (batched)
    anomalies: List[Dict[str, Any]] = [{"available": False}] * len(objects)
    if objects and anomaly_scorer is not None and anomaly_scorer.available:
        ctx = int(cfg.get("autoencoder.context_px", 0))
        src = ae_image if ae_image is not None else image_bgr        # same geometry, possibly the un-enhanced image
        rois = [context_crop(src, o["bbox"], pad, ctx) for o in objects]
        keep = [i for i, r in enumerate(rois) if r.size > 0]
        scored = anomaly_scorer.score_rois([rois[i] for i in keep])
        for i, s in zip(keep, scored):
            anomalies[i] = s

    # 3. evidence -> fused confidence -> category
    fused: List[FusedObject] = []
    for o, an in zip(objects, anomalies):
        y, c = o["yolo"], o["cfar"]
        cx, cy = (y["centroid"] if y else None) or _center(o["bbox"])
        tile_snr = float(snr_result.tile_quality_at(cx, cy)) if snr_result is not None else None
        check = verify_acoustic_shadow if waterfall else shadow_any_direction
        has_shadow, shadow_contrast = check(image_bgr, [int(v) for v in o["bbox"]])
        ev = {
            "cfar": cfar_evidence(c) if c else None,
            "yolo": float(y["conf"]) if y else None,
            "anomaly": float(an["anomaly_score"]) if an.get("available") else None,
            "snr": snr_evidence(tile_snr, cfg.snr.min_tile_snr_db),
            # shadow_contrast is shadow-intensity / ambient: LOWER = darker = stronger shadow, so invert it
            "shadow": float(np.clip(1.0 - shadow_contrast, 0, 1)) if (has_shadow and weights["shadow"] > 0) else
                      (0.0 if weights["shadow"] > 0 else None),
        }
        score = fuse_evidence(ev, weights)
        area = float((c or {}).get("area", 0) or (y or {}).get("mask_area_px", 0) or 1)
        thr_anom = float((an or {}).get("threshold") or 0.0)
        feats = {"cfar": ev["cfar"] or 0.0, "anomaly": ev["anomaly"] or 0.0,
                 "recon_ratio": float(np.log1p((an.get("reconstruction_error", 0.0) / thr_anom) if thr_anom > 0 else 0.0)),
                 "snr": ev["snr"] or 0.0, "shadow": ev["shadow"] or 0.0, "log_area": float(np.log1p(area)),
                 "contrast": float((c or {}).get("contrast_ratio", 1.0))}
        reasons: List[str] = []
        noise_thr = f.noise_max_fused
        if learned is not None and not y:                       # learned model scores the unknown/CFAR branch
            score, noise_thr = learned.predict(feats), learned.threshold
            ev["learned"] = round(score, 4)
            reasons.append(f"learned fusion p={score:.2f} (validated threshold {noise_thr:.2f})")

        if y and y["conf"] >= f.known_min_yolo_conf:
            cat, name, dtype = KNOWN, y["class_name"], "known"
            reasons.append(f"YOLO11-Seg {y['class_name']} conf {y['conf']:.2f}")
        elif an.get("available") and an["anomaly_score"] >= f.unknown_min_anomaly_score:
            cat, name, dtype = UNKNOWN, "Unknown Anomaly", "unknown"
            reasons.append(f"autoencoder error {an['reconstruction_error']:.5f} > validated threshold {an['threshold']:.5f}")
        elif score >= noise_thr:
            cat, name, dtype = UNKNOWN, "Unknown Anomaly", "unknown"
            reasons.append(f"combined evidence {score:.2f} >= {noise_thr:.2f} (no single decisive cue)")
        else:
            cat, name, dtype = NOISE, "Noise", "noise"
            reasons.append(f"combined evidence {score:.2f} < {noise_thr:.2f}")
        if c:
            reasons.append(f"SA-CFAR candidate ({c.get('regime_name', 'seabed')})")
        if an.get("available") and an["anomaly_score"] < f.unknown_min_anomaly_score and cat != NOISE:
            reasons.append("low anomaly score did not veto the object")

        fused.append(FusedObject(
            category=cat, class_name=name, detection_type=dtype, confidence=round(score, 4),
            bbox=[float(v) for v in o["bbox"]], evidence={k: (round(v, 4) if v is not None else None) for k, v in ev.items()},
            reasons=reasons, yolo=y, cfar=c, anomaly=an if an.get("available") else None, tile_snr_db=tile_snr, features=feats,
            has_shadow=bool(has_shadow), centroid=[float(cx), float(cy)],
            mask_area_px=int(y["mask_area_px"]) if y else 0,
            mask_polygon=y["mask_polygon"] if y else [],
        ))
    return fused
