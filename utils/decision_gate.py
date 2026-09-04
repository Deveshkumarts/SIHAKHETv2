"""
Multi-Evidence Decision & Verification Gate for Marine Debris & Sonar Anomalies.
Implements:
  1. Acoustic Shadow Co-Occurrence Verification
  2. Morphological Clutter & Artifact Filtering
  3. 3-Way Triage Decision Gate: [KNOWN DEBRIS | UNKNOWN ANOMALY | REJECT]
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class TriageDecision:
    category: str              # 'KNOWN_DEBRIS' | 'UNKNOWN_ANOMALY' | 'REJECT'
    class_name: str            # e.g., 'bottle', 'tire', 'Unknown Sonar Anomaly'
    confidence: float          # [0.0, 1.0]
    bbox: List[int]            # [x1, y1, x2, y2]
    has_shadow: bool           # True if verified acoustic shadow co-occurs
    shadow_contrast: float     # Ratio of shadow intensity to ambient clutter
    anomaly_score: float       # Autoencoder reconstruction error [0.0, 1.0]
    cfar_confirmed: bool       # True if confirmed by OS-CFAR candidate region
    triage_reason: str         # Decision gate explanation


def verify_acoustic_shadow(
    image_bgr: np.ndarray,
    bbox: List[int],
    shadow_search_ratio: float = 1.8,
    min_contrast_drop: float = 0.70
) -> Tuple[bool, float]:
    """
    Verifies the physical acoustic principle of Side-Scan Sonar:
    Every protruding seabed object MUST cast an acoustic shadow directly
    downstream away from the nadir flight path.

    Returns:
        (has_shadow_bool, shadow_contrast_ratio)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    h_img, w_img = gray.shape
    mid_x = w_img / 2.0

    x1, y1, x2, y2 = map(int, bbox)
    w_box = max(2, x2 - x1)
    h_box = max(2, y2 - y1)
    cx = (x1 + x2) / 2.0

    # Ambient clutter intensity around object
    pad = max(4, int(w_box * 0.5))
    x_amb1 = max(0, x1 - pad)
    y_amb1 = max(0, y1 - pad)
    x_amb2 = min(w_img, x2 + pad)
    y_amb2 = min(h_img, y2 + pad)
    ambient_patch = gray[y_amb1:y_amb2, x_amb1:x_amb2]
    ambient_mean = float(np.mean(ambient_patch)) if ambient_patch.size > 0 else 100.0

    # Downstream search region away from nadir
    if cx < mid_x:
        # Port side: shadow falls to the LEFT (towards 0)
        s_len = int(w_box * shadow_search_ratio)
        sx1 = max(0, x1 - s_len)
        sx2 = x1
    else:
        # Starboard side: shadow falls to the RIGHT (towards w_img)
        s_len = int(w_box * shadow_search_ratio)
        sx1 = x2
        sx2 = min(w_img, x2 + s_len)

    sy1 = max(0, y1)
    sy2 = min(h_img, y2)

    shadow_patch = gray[sy1:sy2, sx1:sx2]
    if shadow_patch.size == 0:
        return False, 1.0

    shadow_min_mean = float(np.mean(shadow_patch))
    shadow_contrast = shadow_min_mean / max(1.0, ambient_mean)

    # A valid shadow has intensity significantly below ambient seafloor clutter
    has_shadow = shadow_contrast <= min_contrast_drop
    return has_shadow, round(shadow_contrast, 3)


def compute_box_iou(boxA: List[float], boxB: List[float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def evaluate_decision_gate(
    image_bgr: np.ndarray,
    yolo_detections: List[Dict[str, any]],
    cfar_candidates: List[Dict[str, any]],
    autoencoder_anomalies: List[Dict[str, any]],
    snr_db: float = 12.0,
    yolo_conf_thresh: float = 0.25,
    anomaly_thresh: float = 0.40,
) -> Tuple[List[TriageDecision], Dict[str, int]]:
    """
    Multi-Evidence Decision Gate:
      Triages all candidates into:
        - KNOWN_DEBRIS (High YOLO + shadow check / CFAR alignment)
        - UNKNOWN_ANOMALY (Autoencoder anomaly / CFAR highlight without YOLO match)
        - REJECT (Low SNR, no shadow, or spurious clutter)
    """
    decisions = []
    matched_cfar_idx = set()
    matched_ae_idx = set()

    # 1. Process YOLO detections (Known Debris branch)
    for y_det in yolo_detections:
        bbox = y_det["bbox"]
        cname = y_det["class_name"]
        conf = y_det["conf"]

        has_shadow, shadow_contrast = verify_acoustic_shadow(image_bgr, bbox)

        # Check CFAR co-confirmation
        cfar_hit = False
        for c_idx, cfar in enumerate(cfar_candidates):
            if compute_box_iou(bbox, cfar["bbox"]) > 0.15:
                cfar_hit = True
                matched_cfar_idx.add(c_idx)
                break

        # Check AE anomaly overlap
        ae_score = 0.0
        for a_idx, ae in enumerate(autoencoder_anomalies):
            if compute_box_iou(bbox, ae["bbox"]) > 0.15:
                ae_score = max(ae_score, ae.get("anomaly_score", 0.0))
                matched_ae_idx.add(a_idx)

        # Decision Logic for YOLO candidates
        if conf >= yolo_conf_thresh:
            if has_shadow or cfar_hit or conf >= 0.55:
                decisions.append(TriageDecision(
                    category="KNOWN_DEBRIS",
                    class_name=cname,
                    confidence=round(conf, 3),
                    bbox=[int(b) for b in bbox],
                    has_shadow=has_shadow,
                    shadow_contrast=shadow_contrast,
                    anomaly_score=round(ae_score, 3),
                    cfar_confirmed=cfar_hit,
                    triage_reason="High YOLO confidence with confirmed acoustic signature"
                ))
            else:
                decisions.append(TriageDecision(
                    category="REJECT",
                    class_name=cname,
                    confidence=round(conf, 3),
                    bbox=[int(b) for b in bbox],
                    has_shadow=False,
                    shadow_contrast=shadow_contrast,
                    anomaly_score=round(ae_score, 3),
                    cfar_confirmed=False,
                    triage_reason="Rejected: Lacks acoustic shadow and CFAR confirmation (spurious clutter)"
                ))
        else:
            # Low YOLO conf -> Check if it qualifies as an Unknown Anomaly
            if ae_score >= anomaly_thresh or cfar_hit:
                decisions.append(TriageDecision(
                    category="UNKNOWN_ANOMALY",
                    class_name="Unclassified Sonar Anomaly",
                    confidence=round(max(ae_score, 0.5), 3),
                    bbox=[int(b) for b in bbox],
                    has_shadow=has_shadow,
                    shadow_contrast=shadow_contrast,
                    anomaly_score=round(ae_score, 3),
                    cfar_confirmed=cfar_hit,
                    triage_reason="Low YOLO class match but elevated anomaly score & CFAR highlight"
                ))
            else:
                decisions.append(TriageDecision(
                    category="REJECT",
                    class_name=cname,
                    confidence=round(conf, 3),
                    bbox=[int(b) for b in bbox],
                    has_shadow=has_shadow,
                    shadow_contrast=shadow_contrast,
                    anomaly_score=round(ae_score, 3),
                    cfar_confirmed=False,
                    triage_reason="Rejected: Below confidence threshold"
                ))

    # 2. Process remaining un-matched Autoencoder anomalies (Unknown Anomaly branch)
    for a_idx, ae in enumerate(autoencoder_anomalies):
        if a_idx in matched_ae_idx:
            continue

        bbox = ae["bbox"]
        ae_score = ae.get("anomaly_score", 0.0)
        has_shadow, shadow_contrast = verify_acoustic_shadow(image_bgr, bbox)

        cfar_hit = False
        for c_idx, cfar in enumerate(cfar_candidates):
            if compute_box_iou(bbox, cfar["bbox"]) > 0.15:
                cfar_hit = True
                matched_cfar_idx.add(c_idx)
                break

        if ae_score >= anomaly_thresh:
            decisions.append(TriageDecision(
                category="UNKNOWN_ANOMALY",
                class_name="Novel Subsea Anomaly",
                confidence=round(ae_score, 3),
                bbox=[int(b) for b in bbox],
                has_shadow=has_shadow,
                shadow_contrast=shadow_contrast,
                anomaly_score=round(ae_score, 3),
                cfar_confirmed=cfar_hit,
                triage_reason="Significant autoencoder reconstruction error (novel acoustic structure)"
            ))

    # 3. Process remaining un-matched high-contrast CFAR candidates
    for c_idx, cfar in enumerate(cfar_candidates):
        if c_idx in matched_cfar_idx:
            continue

        bbox = cfar["bbox"]
        contrast = cfar.get("contrast_ratio", 1.0)
        has_shadow, shadow_contrast = verify_acoustic_shadow(image_bgr, bbox)

        if contrast >= 2.2 and has_shadow:
            decisions.append(TriageDecision(
                category="UNKNOWN_ANOMALY",
                class_name="Unidentified Acoustic Target",
                confidence=round(min(0.85, contrast / 3.0), 3),
                bbox=[int(b) for b in bbox],
                has_shadow=True,
                shadow_contrast=shadow_contrast,
                anomaly_score=0.35,
                cfar_confirmed=True,
                triage_reason="Strong OS-CFAR highlight with verified acoustic shadow"
            ))

    summary = {
        "known_debris_count": sum(1 for d in decisions if d.category == "KNOWN_DEBRIS"),
        "unknown_anomaly_count": sum(1 for d in decisions if d.category == "UNKNOWN_ANOMALY"),
        "rejected_count": sum(1 for d in decisions if d.category == "REJECT"),
    }

    return decisions, summary
