"""
YOLO + SegFormer Fusion Module.
Combines YOLO detection confidence with SegFormer segmentation quality score
to produce a final weighted fusion score and VERIFIED / REVIEW / REJECT classification.
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def compute_segmentation_quality_score(
    roi_binary_mask: np.ndarray,
    roi_bbox: List[int],
    roi_shape: tuple,
    fg_probability: float,
) -> float:
    """
    Compute a meaningful Segmentation Quality Score (S_seg) from measurable properties.

    Score Components:
    1. Foreground Probability (fg_probability): Mean sigmoid output in mask region [0, 1]
    2. Mask Area Validity (area_factor): Penalizes trivially empty or full masks
    3. IoU Overlap with YOLO ROI Center (overlap_factor): Ensures mask is within the expected object region

    Final Score = fg_probability * area_factor * overlap_factor

    Args:
        roi_binary_mask: (H, W) uint8 binary mask (0 or 255)
        roi_bbox: [rx1, ry1, rx2, ry2] original image ROI coordinates
        roi_shape: (H_roi, W_roi) dimensions of the ROI crop
        fg_probability: Mean foreground probability from SegFormer sigmoid output

    Returns:
        S_seg in [0.0, 1.0]
    """
    h_roi, w_roi = roi_shape[:2]
    total_pixels = max(1, h_roi * w_roi)
    fg_pixels = (roi_binary_mask > 127).sum()
    fg_ratio = fg_pixels / total_pixels

    # 1. Area validity: penalize empty masks (< 1%) or masks that flood > 90% of ROI
    if fg_ratio < 0.01:
        area_factor = fg_ratio * 10.0  # reward first few detections
    elif fg_ratio > 0.90:
        area_factor = max(0.1, 1.0 - fg_ratio)
    else:
        # Peak at 20%-60% foreground ratio (typical sonar target coverage)
        peak = 0.35
        area_factor = 1.0 - abs(fg_ratio - peak) / peak
        area_factor = max(0.3, min(1.0, area_factor))

    # 2. Overlap factor: Does mask fall in the center 70% of ROI (expected target location)?
    if fg_pixels > 0:
        center_x = int(w_roi * 0.15)
        center_y = int(h_roi * 0.15)
        center_mask = roi_binary_mask[center_y:h_roi - center_y, center_x:w_roi - center_x]
        center_fg = (center_mask > 127).sum()
        overlap_factor = min(1.0, center_fg / max(1, fg_pixels))
    else:
        overlap_factor = 0.0

    s_seg = fg_probability * area_factor * overlap_factor
    return round(min(1.0, max(0.0, s_seg)), 4)


def fuse_detection(
    yolo_conf: float,
    seg_quality: float,
    alpha: float = 0.6,
    beta: float = 0.4,
    threshold_verified: float = 0.55,
    threshold_reject: float = 0.35,
) -> Dict[str, Any]:
    """
    Compute weighted fusion score and classify detection.

    FinalScore = alpha * YOLO_conf + beta * S_seg

    Classification:
        VERIFIED: FinalScore >= threshold_verified
        REVIEW:   threshold_reject <= FinalScore < threshold_verified
        REJECT:   FinalScore < threshold_reject

    Returns:
        {
            "yolo_conf": float,
            "seg_quality": float,
            "fusion_score": float,
            "alpha": float,
            "beta": float,
            "decision": "VERIFIED" | "REVIEW" | "REJECT"
        }
    """
    fusion_score = alpha * yolo_conf + beta * seg_quality
    fusion_score = round(min(1.0, max(0.0, fusion_score)), 4)

    if fusion_score >= threshold_verified:
        decision = "VERIFIED"
    elif fusion_score >= threshold_reject:
        decision = "REVIEW"
    else:
        decision = "REJECT"

    return {
        "yolo_conf": round(yolo_conf, 4),
        "seg_quality": round(seg_quality, 4),
        "fusion_score": fusion_score,
        "alpha": alpha,
        "beta": beta,
        "decision": decision,
    }


def ablation_sweep(
    yolo_conf: float,
    seg_quality: float,
    alpha_beta_pairs: Optional[List] = None,
    threshold_verified: float = 0.55,
    threshold_reject: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple alpha/beta weight ratios and return all fusion results.
    """
    if alpha_beta_pairs is None:
        alpha_beta_pairs = [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6)]

    results = []
    for alpha, beta in alpha_beta_pairs:
        result = fuse_detection(yolo_conf, seg_quality, alpha, beta, threshold_verified, threshold_reject)
        results.append(result)
    return results


def run_fusion_on_detections(
    detections: List[Dict[str, Any]],
    alpha: float = 0.6,
    beta: float = 0.4,
    threshold_verified: float = 0.55,
    threshold_reject: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    Apply fusion to a list of detections that already include 'seg_quality' key.
    Each detection dict must have: class_id, class_name, confidence, bbox, seg_quality
    Returns enriched list with fusion_score and decision added.
    """
    fused = []
    for det in detections:
        fusion_result = fuse_detection(
            yolo_conf=det["confidence"],
            seg_quality=det.get("seg_quality", 0.0),
            alpha=alpha,
            beta=beta,
            threshold_verified=threshold_verified,
            threshold_reject=threshold_reject,
        )
        fused.append({**det, **fusion_result})
    return fused
