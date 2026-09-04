"""
Dynamic ROI Extraction and Coordinate Projection Utilities for Sonar Imagery.
Handles expanded bounding box cropping, boundary clamping, ROI metadata tracking,
and reverse coordinate mapping from ROI segmentation space back to full sonar image space.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Optional


def get_adaptive_padding_ratio(
    conf: float = 0.80,
    snr_db: float = 12.0,
    uncertainty: str = "LOW"
) -> float:
    """
    Selects adaptive expansion padding scale:
      - Tight (1.2× / pad=0.10): High confidence, low uncertainty, clear acoustic signature.
      - Nominal (1.5× / pad=0.25): Standard survey detections.
      - Loose (2.0× / pad=0.50): High uncertainty, low SNR (< 8 dB) to capture full acoustic shadow.
    """
    if uncertainty == "HIGH" or snr_db < 8.0 or conf < 0.35:
        return 0.50  # Loose (2.0x bounding box)
    elif conf >= 0.70 and uncertainty == "LOW" and snr_db >= 12.0:
        return 0.10  # Tight (1.2x bounding box)
    return 0.25      # Nominal (1.5x bounding box)


def validate_roi_quality(
    roi_crop: np.ndarray,
    original_bbox: List[float],
    clamped_bbox: List[int],
    image_shape: Tuple[int, int, ...]
) -> Tuple[bool, float, str]:
    """
    ROI Quality Gate:
    Validates crop dimensions, boundary clipping, and pixel variance before feeding to SegFormer.

    Returns:
        (is_valid, quality_score_0_to_1, status_reason)
    """
    if roi_crop is None or roi_crop.size == 0:
        return False, 0.0, "Empty crop"

    h_crop, w_crop = roi_crop.shape[:2]
    if h_crop < 10 or w_crop < 10:
        return False, 0.2, "Crop too small (< 10px)"

    # Check pixel variance (reject solid black water-column or saturated washout)
    gray = cv2.cvtColor(roi_crop, cv2.COLOR_BGR2GRAY) if len(roi_crop.shape) == 3 else roi_crop
    std_val = float(np.std(gray))
    if std_val < 3.0:
        return False, 0.3, "Uniform intensity / zero contrast residual"

    # Check edge clipping fraction
    h_img, w_img = image_shape[:2]
    rx1, ry1, rx2, ry2 = clamped_bbox
    clipped_edges = 0
    if rx1 == 0: clipped_edges += 1
    if ry1 == 0: clipped_edges += 1
    if rx2 == w_img: clipped_edges += 1
    if ry2 == h_img: clipped_edges += 1

    clip_penalty = clipped_edges * 0.15
    quality_score = max(0.4, round(1.0 - clip_penalty, 2))
    return True, quality_score, "Good ROI"


def expand_and_clamp_bbox(
    bbox: List[float],
    image_shape: Tuple[int, int, ...],
    padding_ratio: float = 0.25
) -> List[int]:
    """
    Expand bounding box [x1, y1, x2, y2] by padding_ratio and clamp strictly to image boundaries.

    Args:
        bbox: [x1, y1, x2, y2]
        image_shape: (H, W, C)
        padding_ratio: Fractional expansion applied to width and height (e.g. 0.10=1.2x, 0.25=1.5x, 0.50=2.0x)

    Returns:
        [rx1, ry1, rx2, ry2] as integers clamped to [0, 0, W, H]
    """
    h_img, w_img = image_shape[:2]
    x1, y1, x2, y2 = bbox

    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)

    pad_x = w * padding_ratio
    pad_y = h * padding_ratio

    rx1 = int(max(0, np.floor(x1 - pad_x)))
    ry1 = int(max(0, np.floor(y1 - pad_y)))
    rx2 = int(min(w_img, np.ceil(x2 + pad_x)))
    ry2 = int(min(h_img, np.ceil(y2 + pad_y)))

    # Ensure non-zero width and height
    if rx2 <= rx1:
        rx2 = min(w_img, rx1 + 1)
    if ry2 <= ry1:
        ry2 = min(h_img, ry1 + 1)

    return [rx1, ry1, rx2, ry2]


def extract_rois_from_image(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    padding_ratio: float = 0.25,
    target_size: Tuple[int, int] = (224, 224),
    source_id: str = "sonar_image"
) -> List[Dict[str, Any]]:
    """
    Extract expanded ROI crops for each YOLO detection with complete spatial metadata.

    Returns:
        List of ROI objects:
        {
            "roi_id": str,
            "source_id": str,
            "class_id": int,
            "class_name": str,
            "yolo_conf": float,
            "original_bbox": [x1, y1, x2, y2],
            "roi_bbox": [rx1, ry1, rx2, ry2],
            "padding_ratio": float,
            "roi_crop_raw": np.ndarray (H_crop, W_crop, 3),
            "roi_crop_resized": np.ndarray (target_size[1], target_size[0], 3),
            "original_image_shape": (H, W, 3)
        }
    """
    rois = []
    h_img, w_img = image_bgr.shape[:2]

    for idx, det in enumerate(detections):
        bbox = det["bbox"]
        rx1, ry1, rx2, ry2 = expand_and_clamp_bbox(bbox, image_bgr.shape, padding_ratio)

        roi_crop_raw = image_bgr[ry1:ry2, rx1:rx2].copy()
        if roi_crop_raw.size == 0:
            continue

        roi_crop_resized = cv2.resize(roi_crop_raw, target_size, interpolation=cv2.INTER_LINEAR)

        rois.append({
            "roi_id": f"{source_id}_det{idx:02d}_{det['class_name']}",
            "source_id": source_id,
            "class_id": det["class_id"],
            "class_name": det["class_name"],
            "yolo_conf": det["confidence"],
            "original_bbox": bbox,
            "roi_bbox": [rx1, ry1, rx2, ry2],
            "padding_ratio": padding_ratio,
            "roi_crop_raw": roi_crop_raw,
            "roi_crop_resized": roi_crop_resized,
            "original_image_shape": (h_img, w_img, image_bgr.shape[2] if len(image_bgr.shape) > 2 else 1)
        })

    return rois


def roi_mask_to_full_image(
    roi_mask: np.ndarray,
    roi_bbox: List[int],
    original_image_shape: Tuple[int, int, ...],
    threshold: float = 0.5
) -> np.ndarray:
    """
    Project a predicted ROI segmentation mask (e.g. 224x224) back to the full resolution sonar image space.

    Args:
        roi_mask: Binary or probability mask from SegFormer (H_mask, W_mask) in [0, 1]
        roi_bbox: [rx1, ry1, rx2, ry2] in original image pixel coordinates
        original_image_shape: (H_full, W_full, ...)
        threshold: Binarization threshold (default 0.5)

    Returns:
        Full resolution binary mask array of shape (H_full, W_full) with dtype uint8 (0 or 255)
    """
    rx1, ry1, rx2, ry2 = roi_bbox
    crop_w = max(1, rx2 - rx1)
    crop_h = max(1, ry2 - ry1)
    h_full, w_full = original_image_shape[:2]

    # Resize ROI mask back to raw crop dimensions using NEAREST neighbor to prevent interpolation artifacts
    mask_crop_resized = cv2.resize(roi_mask.astype(np.float32), (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
    binary_crop = (mask_crop_resized >= threshold).astype(np.uint8) * 255

    # Embed into full-size zero array
    full_mask = np.zeros((h_full, w_full), dtype=np.uint8)
    full_mask[ry1:ry2, rx1:rx2] = binary_crop

    return full_mask
