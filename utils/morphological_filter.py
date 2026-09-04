"""
Advanced Morphological Filtering & Shape-Constrained False Positive Filter.
Implements:
  1. Morphological Top-Hat, Opening, Closing & Noise Blob Rejection
  2. Connected Component Analysis & Labeling (CCL)
  3. Explicit Size, Solidity, and Aspect Ratio Geometrical Shape Constraints
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class MorphologicalFeatures:
    area: int
    aspect_ratio: float
    extent: float       # Area / BoundingBox Area
    solidity: float     # Area / Convex Hull Area
    compactness: float  # 4 * pi * Area / Perimeter^2
    is_valid_debris: bool
    rejection_reason: str


def apply_tophat_morphology(
    image_gray: np.ndarray,
    kernel_size: int = 15
) -> np.ndarray:
    """
    White Top-Hat Transform:
    TopHat(I) = I - Opening(I)
    Isolates small, bright acoustic highlights sitting on non-uniform seafloor reverberation.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(image_gray, cv2.MORPH_TOPHAT, kernel)
    return tophat


def clean_noise_blobs(
    binary_mask: np.ndarray,
    min_area: int = 15,
    max_area: int = 25000
) -> np.ndarray:
    """
    Morphological noise cleanup removing isolated speckle spikes and oversized scanline artifacts.
    """
    # Morphological opening (erosion followed by dilation)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)

    # Morphological closing (dilation followed by erosion) to connect fragmented highlight parts
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    # Filter connected components by area bounds
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    clean_mask = np.zeros_like(binary_mask)

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            clean_mask[labels == i] = 255

    return clean_mask


def extract_morphological_features(
    roi_binary_mask: np.ndarray
) -> MorphologicalFeatures:
    """
    Computes spatial shape descriptors using Connected Component Labeling & Contour Geometry.
    """
    if roi_binary_mask is None or np.count_nonzero(roi_binary_mask) == 0:
        return MorphologicalFeatures(
            area=0, aspect_ratio=1.0, extent=0.0, solidity=0.0,
            compactness=0.0, is_valid_debris=False, rejection_reason="Empty mask"
        )

    contours, _ = cv2.findContours(roi_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return MorphologicalFeatures(
            area=0, aspect_ratio=1.0, extent=0.0, solidity=0.0,
            compactness=0.0, is_valid_debris=False, rejection_reason="No contour found"
        )

    main_contour = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(main_contour))
    if area < 10:
        return MorphologicalFeatures(
            area=area, aspect_ratio=1.0, extent=0.0, solidity=0.0,
            compactness=0.0, is_valid_debris=False, rejection_reason="Area too small (< 10 px)"
        )

    x, y, w, h = cv2.boundingRect(main_contour)
    aspect_ratio = round(float(w) / max(1.0, float(h)), 3)
    extent = round(float(area) / max(1.0, float(w * h)), 3)

    hull = cv2.convexHull(main_contour)
    hull_area = cv2.contourArea(hull)
    solidity = round(float(area) / max(1.0, hull_area), 3)

    perimeter = cv2.arcLength(main_contour, True)
    compactness = round(float(4.0 * np.pi * area) / max(1.0, (perimeter ** 2)), 3)

    # Physical Side-Scan Sonar Constraints:
    # 1. Reject scanline stripes (extremely stretched aspect ratio: AR > 7.0 or AR < 0.14)
    if aspect_ratio > 7.5 or aspect_ratio < 0.13:
        return MorphologicalFeatures(
            area=area, aspect_ratio=aspect_ratio, extent=extent, solidity=solidity,
            compactness=compactness, is_valid_debris=False,
            rejection_reason=f"Aspect ratio anomaly (AR={aspect_ratio:.2f}, likely sonar scanline stripe)"
        )

    # 2. Reject extremely porous speckle clusters (solidity < 0.20)
    if solidity < 0.18:
        return MorphologicalFeatures(
            area=area, aspect_ratio=aspect_ratio, extent=extent, solidity=solidity,
            compactness=compactness, is_valid_debris=False,
            rejection_reason=f"Low solidity / diffuse reverberation (solidity={solidity:.2f})"
        )

    return MorphologicalFeatures(
        area=area, aspect_ratio=aspect_ratio, extent=extent, solidity=solidity,
        compactness=compactness, is_valid_debris=True,
        rejection_reason="Valid geometry"
    )


def filter_detection_by_morphology(
    image_bgr: np.ndarray,
    bbox: List[int],
    min_area: int = 15,
    max_area: int = 25000
) -> Tuple[bool, MorphologicalFeatures]:
    """
    Applies explicit size and shape constraints on detection bounding box crop.
    """
    h_img, w_img = image_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False, extract_morphological_features(None)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

    # Otsu thresholding on crop highlight
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clean_bin = clean_noise_blobs(binary, min_area=min_area, max_area=max_area)

    feat = extract_morphological_features(clean_bin)
    return feat.is_valid_debris, feat
