"""
Acoustic Mask Utilities & Pseudo-Label Generator.
Generates acoustic shadow & target foreground pseudo-masks from sonar ROI crops via:
1. Otsu Adaptive Intensity Thresholding (acoustic highlight & shadow extraction)
2. GrabCut Iterative Refinement
3. Morphological Cleaning
"""

import cv2
import numpy as np


def generate_acoustic_pseudo_mask(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Generate acoustic target pseudo-mask from sonar ROI image crop.
    
    Side-scan sonar targets consist of a bright acoustic highlight followed by a dark acoustic shadow.
    This function combines highlight detection (Otsu) and shadow segmentation to produce a binary target mask.
    
    Returns:
        Binary mask numpy array (uint8) of shape (H, W) where 255 = Target Foreground, 0 = Background.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY) if len(roi_bgr.shape) == 3 else roi_bgr.copy()
    h, w = gray.shape

    # 1. Bilateral filter to reduce acoustic speckle noise while preserving object boundaries
    filtered = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)

    # 2. Otsu thresholding for high-intensity acoustic echo highlights
    _, thresh_highlight = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Adaptive thresholding for low-intensity acoustic shadow regions in center area
    thresh_shadow = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Center mask prior (targets sit in the central 70% of the cropped ROI)
    center_mask = np.zeros((h, w), dtype=np.uint8)
    margin_y, margin_x = int(h * 0.15), int(w * 0.15)
    center_mask[margin_y:h-margin_y, margin_x:w-margin_x] = 255

    combined_initial = cv2.bitwise_or(thresh_highlight, thresh_shadow)
    combined_initial = cv2.bitwise_and(combined_initial, center_mask)

    # 4. GrabCut Refinement using initial mask as seed
    mask_grabcut = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    # Probable foreground in center
    mask_grabcut[margin_y:h-margin_y, margin_x:w-margin_x] = cv2.GC_PR_FGD
    # Definite foreground where Otsu highlight is strong
    mask_grabcut[thresh_highlight > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(roi_bgr, mask_grabcut, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
        binary_mask = np.where((mask_grabcut == cv2.GC_FGD) | (mask_grabcut == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    except Exception:
        # Fallback to combined initial threshold if GrabCut fails
        binary_mask = combined_initial

    # 5. Morphological closing to fill acoustic holes and remove speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return binary_mask
