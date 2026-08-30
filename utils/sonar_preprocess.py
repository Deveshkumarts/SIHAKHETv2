"""
Universal Marine & Sonar Image Preprocessing Module
Implements the standardized 3-Stage Acoustic & Optical Enhancement Pipeline:
  Step 1: Median Filter (Removes isolated salt-and-pepper noise & high-frequency speckle spikes)
  Step 2: Bilateral Filter (Edge-preserving smoothing, protects structural boundaries & acoustic shadows)
  Step 3: CLAHE (Contrast-Limited Adaptive Histogram Equalization on Luminance channel in LAB color space)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


def apply_median_filter(image_bgr: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Step 1: Median Filtering.
    Suppresses salt-and-pepper acoustic reverberation spikes and impulse sensor noise.
    """
    if ksize <= 1:
        return image_bgr
    if ksize % 2 == 0:
        ksize += 1  # Kernel size must be odd
    return cv2.medianBlur(image_bgr, ksize)


def apply_bilateral_denoise(
    image_bgr: np.ndarray,
    d: int = 5,
    sigma_color: float = 35.0,
    sigma_space: float = 35.0
) -> np.ndarray:
    """
    Step 2: Bilateral Filtering.
    Non-linear spatial and radiometric smoothing that removes diffuse speckle
    while strictly preserving acoustic shadow edges and physical object silhouettes.
    """
    return cv2.bilateralFilter(image_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def apply_clahe(
    image_bgr: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Step 3: Contrast-Limited Adaptive Histogram Equalization.
    Converts to CIE-LAB color space and equalizes only the Luminance (L) channel,
    enhancing deep shadow gradients and low-contrast underwater target highlights.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl_channel = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((cl_channel, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def preprocess_universal_image(
    image_bgr: np.ndarray,
    median_ksize: int = 3,
    bilateral_d: int = 5,
    bilateral_sigma: float = 35.0,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Standardized 3-Stage Preprocessing Pipeline:
      Raw -> Median Filter -> Bilateral Filter -> CLAHE
    """
    # 1. Median Filter
    step1 = apply_median_filter(image_bgr, ksize=median_ksize)
    
    # 2. Bilateral Filter
    step2 = apply_bilateral_denoise(
        step1,
        d=bilateral_d,
        sigma_color=bilateral_sigma,
        sigma_space=bilateral_sigma
    )
    
    # 3. CLAHE
    step3 = apply_clahe(
        step2,
        clip_limit=clahe_clip,
        tile_grid_size=clahe_grid
    )
    
    return step3


# Backwards compatibility alias
def preprocess_sonar_image(image_bgr: np.ndarray, mode: str = "3stage") -> np.ndarray:
    if mode in ["3stage", "clahe_denoise"]:
        return preprocess_universal_image(image_bgr)
    elif mode == "clahe":
        return apply_clahe(image_bgr)
    elif mode == "denoise":
        return apply_bilateral_denoise(image_bgr)
    elif mode == "median":
        return apply_median_filter(image_bgr)
    elif mode == "raw":
        return image_bgr
    return preprocess_universal_image(image_bgr)
