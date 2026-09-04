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
    bilateral_d: int = 7,
    bilateral_sigma: float = 50.0,
    clahe_clip: float = 1.3,
    clahe_grid: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Standardized 3-Stage Acoustic & Optical Denoising & Enhancement Pipeline:
      Step 1: Median Filter (Removes isolated salt-and-pepper noise & high-frequency acoustic speckle spikes)
      Step 2: Bilateral Filter (Smooths background speckle and seabed grain while preserving physical object boundaries)
      Step 3: Gentle CLAHE (Enhances target highlight vs shadow dynamic range without noise amplification)
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # 1. Median Filter (Speckle spike suppression)
    step1 = apply_median_filter(image_bgr, ksize=median_ksize)
    
    # 2. Bilateral Denoising (Edge-preserving background smoothing)
    step2 = apply_bilateral_denoise(
        step1,
        d=bilateral_d,
        sigma_color=bilateral_sigma,
        sigma_space=bilateral_sigma
    )
    
    # 3. CLAHE with adaptive grid on Luminance channel only
    lab = cv2.cvtColor(step2, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    h, w = l_channel.shape
    gw = max(2, min(8, w // 20))
    gh = max(2, min(8, h // 20))
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(gw, gh))
    cl_channel = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((cl_channel, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def calibrate_and_preprocess_sonar(
    image_bgr: np.ndarray,
    altitude_m: float = 10.0,
    slant_range_m: float = 75.0,
    enable_calibration: bool = True,
    enable_wcr: bool = True,
    enable_src: bool = True,
    enable_tvg: bool = True,
    tvg_alpha: float = 1.15,
    tvg_beta: float = 0.012,
    enable_3stage: bool = True,
    median_ksize: int = 3,
    bilateral_d: int = 5,
    bilateral_sigma: float = 35.0,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
) -> Tuple[np.ndarray, dict]:
    """
    Complete Side-Scan Sonar Acoustic Signal Chain:
      Raw -> TVG Compensation -> Water-Column Removal -> Slant-to-Ground -> 3-Stage Enhancement -> SNR Index
    """
    from utils.sonar_calibration import calibrate_side_scan_sonar, compute_snr_index

    raw_snr = compute_snr_index(image_bgr)
    processed = image_bgr.copy()
    report = {"raw_snr_db": raw_snr.snr_db, "warnings": list(raw_snr.warnings)}

    if enable_calibration:
        processed, cal_report = calibrate_side_scan_sonar(
            processed,
            altitude_m=altitude_m,
            slant_range_m=slant_range_m,
            enable_wcr=enable_wcr,
            enable_src=enable_src,
            enable_tvg=enable_tvg,
            tvg_alpha=tvg_alpha,
            tvg_beta=tvg_beta,
        )
        report.update(cal_report)

    if enable_3stage:
        processed = preprocess_universal_image(
            processed,
            median_ksize=median_ksize,
            bilateral_d=bilateral_d,
            bilateral_sigma=bilateral_sigma,
            clahe_clip=clahe_clip,
            clahe_grid=clahe_grid,
        )

    final_snr = compute_snr_index(processed)
    report["final_snr_db"] = final_snr.snr_db
    report["final_dynamic_range_db"] = final_snr.dynamic_range_db

    return processed, report


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
