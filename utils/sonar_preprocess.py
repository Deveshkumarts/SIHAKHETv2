"""
Universal Marine & Sonar Image Preprocessing Module
Implements the standardized 4-Stage Acoustic & Optical Enhancement Pipeline:
  Step 1: Median Filter (Removes isolated salt-and-pepper noise & high-frequency speckle spikes)
  Step 2: Bilateral Filter (Edge-preserving smoothing, protects structural boundaries & acoustic shadows)
  Step 3: CLAHE (Contrast-Limited Adaptive Histogram Equalization on Luminance channel in LAB color space)
  Step 4: Unsharp-Mask Sharpening (Restores target/edge micro-contrast softened by denoising)
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


def apply_unsharp_mask(
    image_bgr: np.ndarray,
    amount: float = 1.0,
    radius: float = 2.0,
    threshold: int = 3
) -> np.ndarray:
    """
    Step 4: Unsharp-Mask Sharpening.
    Restores target-edge and micro-texture contrast that median/bilateral denoising
    softens, so debris silhouettes and acoustic shadow boundaries visibly "pop"
    against the seabed instead of blending into it. `threshold` avoids re-amplifying
    flat-noise pixels (only edges above the threshold get sharpened).
    """
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), sigmaX=radius, sigmaY=radius)
    sharpened = cv2.addWeighted(image_bgr, 1.0 + amount, blurred, -amount, 0)

    if threshold > 0:
        low_contrast_mask = np.abs(image_bgr.astype(np.int16) - blurred.astype(np.int16)) < threshold
        sharpened = np.where(low_contrast_mask, image_bgr, sharpened)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def preprocess_universal_image(
    image_bgr: np.ndarray,
    median_ksize: int = 3,
    bilateral_d: int = 7,
    bilateral_sigma: float = 50.0,
    clahe_clip: float = 2.6,
    clahe_grid: Tuple[int, int] = (8, 8),
    sharpen_amount: float = 1.1,
    sharpen_radius: float = 2.0,
) -> np.ndarray:
    """
    Standardized 4-Stage Acoustic & Optical Denoising & Enhancement Pipeline:
      Step 1: Median Filter (Removes isolated salt-and-pepper noise & high-frequency acoustic speckle spikes)
      Step 2: Bilateral Filter (Smooths background speckle and seabed grain while preserving physical object boundaries)
      Step 3: Strong Adaptive CLAHE (Pushes target-highlight vs acoustic-shadow dynamic range hard, tile-local)
      Step 4: Unsharp-Mask Sharpening (Re-injects edge/texture micro-contrast so the result reads as visibly enhanced,
              not just denoised)
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    # Tiny thumbnails (e.g. sample-library preview chips, small dataset crops)
    # have very few pixels per tile — full strength posterizes them instead of
    # enhancing them. Taper down smoothly below ~120px on the short side, but
    # never below 60% strength — small images still need a visibly enhanced result.
    min_side = min(image_bgr.shape[0], image_bgr.shape[1])
    if min_side < 120:
        taper = max(0.6, min_side / 120.0)
        clahe_clip = 1.1 + (clahe_clip - 1.1) * taper
        sharpen_amount = sharpen_amount * taper

    # 1. Median Filter (Speckle spike suppression)
    step1 = apply_median_filter(image_bgr, ksize=median_ksize)

    # 2. Bilateral Denoising (Edge-preserving background smoothing)
    step2 = apply_bilateral_denoise(
        step1,
        d=bilateral_d,
        sigma_color=bilateral_sigma,
        sigma_space=bilateral_sigma
    )

    # 3. CLAHE with a finer adaptive grid on the Luminance channel only —
    #    smaller tiles + a higher clip limit make the local contrast boost obvious
    #    rather than barely perceptible.
    lab = cv2.cvtColor(step2, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Target ~48px per tile so small crops get near-global (gentle) CLAHE while
    # large full-frame sonar images get finer, more locally-adaptive contrast.
    h, w = l_channel.shape
    gw = max(2, min(12, w // 48))
    gh = max(2, min(12, h // 48))
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(gw, gh))
    cl_channel = clahe.apply(l_channel)

    merged_lab = cv2.merge((cl_channel, a_channel, b_channel))
    step3 = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # 4. Unsharp-mask sharpening — brings back the edge micro-contrast the
    #    denoising stages removed, so the enhancement reads clearly against the raw image.
    step4 = apply_unsharp_mask(step3, amount=sharpen_amount, radius=sharpen_radius)

    return step4


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
    clahe_clip: float = 2.6,
    clahe_grid: Tuple[int, int] = (8, 8),
    sharpen_amount: float = 1.1,
    sharpen_radius: float = 2.0,
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
            sharpen_amount=sharpen_amount,
            sharpen_radius=sharpen_radius,
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
