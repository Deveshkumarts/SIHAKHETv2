"""
Side-Scan Sonar (SSS) Calibration, Correction & Data Quality Module.
Implements:
  1. Data Quality & Sensor Integrity Validation (Frame check, duplicate detection, SNR check)
  2. Automated Nadir & First Bottom Return (FBR) Detection
  3. Water-Column Removal (WCR)
  4. Slant-to-Ground Range Conversion (SRC)
  5. Time-Varying Gain (TVG) Radiometric Attenuation Compensation
  6. Quantitative Acoustic SNR & Clutter Metrics
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


@dataclass
class QualityMetrics:
    is_valid: bool
    snr_db: float
    dynamic_range_db: float
    clutter_mean: float
    clutter_std: float
    saturation_ratio: float
    zero_ratio: float
    warnings: List[str]


def compute_snr_index(image_bgr: np.ndarray) -> QualityMetrics:
    """
    Computes acoustic Signal-to-Noise Ratio (SNR) in dB, dynamic range,
    clutter noise statistics, and checks for sensor saturation or blackout.
    """
    warnings = []
    if image_bgr is None or image_bgr.size == 0:
        return QualityMetrics(
            is_valid=False, snr_db=0.0, dynamic_range_db=0.0,
            clutter_mean=0.0, clutter_std=0.0, saturation_ratio=0.0,
            zero_ratio=1.0, warnings=["Image is empty or null"]
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    gray_f = gray.astype(np.float32)

    total_pixels = gray.size
    zero_count = np.count_nonzero(gray == 0)
    sat_count = np.count_nonzero(gray == 255)
    zero_ratio = zero_count / total_pixels
    sat_ratio = sat_count / total_pixels

    if zero_ratio > 0.40:
        warnings.append(f"High black-level ratio ({zero_ratio*100:.1f}%); wide water column or sensor dropout.")
    if sat_ratio > 0.15:
        warnings.append(f"High acoustic saturation ratio ({sat_ratio*100:.1f}%); gain setting too high.")

    # Target highlight vs background clutter noise floor
    # Bottom 70% intensity represents seafloor clutter, top 5% represents acoustic highlights
    p5 = float(np.percentile(gray_f, 5))
    p70 = float(np.percentile(gray_f, 70))
    p98 = float(np.percentile(gray_f, 98))

    clutter_mask = (gray_f >= p5) & (gray_f <= p70)
    clutter_pixels = gray_f[clutter_mask]

    clutter_mean = float(np.mean(clutter_pixels)) if clutter_pixels.size > 0 else 1.0
    clutter_std = float(np.std(clutter_pixels)) if clutter_pixels.size > 0 else 1.0

    # Acoustic Signal-to-Clutter/Noise Ratio (SCNR / SNR in dB)
    signal_level = max(1.0, p98 - clutter_mean)
    noise_level = max(1.0, clutter_std)
    snr_db = float(20.0 * np.log10(signal_level / noise_level))

    min_val = float(np.min(gray_f))
    max_val = float(np.max(gray_f))
    dynamic_range_db = float(20.0 * np.log10(max(1.0, max_val - min_val)))

    is_valid = True
    if snr_db < 3.0:
        warnings.append(f"Low Acoustic SNR ({snr_db:.1f} dB < 3.0 dB); low contrast or high turbidity.")
        is_valid = False

    return QualityMetrics(
        is_valid=is_valid,
        snr_db=round(snr_db, 2),
        dynamic_range_db=round(dynamic_range_db, 2),
        clutter_mean=round(clutter_mean, 2),
        clutter_std=round(clutter_std, 2),
        saturation_ratio=round(sat_ratio, 4),
        zero_ratio=round(zero_ratio, 4),
        warnings=warnings
    )


def detect_duplicate_frame(
    current_img: np.ndarray,
    previous_img: np.ndarray,
    mse_threshold: float = 2.0
) -> Tuple[bool, float]:
    """
    Detects duplicate / stalled survey frames using Mean Squared Error.
    """
    if previous_img is None or current_img.shape != previous_img.shape:
        return False, 999.0

    mse = float(np.mean((current_img.astype(np.float32) - previous_img.astype(np.float32)) ** 2))
    is_duplicate = mse < mse_threshold
    return is_duplicate, round(mse, 4)


def detect_nadir_lines(
    image_bgr: np.ndarray,
    gradient_threshold: float = 15.0,
    min_water_column_frac: float = 0.03,
    max_water_column_frac: float = 0.35
) -> Tuple[int, int]:
    """
    Automatically detects the Port and Starboard First Bottom Return (FBR) / Nadir lines
    where acoustic backscatter first hits the seafloor.

    Returns:
        (nadir_port_col, nadir_starboard_col) pixel indices.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
    h, w = gray.shape
    mid = w // 2

    # Mean intensity profile along range axis
    intensity_profile = np.mean(gray.astype(np.float32), axis=0)

    # Smooth profile to eliminate speckle spikes
    profile_smooth = cv2.GaussianBlur(intensity_profile.reshape(1, -1), (1, 15), 3.0).flatten()

    # Calculate spatial derivative from center outward
    grad = np.gradient(profile_smooth)

    min_offset = int(w * min_water_column_frac)
    max_offset = int(w * max_water_column_frac)

    # Port search (center to left: mid -> 0)
    port_slice = grad[mid - max_offset:mid - min_offset]
    port_idx = int(np.argmin(port_slice)) if port_slice.size > 0 else 0
    nadir_port = (mid - max_offset) + port_idx

    # Starboard search (center to right: mid -> w)
    stbd_slice = grad[mid + min_offset:mid + max_offset]
    stbd_idx = int(np.argmax(stbd_slice)) if stbd_slice.size > 0 else 0
    nadir_stbd = (mid + min_offset) + stbd_idx

    # Safety clamps
    nadir_port = int(np.clip(nadir_port, 0, mid - 5))
    nadir_stbd = int(np.clip(nadir_stbd, mid + 5, w - 1))

    return nadir_port, nadir_stbd


def remove_water_column(
    image_bgr: np.ndarray,
    nadir_port: Optional[int] = None,
    nadir_stbd: Optional[int] = None,
    mode: str = "crop"
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Removes the blind water-column zone from side-scan sonar imagery.

    Args:
        image_bgr: Input SSS image (H, W, 3)
        nadir_port: Left seafloor contact column (auto-detected if None)
        nadir_stbd: Right seafloor contact column (auto-detected if None)
        mode: 'crop' (stitches port and stbd seafloor together) or 'mask' (zeros out water column)

    Returns:
        (calibrated_image, (nadir_port, nadir_stbd))
    """
    h, w = image_bgr.shape[:2]
    if nadir_port is None or nadir_stbd is None:
        nadir_port, nadir_stbd = detect_nadir_lines(image_bgr)

    if mode == "crop":
        port_seafloor = image_bgr[:, :nadir_port]
        stbd_seafloor = image_bgr[:, nadir_stbd:]
        if port_seafloor.shape[1] > 0 and stbd_seafloor.shape[1] > 0:
            corrected = np.hstack([port_seafloor, stbd_seafloor])
            # Resize back to original width to preserve aspect ratio
            corrected_resized = cv2.resize(corrected, (w, h), interpolation=cv2.INTER_LINEAR)
            return corrected_resized, (nadir_port, nadir_stbd)
        return image_bgr, (nadir_port, nadir_stbd)

    elif mode == "mask":
        masked = image_bgr.copy()
        masked[:, nadir_port:nadir_stbd] = 0
        return masked, (nadir_port, nadir_stbd)

    return image_bgr, (nadir_port, nadir_stbd)


def slant_to_ground_range_conversion(
    image_bgr: np.ndarray,
    altitude_m: float = 10.0,
    slant_range_m: float = 75.0
) -> np.ndarray:
    """
    Converts acoustic slant range Rs to flat seafloor ground range Rg:
      Rg = sqrt(max(0, Rs^2 - h^2))

    Corrects non-linear pixel compression near nadir and restores true spatial geometry.
    """
    if altitude_m <= 0.5 or altitude_m >= slant_range_m:
        return image_bgr

    h, w = image_bgr.shape[:2]
    mid = w / 2.0

    # Normalize pixel range to meters: distance from center
    # Port side: 0 -> -slant_range_m; Stbd side: 0 -> +slant_range_m
    x_indices = np.arange(w, dtype=np.float32)
    rs_m = np.abs((x_indices - mid) / mid) * slant_range_m

    # Ground range formula
    rg_m = np.sqrt(np.maximum(0.0, rs_m**2 - altitude_m**2))
    max_rg = math.sqrt(max(1.0, slant_range_m**2 - altitude_m**2))

    # Map ground range back to rectilinear pixel coordinates
    rg_norm = (rg_m / max_rg)
    target_x = np.where(x_indices < mid, mid - rg_norm * mid, mid + rg_norm * mid).astype(np.float32)

    # Remap image using inverse lookup grid
    map_x = np.tile(target_x, (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w)).astype(np.float32)

    corrected = cv2.remap(image_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return corrected


def apply_tvg_compensation(
    image_bgr: np.ndarray,
    alpha: float = 1.15,
    beta: float = 0.012,
    slant_range_m: float = 75.0
) -> np.ndarray:
    """
    Time-Varying Gain (TVG) Radiometric Correction.
    Compensates for acoustic spherical spreading loss (r^alpha) and seawater absorption (10^(beta*r/20)).
    Balances near-nadir high backscatter with far-range signal decay.
    """
    h, w = image_bgr.shape[:2]
    mid = w / 2.0

    x_indices = np.arange(w, dtype=np.float32)
    r_norm = np.abs(x_indices - mid) / mid * slant_range_m
    r_safe = np.maximum(1.0, r_norm)

    # TVG Gain curve: G(r) = (r / r_ref)^alpha * 10^(beta * r / 20)
    r_ref = slant_range_m * 0.5
    spreading_loss = (r_safe / r_ref) ** alpha
    absorption_loss = 10.0 ** (beta * (r_safe - r_ref) / 20.0)
    gain_curve = spreading_loss * absorption_loss

    # Normalize gain to keep average multiplier bounded [0.5, 3.0]
    gain_curve = np.clip(gain_curve / np.mean(gain_curve), 0.4, 2.8)

    gain_map = np.tile(gain_curve.reshape(1, w, 1), (h, 1, image_bgr.shape[2] if len(image_bgr.shape) == 3 else 1))

    img_f = image_bgr.astype(np.float32) * gain_map
    img_corrected = np.clip(img_f, 0, 255).astype(np.uint8)

    return img_corrected


def apply_beam_pattern_correction(
    image_bgr: np.ndarray,
    beamwidth_deg: float = 50.0,
    aperture_factor: float = 2.4
) -> np.ndarray:
    """
    Transducer Beam Pattern / Beam Angle Directivity Correction:
    Compensates for the sinc-squared transducer directivity b(theta) across cross-track angles:
      b(theta) = (sin(beta * sin(theta)) / (beta * sin(theta)))^2
    Equalizes outer beam falloff and central nadir intensity lobes.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    h, w = image_bgr.shape[:2]
    mid_col = w / 2.0

    # Cross-track angle theta from nadir [-theta_max, +theta_max]
    theta_max_rad = math.radians(beamwidth_deg / 2.0)
    col_indices = np.arange(w, dtype=np.float32)
    # Normalized position [-1, 1]
    norm_x = (col_indices - mid_col) / max(1.0, mid_col)
    theta_rad = norm_x * theta_max_rad

    # Sinc beam directivity profile
    arg = aperture_factor * np.sin(theta_rad)
    # Avoid div by zero at nadir theta=0
    sinc_val = np.ones_like(arg)
    nonzero = np.abs(arg) > 1e-5
    sinc_val[nonzero] = np.sin(arg[nonzero]) / arg[nonzero]
    directivity = sinc_val ** 2
    directivity = np.clip(directivity, 0.25, 1.0) # Clip extreme edges

    # Inverse directivity gain curve
    beam_gain = 1.0 / directivity
    beam_gain_2d = np.tile(beam_gain, (h, 1))

    img_f = image_bgr.astype(np.float32)
    corrected = np.zeros_like(img_f)
    for c in range(image_bgr.shape[2] if len(image_bgr.shape) == 3 else 1):
        if len(image_bgr.shape) == 3:
            corrected[:, :, c] = np.clip(img_f[:, :, c] * beam_gain_2d, 0, 255)
        else:
            corrected = np.clip(img_f * beam_gain_2d, 0, 255)

    return corrected.astype(np.uint8)


def calibrate_side_scan_sonar(
    image_bgr: np.ndarray,
    altitude_m: float = 10.0,
    slant_range_m: float = 75.0,
    enable_tvg: bool = True,
    enable_wcr: bool = True,
    enable_src: bool = True,
    enable_beam_correction: bool = True,
    tvg_alpha: float = 1.2,
    tvg_beta: float = 0.04
) -> Tuple[np.ndarray, Dict[str, Union[float, int, QualityMetrics]]]:
    """
    Full Acoustic Calibration & Quality Pipeline:
      1. SNR & Quality Metrics Assessment
      2. TVG Radiometric Correction
      3. Transducer Beam Pattern Directivity Equalization
      4. Nadir Detection & Water-Column Removal (WCR)
      5. Slant-to-Ground Range Conversion (SRC)
    """
    quality = compute_snr_index(image_bgr)
    processed = image_bgr.copy()

    # 1. TVG Compensation
    if enable_tvg:
        processed = apply_tvg_compensation(
            processed,
            alpha=tvg_alpha,
            beta=tvg_beta,
            slant_range_m=slant_range_m
        )

    # 2. Transducer Beam Pattern Directivity Correction
    if enable_beam_correction:
        processed = apply_beam_pattern_correction(processed)

    # 3. Nadir & Water-Column Removal
    nadir_port, nadir_stbd = detect_nadir_lines(processed)
    if enable_wcr:
        processed, (nadir_port, nadir_stbd) = remove_water_column(
            processed,
            nadir_port=nadir_port,
            nadir_stbd=nadir_stbd,
            mode="crop"
        )

    # 4. Slant-to-Ground Range Conversion
    if enable_src:
        processed = slant_to_ground_range_conversion(
            processed,
            altitude_m=altitude_m,
            slant_range_m=slant_range_m
        )

    # Recompute calibrated SNR
    calibrated_quality = compute_snr_index(processed)

    report = {
        "raw_snr_db": quality.snr_db,
        "calibrated_snr_db": calibrated_quality.snr_db,
        "nadir_port_col": nadir_port,
        "nadir_stbd_col": nadir_stbd,
        "clutter_mean": calibrated_quality.clutter_mean,
        "clutter_std": calibrated_quality.clutter_std,
        "quality_metrics": calibrated_quality
    }

    return processed, report
