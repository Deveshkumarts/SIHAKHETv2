"""
Sonar Coverage Quality Index (SCQI) and Targeted Resurvey Engine
SIH26057 - Akhet: Marine Guard ("Turning Echoes into Impact")

Implements the multi-criteria survey quality assessment:
  SCQI = w1*SNR + w2*Altitude_Compliance + w3*Motion_Stability + w4*Speed_Uniformity + w5*Acoustic_Shadow_Clarity
Flags low-quality acoustic swaths that fail hydrographic standards and generates
targeted resurvey advisories.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import cv2

from utils.telemetry_parser import TelemetryRecord
from utils.sonar_calibration import compute_snr_index, QualityMetrics


@dataclass
class SCQIResult:
    overall_score: float              # 0 to 100
    grade: str                        # "EXCELLENT", "GOOD", "ACCEPTABLE", "DEGRADED", "RESURVEY_REQUIRED"
    snr_score: float                  # 0 to 100
    altitude_compliance_score: float  # 0 to 100
    motion_stability_score: float     # 0 to 100
    speed_uniformity_score: float     # 0 to 100
    shadow_clarity_score: float       # 0 to 100
    resurvey_recommended: bool        # True if quality falls below threshold
    resurvey_reasons: List[str]       # Specific triggers for resurvey
    spatial_coverage_status: str      # e.g., "Full 100% Swath Overlap", "Along-track Gap Risk"
    recommended_speed_knots: float    # Corrective advisory
    recommended_altitude_m: float     # Corrective advisory

    def to_dict(self) -> Dict[str, Union[float, str, bool, List[str]]]:
        return asdict(self)


def compute_scqi(
    image_bgr: np.ndarray,
    telemetry: Optional[Union[TelemetryRecord, List[TelemetryRecord], Dict[str, Any]]] = None,
    quality_metrics: Optional[QualityMetrics] = None,
    resurvey_threshold: float = 55.0,
) -> SCQIResult:
    reasons = []

    # Safely normalize telemetry input if passed as list or dict
    if isinstance(telemetry, list):
        telemetry = telemetry[0] if len(telemetry) > 0 else None
    elif isinstance(telemetry, dict):
        class _DictTelemetry:
            def __init__(self, d: Dict[str, Any]):
                self.altitude_m = float(d.get("altitude_m", d.get("altitude", 10.0)))
                self.slant_range_m = float(d.get("slant_range_m", d.get("slant_range", 75.0)))
                self.pitch_deg = float(d.get("pitch_deg", d.get("pitch", 0.0)))
                self.roll_deg = float(d.get("roll_deg", d.get("roll", 0.0)))
                self.heave_m = float(d.get("heave_m", d.get("heave", 0.0)))
                self.speed_knots = float(d.get("speed_knots", d.get("speed", 3.5)))
        telemetry = _DictTelemetry(telemetry)

    # 1. SNR and Signal Quality Assessment (Weight: 30%)
    if quality_metrics is None and image_bgr is not None and image_bgr.size > 0:
        quality_metrics = compute_snr_index(image_bgr)

    if quality_metrics is not None and quality_metrics.is_valid:
        snr_db = quality_metrics.snr_db
        snr_score = float(np.clip((snr_db - 2.0) / (16.0 - 2.0) * 100.0, 0.0, 100.0))
        if snr_score < 40.0:
            reasons.append(f"Degraded acoustic SNR ({snr_db:.1f} dB); high ambient seabed clutter / turbidity.")
    else:
        snr_score = 30.0
        reasons.append("Acoustic SNR could not be verified or is critically degraded.")

    # 2. Towfish Altitude Compliance (Weight: 25%)
    alt = telemetry.altitude_m if telemetry is not None else 10.0
    slant = telemetry.slant_range_m if telemetry is not None else 75.0
    alt_ratio = (alt / max(1.0, slant)) * 100.0

    optimal_min_ratio = 10.0
    optimal_max_ratio = 20.0
    rec_alt = float(slant * 0.15)

    if optimal_min_ratio <= alt_ratio <= optimal_max_ratio:
        altitude_score = 100.0
    elif alt_ratio < optimal_min_ratio:
        altitude_score = float(max(10.0, 100.0 - (optimal_min_ratio - alt_ratio) * 12.0))
        reasons.append(f"Towfish altitude too low ({alt:.1f}m, {alt_ratio:.1f}% of range); acoustic grazing angle suboptimal.")
    else:
        altitude_score = float(max(10.0, 100.0 - (alt_ratio - optimal_max_ratio) * 6.0))
        reasons.append(f"Towfish altitude too high ({alt:.1f}m, {alt_ratio:.1f}% of range); acoustic shadow length compressed.")

    # 3. Motion Data Quality (Weight: 20%)
    pitch = abs(telemetry.pitch_deg) if telemetry is not None else 0.0
    roll = abs(telemetry.roll_deg) if telemetry is not None else 0.0
    heave = abs(telemetry.heave_m) if telemetry is not None else 0.0

    motion_penalty = (pitch * 8.0) + (roll * 6.0) + (heave * 20.0)
    motion_score = float(np.clip(100.0 - motion_penalty, 0.0, 100.0))
    if motion_score < 50.0:
        reasons.append(f"High vehicle attitude disturbance (Pitch: {pitch:.1f}deg, Roll: {roll:.1f}deg, Heave: {heave:.2f}m); causes cross-track beam distortion.")

    # 4. Survey Speed Uniformity (Weight: 15%)
    speed = telemetry.vessel_speed_knots if telemetry is not None else 3.5
    rec_speed = 3.5

    if 3.0 <= speed <= 4.5:
        speed_score = 100.0
    elif speed > 4.5:
        speed_score = float(max(15.0, 100.0 - (speed - 4.5) * 22.0))
        if speed > 5.5:
            reasons.append(f"Survey speed excessive ({speed:.1f} knots > 4.5 knots); risk of along-track ping gaps.")
    else:
        speed_score = float(max(30.0, 100.0 - (3.0 - speed) * 25.0))
        if speed < 2.0:
            reasons.append(f"Survey speed low ({speed:.1f} knots); potential towfish crabbing.")

    # 5. Acoustic Shadow Clarity and Dynamic Contrast (Weight: 10%)
    if image_bgr is not None and image_bgr.size > 0:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
        p10 = float(np.percentile(gray, 10))
        p90 = float(np.percentile(gray, 90))
        contrast_span = p90 - p10
        shadow_score = float(np.clip((contrast_span / 140.0) * 100.0, 0.0, 100.0))
        if shadow_score < 35.0:
            reasons.append("Low acoustic shadow contrast; seabed highlights blend into backscatter.")
    else:
        shadow_score = 50.0

    overall_score = float(
        0.30 * snr_score +
        0.25 * altitude_score +
        0.20 * motion_score +
        0.15 * speed_score +
        0.10 * shadow_score
    )
    overall_score = round(float(np.clip(overall_score, 0.0, 100.0)), 1)

    resurvey_needed = overall_score < resurvey_threshold or len(reasons) >= 3

    if overall_score >= 85.0:
        grade = "EXCELLENT"
        coverage_status = "Nominal Hydrographic Survey Quality (100% Swath Overlap)"
    elif overall_score >= 70.0:
        grade = "GOOD"
        coverage_status = "Acceptable Survey Quality (<5% Edge Distortion)"
    elif overall_score >= 55.0:
        grade = "ACCEPTABLE"
        coverage_status = "Sub-Optimal Swath (Minor Motion Jitter)"
    elif overall_score >= 40.0:
        grade = "DEGRADED"
        coverage_status = "Severe Acoustic Degradation (Resurvey Advised)"
    else:
        grade = "RESURVEY_REQUIRED"
        coverage_status = "Critical Data Dropout / Blind Zone (Resurvey Mandatory)"

    return SCQIResult(
        overall_score=overall_score,
        grade=grade,
        snr_score=round(snr_score, 1),
        altitude_compliance_score=round(altitude_score, 1),
        motion_stability_score=round(motion_score, 1),
        speed_uniformity_score=round(speed_score, 1),
        shadow_clarity_score=round(shadow_score, 1),
        resurvey_recommended=resurvey_needed,
        resurvey_reasons=reasons,
        spatial_coverage_status=coverage_status,
        recommended_speed_knots=rec_speed,
        recommended_altitude_m=round(rec_alt, 1),
    )
