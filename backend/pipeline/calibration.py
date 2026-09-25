"""Stage 3 - sonar calibration & correction: TVG, water-column removal, slant->ground range, beam pattern."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from utils.sonar_calibration import calibrate_side_scan_sonar
from utils.telemetry_parser import TelemetryRecord


def run_calibration(image_bgr: np.ndarray, telemetry: TelemetryRecord | None, cfg) -> Tuple[np.ndarray, Dict[str, Any]]:
    c = cfg.calibration
    if not c.enable:
        return image_bgr, {"applied": False}
    altitude = telemetry.altitude_m if telemetry is not None else c.altitude_m
    slant = telemetry.slant_range_m if telemetry is not None else c.slant_range_m
    out, report = calibrate_side_scan_sonar(
        image_bgr,
        altitude_m=float(altitude),
        slant_range_m=float(slant),
        enable_tvg=c.enable_tvg,
        enable_wcr=c.enable_wcr,
        enable_src=c.enable_src,
        enable_beam_correction=c.enable_beam_correction,
    )
    clean = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in report.items()}
    clean["applied"] = True
    clean["altitude_m"] = float(altitude)
    clean["slant_range_m"] = float(slant)
    return out, clean
