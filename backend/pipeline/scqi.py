"""Stage 12 - SCQI survey-quality score (per frame and overall)."""

from __future__ import annotations

from typing import List

import numpy as np

from utils.scqi_engine import SCQIResult, compute_scqi
from utils.sonar_calibration import QualityMetrics
from utils.telemetry_parser import TelemetryRecord


def run_scqi(image_bgr: np.ndarray, telemetry: List[TelemetryRecord], quality: QualityMetrics, cfg) -> SCQIResult:
    return compute_scqi(image_bgr, telemetry=telemetry if len(telemetry) != 1 else telemetry[0], quality_metrics=quality,
                        resurvey_threshold=float(cfg.scqi.resurvey_threshold))
