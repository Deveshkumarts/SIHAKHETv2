"""Stage 2 - data quality control. Runs BEFORE any enhancement so bad data is flagged, not "fixed" silently."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import cv2
import numpy as np

from utils.sonar_calibration import compute_snr_index
from utils.telemetry_parser import TelemetryRecord, TelemetryValidator


@dataclass
class QCCheck:
    name: str
    passed: bool
    value: float
    limit: str
    detail: str = ""


@dataclass
class QCReport:
    passed: bool
    checks: List[QCCheck] = field(default_factory=list)
    ping_dropout_ratio: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def ping_dropout_ratio(gray: np.ndarray, dead_mean: float = 2.0) -> float:
    """Fraction of ping lines (rows of the waterfall) that are effectively dead (mean intensity ~ 0)."""
    if gray.size == 0:
        return 1.0
    row_mean = gray.reshape(gray.shape[0], -1).mean(axis=1)
    return float(np.mean(row_mean < dead_mean))


def run_quality_control(image_bgr: np.ndarray, telemetry: List[TelemetryRecord], cfg) -> QCReport:
    q = cfg.quality_control
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    m = compute_snr_index(image_bgr)
    drop = ping_dropout_ratio(gray)

    checks = [
        QCCheck("image_not_empty", image_bgr.size > 0, float(image_bgr.size), "> 0"),
        QCCheck("zero_ratio", m.zero_ratio <= q.max_zero_ratio, round(m.zero_ratio, 4), f"<= {q.max_zero_ratio}",
                "black samples / dead channel"),
        QCCheck("saturation_ratio", m.saturation_ratio <= q.max_saturation_ratio, round(m.saturation_ratio, 4),
                f"<= {q.max_saturation_ratio}", "gain too high"),
        QCCheck("dynamic_range_db", m.dynamic_range_db >= q.min_dynamic_range_db, round(m.dynamic_range_db, 2),
                f">= {q.min_dynamic_range_db}"),
        QCCheck("ping_dropout", drop <= q.max_zero_ratio, round(drop, 4), f"<= {q.max_zero_ratio}",
                "fraction of dead ping lines"),
    ]

    warnings = list(m.warnings)
    if telemetry:
        t = telemetry[len(telemetry) // 2]
        ok_rec, w = TelemetryValidator.validate_record(t)
        warnings += w
        checks += [
            QCCheck("telemetry_record", ok_rec, float(ok_rec), "valid"),
            QCCheck("altitude_m", q.min_altitude_m <= t.altitude_m <= q.max_altitude_m, round(t.altitude_m, 2),
                    f"[{q.min_altitude_m}, {q.max_altitude_m}]"),
            QCCheck("pitch_roll_deg", max(abs(t.pitch_deg), abs(t.roll_deg)) <= q.max_pitch_roll_deg,
                    round(max(abs(t.pitch_deg), abs(t.roll_deg)), 2), f"<= {q.max_pitch_roll_deg}"),
        ]
        if len(telemetry) > 1:
            ok_traj, w2 = TelemetryValidator.validate_trajectory(telemetry)
            warnings += w2
            checks.append(QCCheck("trajectory_consistency", ok_traj, float(ok_traj), "consistent"))

    return QCReport(passed=all(c.passed for c in checks), checks=checks, ping_dropout_ratio=round(drop, 4),
                    warnings=warnings)
