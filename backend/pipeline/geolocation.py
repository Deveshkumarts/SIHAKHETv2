"""Stage 11 - geolocation (WGS-84) with 95% positional uncertainty, per object."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.pipeline.fusion import FusedObject
from utils.geolocation import project_pixel_to_latlon
from utils.telemetry_parser import TelemetryRecord


def telemetry_for_row(telemetry: List[TelemetryRecord], row: float, n_rows: int) -> TelemetryRecord:
    """Ping telemetry for an image row (rows of the waterfall are pings)."""
    if len(telemetry) == 1 or n_rows <= 1:
        return telemetry[0]
    idx = int(min(len(telemetry) - 1, max(0, round(row / (n_rows - 1) * (len(telemetry) - 1)))))
    return telemetry[idx]


def geolocate(objects: List[FusedObject], telemetry: List[TelemetryRecord], full_shape, frame_offset_y: float,
              ) -> None:
    """Fill lat/lon + uncertainty on each object in place (global pixel coordinates)."""
    for o in objects:
        cx, cy = o.centroid
        gy = cy + frame_offset_y
        tel = telemetry_for_row(telemetry, gy, full_shape[0])
        est = project_pixel_to_latlon(cx, gy, full_shape, tel)
        o.latitude, o.longitude = float(est.latitude), float(est.longitude)
        o.uncertainty_m = float(est.error_ellipse_semi_major_m)
        o.uncertainty_minor_m = float(est.error_ellipse_semi_minor_m)
        o.ground_range_m = float(est.ground_range_m)
        o.channel = est.channel
        o.timestamp = float(tel.timestamp)
