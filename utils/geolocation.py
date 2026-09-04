"""
Side-Scan Sonar Acoustic Geolocation & Spatial Post-Processing Engine.
Implements:
  1. Acoustic Ray-Tracing Transform: Pixel (u, v) -> WGS-84 Coordinates (Lat, Lon)
  2. Covariance-Based 95% Position Error Ellipses
  3. Cross-Track Spatial Deduplication & Clustering
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from utils.telemetry_parser import TelemetryRecord


@dataclass
class GeolocationEstimate:
    latitude: float
    longitude: float
    ground_range_m: float
    channel: str                       # 'Port' or 'Starboard'
    error_ellipse_semi_major_m: float  # a (95% confidence radius along major axis)
    error_ellipse_semi_minor_m: float  # b (95% confidence radius along minor axis)
    error_ellipse_orientation_deg: float # phi (orientation angle in degrees)
    towfish_lat: float
    towfish_lon: float

    def to_dict(self) -> Dict[str, Union[float, str]]:
        return asdict(self)


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two GPS coordinates in meters.
    """
    r_earth = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_earth * c


def compute_error_ellipse_95(
    range_m: float,
    beamwidth_deg: float = 0.5,
    gps_accuracy_m: float = 2.5,
    heading_deg: float = 45.0,
    detection_conf: float = 0.85
) -> Tuple[float, float, float]:
    """
    Computes 95% confidence error ellipse parameters (a, b, phi) based on
    GPS error covariance, acoustic along-track beam spreading, and slant-range resolution.

    Returns:
        (semi_major_axis_m, semi_minor_axis_m, orientation_deg)
    """
    # Along-track cross-beam uncertainty: delta_x = range * sin(beamwidth)
    beam_spread_m = range_m * math.sin(math.radians(beamwidth_deg))

    # Cross-track range resolution uncertainty (~0.05m base + confidence scaling)
    conf_factor = max(1.0, (1.05 - detection_conf) * 2.0)
    range_res_m = 0.25 * conf_factor

    # Total 1-sigma positional standard deviations
    sigma_along = math.sqrt(gps_accuracy_m**2 + beam_spread_m**2)
    sigma_across = math.sqrt(gps_accuracy_m**2 + range_res_m**2)

    # 95% confidence scale factor for 2-DOF Gaussian (chi-squared quantile sqrt(5.991) = 2.4477)
    k_95 = 2.4477
    a_95 = round(sigma_along * k_95, 2)
    b_95 = round(sigma_across * k_95, 2)

    # Orientation aligns with acoustic beam angle (perpendicular to flight heading)
    orientation_deg = round((heading_deg + 90.0) % 360.0, 1)

    return a_95, b_95, orientation_deg


def project_pixel_to_latlon(
    u_col: float,
    v_row: float,
    image_shape: Tuple[int, int, ...],
    telemetry: TelemetryRecord,
    nadir_col: Optional[float] = None
) -> GeolocationEstimate:
    """
    Ray-Tracing Geolocation Transform:
    Maps pixel coordinate (u_col, v_row) on side-scan sonar image to true WGS-84 (Lat, Lon)
    using towfish altitude, heading, layback, and slant range.
    """
    h_img, w_img = image_shape[:2]
    mid_col = nadir_col if nadir_col is not None else (w_img / 2.0)

    # 1. Towfish Position (Vessel GPS compensated for cable layback behind vessel)
    head_rad = math.radians(telemetry.heading_deg)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(telemetry.latitude))

    # Towfish layback vector (opposite to vessel heading)
    dx_layback = -telemetry.layback_m * math.sin(head_rad)
    dy_layback = -telemetry.layback_m * math.cos(head_rad)

    towfish_lat = telemetry.latitude + (dy_layback / m_per_deg_lat)
    towfish_lon = telemetry.longitude + (dx_layback / m_per_deg_lon)

    # 2. Cross-Track Ground Range Calculation
    is_starboard = (u_col >= mid_col)
    channel = "Starboard" if is_starboard else "Port"

    # Normalized cross-track pixel fraction from nadir [0, 1]
    dist_from_nadir_px = abs(u_col - mid_col)
    max_half_width = max(1.0, w_img - mid_col if is_starboard else mid_col)
    frac_range = min(1.0, dist_from_nadir_px / max_half_width)

    # Slant range to ground range conversion: Rg = sqrt(max(0, Rs^2 - h^2))
    rs_m = frac_range * telemetry.slant_range_m
    rg_m = math.sqrt(max(0.0, rs_m**2 - telemetry.altitude_m**2))

    # 3. Across-Track Acoustic Normal Vector
    # Port beam is heading - 90 deg; Starboard beam is heading + 90 deg
    beam_bearing_deg = (telemetry.heading_deg + 90.0) if is_starboard else (telemetry.heading_deg - 90.0)
    beam_rad = math.radians(beam_bearing_deg % 360.0)

    dx_target = rg_m * math.sin(beam_rad)
    dy_target = rg_m * math.cos(beam_rad)

    target_lat = towfish_lat + (dy_target / m_per_deg_lat)
    target_lon = towfish_lon + (dx_target / m_per_deg_lon)

    # 4. 95% Position Error Ellipse
    a_95, b_95, phi_deg = compute_error_ellipse_95(
        range_m=rg_m,
        beamwidth_deg=telemetry.beamwidth_deg,
        gps_accuracy_m=2.5,
        heading_deg=telemetry.heading_deg
    )

    return GeolocationEstimate(
        latitude=round(target_lat, 6),
        longitude=round(target_lon, 6),
        ground_range_m=round(rg_m, 2),
        channel=channel,
        error_ellipse_semi_major_m=a_95,
        error_ellipse_semi_minor_m=b_95,
        error_ellipse_orientation_deg=phi_deg,
        towfish_lat=round(towfish_lat, 6),
        towfish_lon=round(towfish_lon, 6)
    )


def spatial_clustering_deduplication(
    detections: List[Dict[str, any]],
    distance_threshold_m: float = 4.5
) -> List[Dict[str, any]]:
    """
    Spatial post-processing to deduplicate multiple sightings of the same debris
    across overlapping side-scan survey swaths.
    """
    if len(detections) <= 1:
        return detections

    consolidated = []
    used = set()

    for i in range(len(detections)):
        if i in used:
            continue

        cluster = [detections[i]]
        used.add(i)
        lat1 = detections[i].get("latitude")
        lon1 = detections[i].get("longitude")

        if lat1 is not None and lon1 is not None:
            for j in range(i + 1, len(detections)):
                if j in used:
                    continue
                lat2 = detections[j].get("latitude")
                lon2 = detections[j].get("longitude")

                if lat2 is not None and lon2 is not None:
                    dist = haversine_distance_m(lat1, lon1, lat2, lon2)
                    if dist <= distance_threshold_m:
                        cluster.append(detections[j])
                        used.add(j)

        # Merge cluster: pick highest confidence detection, fuse sightings
        best_det = max(cluster, key=lambda d: d.get("conf", 0.0)).copy()
        best_det["sightings_count"] = len(cluster)
        best_det["fused_confidence"] = round(float(np.mean([d.get("conf", 0.0) for d in cluster])), 3)
        consolidated.append(best_det)

    return consolidated
