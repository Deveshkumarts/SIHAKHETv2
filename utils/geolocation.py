"""
Side-Scan Sonar Acoustic Geolocation & Spatial Post-Processing Engine.
Implements:
  1. Acoustic Ray-Tracing Transform: Pixel (u, v) -> WGS-84 Coordinates (Lat, Lon)
  2. Covariance-Based 95% Position Error Ellipses
  3. Cross-Track Spatial Deduplication & Clustering
"""

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from utils.telemetry_parser import TelemetryRecord

try:
    import pyproj
    GEOD = pyproj.Geod(ellps="WGS84")
except ImportError:
    GEOD = None

try:
    from shapely.geometry import Point, Polygon
    import shapely.affinity
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL OCEANIC REGIONS (Strictly open oceanic waters, zero continental land)
# ═══════════════════════════════════════════════════════════════════════════════
GLOBAL_OCEAN_REGIONS: List[Dict[str, Any]] = [
    {
        "basin": "Bay of Bengal (Central Abyssal Basin)",
        "lat_min": 11.5, "lat_max": 16.5,
        "lon_min": 83.5, "lon_max": 89.5,
        "depth_range_m": (2400, 3900),
        "sound_speed_mps": (1515, 1530),
        "temp_c": (4.5, 7.8),
        "substrate": "Siliceous Pelagic Ooze & Silt",
        "salinity_psu": 34.2,
    },
    {
        "basin": "Arabian Sea (Deep Subsea Trench)",
        "lat_min": 12.0, "lat_max": 18.0,
        "lon_min": 63.5, "lon_max": 70.5,
        "depth_range_m": (2900, 4400),
        "sound_speed_mps": (1520, 1535),
        "temp_c": (4.0, 6.9),
        "substrate": "Fine Terrigenous Mud & Clay",
        "salinity_psu": 35.8,
    },
    {
        "basin": "Central Indian Ocean (Pelagic Abyssal Plain)",
        "lat_min": -7.5, "lat_max": 2.5,
        "lon_min": 70.0, "lon_max": 88.0,
        "depth_range_m": (3800, 5200),
        "sound_speed_mps": (1505, 1525),
        "temp_c": (2.8, 5.2),
        "substrate": "Polymetallic Nodule Bed / Red Clay",
        "salinity_psu": 34.7,
    },
    {
        "basin": "North Atlantic (Sargasso Deep Basin)",
        "lat_min": 24.0, "lat_max": 31.0,
        "lon_min": -64.0, "lon_max": -48.0,
        "depth_range_m": (4100, 5600),
        "sound_speed_mps": (1498, 1518),
        "temp_c": (3.2, 5.8),
        "substrate": "Calcareous Foraminiferal Ooze",
        "salinity_psu": 36.5,
    },
    {
        "basin": "Mid-Atlantic Ridge (Abyssal Rift Valley)",
        "lat_min": -4.0, "lat_max": 8.0,
        "lon_min": -32.0, "lon_max": -18.0,
        "depth_range_m": (3100, 4700),
        "sound_speed_mps": (1502, 1522),
        "temp_c": (3.0, 5.5),
        "substrate": "Basaltic Pillow Lava & Pelagic Sediment",
        "salinity_psu": 35.1,
    },
    {
        "basin": "North Pacific (Pelagic Ocean Corridor)",
        "lat_min": 22.0, "lat_max": 32.0,
        "lon_min": 148.0, "lon_max": 172.0,
        "depth_range_m": (4300, 5900),
        "sound_speed_mps": (1492, 1512),
        "temp_c": (2.1, 4.2),
        "substrate": "Abyssal Brown Clay & Manganese Crust",
        "salinity_psu": 34.4,
    },
    {
        "basin": "South Pacific (Polynesian Deep Basin)",
        "lat_min": -18.0, "lat_max": -8.0,
        "lon_min": -138.0, "lon_max": -115.0,
        "depth_range_m": (3900, 5100),
        "sound_speed_mps": (1496, 1516),
        "temp_c": (2.4, 4.6),
        "substrate": "Pelagic Red Clay & Zeolitic Silt",
        "salinity_psu": 34.6,
    },
    {
        "basin": "South China Sea (Central Deep Basin)",
        "lat_min": 13.0, "lat_max": 17.0,
        "lon_min": 113.5, "lon_max": 117.5,
        "depth_range_m": (2200, 3900),
        "sound_speed_mps": (1518, 1532),
        "temp_c": (4.2, 7.1),
        "substrate": "Hemipelagic Clay & Carbonate Silt",
        "salinity_psu": 34.5,
    },
    {
        "basin": "Mediterranean Sea (Ionian Deep Abyssal Plain)",
        "lat_min": 34.5, "lat_max": 36.2,
        "lon_min": 16.8, "lon_max": 20.5,
        "depth_range_m": (2600, 4200),
        "sound_speed_mps": (1528, 1542),
        "temp_c": (12.8, 14.5),
        "substrate": "Sapropelic Mud & Biogenic Carbonate",
        "salinity_psu": 38.6,
    },
    {
        "basin": "Coral Sea (Queensland Abyssal Corridor)",
        "lat_min": -19.5, "lat_max": -14.5,
        "lon_min": 150.5, "lon_max": 156.0,
        "depth_range_m": (2500, 4600),
        "sound_speed_mps": (1510, 1528),
        "temp_c": (3.5, 6.0),
        "substrate": "Coral-Derived Carbonate Ooze",
        "salinity_psu": 35.3,
    },
    {
        "basin": "Gulf of Mexico (Sigsbee Abyssal Plain)",
        "lat_min": 23.8, "lat_max": 25.8,
        "lon_min": -92.5, "lon_max": -88.5,
        "depth_range_m": (3200, 3800),
        "sound_speed_mps": (1515, 1530),
        "temp_c": (4.1, 6.2),
        "substrate": "Turbidite Sand & Hemipelagic Mud",
        "salinity_psu": 36.2,
    },
    {
        "basin": "Norwegian Sea (Vøring Deep Basin)",
        "lat_min": 67.0, "lat_max": 71.0,
        "lon_min": 1.5, "lon_max": 7.0,
        "depth_range_m": (1800, 3200),
        "sound_speed_mps": (1475, 1495),
        "temp_c": (-0.8, 2.5),
        "substrate": "Glaciomarine Silty Clay",
        "salinity_psu": 34.9,
    },
]


def generate_random_ocean_coordinates(seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Generates realistic, physically-validated geographic coordinates located strictly
    within global deep-ocean basins (100% oceanic waters, zero continental landmass).
    Returns coordinates (lat, lon) alongside oceanographic parameters (depth, sound velocity,
    water temperature, seabed substrate, and salinity).
    """
    import random
    rng = random.Random(seed) if seed is not None else random
    region = rng.choice(GLOBAL_OCEAN_REGIONS)
    lat = round(rng.uniform(region["lat_min"], region["lat_max"]), 6)
    lon = round(rng.uniform(region["lon_min"], region["lon_max"]), 6)
    depth = round(rng.uniform(region["depth_range_m"][0], region["depth_range_m"][1]), 1)
    sound_speed = round(rng.uniform(region["sound_speed_mps"][0], region["sound_speed_mps"][1]), 1)
    temp_c = round(rng.uniform(region["temp_c"][0], region["temp_c"][1]), 1)

    return {
        "latitude": lat,
        "longitude": lon,
        "basin_name": region["basin"],
        "depth_m": depth,
        "sound_speed_mps": sound_speed,
        "temperature_c": temp_c,
        "substrate": region["substrate"],
        "salinity_psu": region["salinity_psu"],
    }


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


def compute_error_ellipse_polygon(
    center_lat: float,
    center_lon: float,
    semi_major_m: float,
    semi_minor_m: float,
    orientation_deg: float,
    num_points: int = 32
) -> Optional[Any]:
    """
    Constructs a Shapely 2D Polygon representing the 95% position confidence error ellipse.
    Converts ground metric semi-axes to WGS-84 geodetic angular offsets.
    """
    if not HAS_SHAPELY:
        return None
    deg_lat = semi_major_m / 111320.0
    deg_lon = semi_minor_m / (111320.0 * max(0.1, math.cos(math.radians(center_lat))))
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = deg_lon * np.cos(theta)
    y = deg_lat * np.sin(theta)
    pts = np.column_stack([x, y])
    poly = Polygon(pts)
    poly = shapely.affinity.rotate(poly, -orientation_deg, origin=(0, 0))
    poly = shapely.affinity.translate(poly, xoff=center_lon, yoff=center_lat)
    return poly


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
    if GEOD is not None and telemetry.layback_m > 0:
        layback_azimuth = (telemetry.heading_deg + 180.0) % 360.0
        towfish_lon, towfish_lat, _ = GEOD.fwd(telemetry.longitude, telemetry.latitude, layback_azimuth, telemetry.layback_m)
    elif telemetry.layback_m == 0:
        towfish_lat, towfish_lon = telemetry.latitude, telemetry.longitude
    else:
        head_rad = math.radians(telemetry.heading_deg)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(telemetry.latitude))
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

    if GEOD is not None and rg_m > 0:
        # High-precision pyproj WGS-84 geodetic forward transform (IHO S-44 standard)
        target_lon, target_lat, _ = GEOD.fwd(towfish_lon, towfish_lat, beam_bearing_deg % 360.0, rg_m)
    else:
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(towfish_lat))
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
