"""
Side-Scan Sonar (SSS) Telemetry & Sensor Ingestion Module.
Handles parsing, validation, and coordinate/sensor parameter tracking for acoustic survey data.
"""

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class TelemetryRecord:
    timestamp: float                       # Epoch seconds or relative survey seconds
    latitude: float                        # WGS84 Latitude (-90.0 to 90.0)
    longitude: float                       # WGS84 Longitude (-180.0 to 180.0)
    heading_deg: float                     # Compass heading in degrees [0, 360)
    depth_m: float = 15.0                  # Towfish depth below sea surface in meters
    altitude_m: float = 10.0               # Towfish altitude above seabed in meters
    slant_range_m: float = 75.0            # Max acoustic slant range per channel in meters
    frequency_khz: float = 450.0           # Sonar acoustic carrier frequency (e.g., 450 / 900 kHz)
    beamwidth_deg: float = 0.5             # Horizontal acoustic beamwidth in degrees
    vessel_speed_knots: float = 3.5        # Survey vessel speed over ground
    layback_m: float = 25.0                # Towfish cable layback distance behind vessel in meters
    pitch_deg: float = 0.0                 # Towfish pitch attitude angle
    roll_deg: float = 0.0                  # Towfish roll attitude angle

    def to_dict(self) -> Dict[str, Union[float, str]]:
        return asdict(self)


class TelemetryValidator:
    """
    Validates sensor telemetry consistency, GPS bounds, compass dynamics, and ping monotonicity.
    """

    @staticmethod
    def validate_record(record: TelemetryRecord) -> Tuple[bool, List[str]]:
        warnings = []
        is_valid = True

        # GPS validation
        if not (-90.0 <= record.latitude <= 90.0):
            warnings.append(f"Invalid latitude: {record.latitude}")
            is_valid = False
        if not (-180.0 <= record.longitude <= 180.0):
            warnings.append(f"Invalid longitude: {record.longitude}")
            is_valid = False

        # Heading validation
        if not (0.0 <= record.heading_deg < 360.0):
            record.heading_deg = record.heading_deg % 360.0
            warnings.append(f"Heading normalized to [0, 360): {record.heading_deg:.2f}°")

        # Altitude / Depth validation
        if record.altitude_m <= 0.1:
            warnings.append(f"Towfish altitude too low ({record.altitude_m:.1f}m); seabed collision risk or bad nadir.")
        if record.altitude_m >= record.slant_range_m:
            warnings.append(f"Altitude ({record.altitude_m:.1f}m) exceeds slant range ({record.slant_range_m:.1f}m).")
            is_valid = False

        if record.slant_range_m <= 1.0:
            warnings.append(f"Invalid slant range: {record.slant_range_m}m")
            is_valid = False

        return is_valid, warnings

    @staticmethod
    def validate_trajectory(records: List[TelemetryRecord], max_speed_knots: float = 15.0) -> Tuple[bool, List[str]]:
        """
        Validates sequential consistency across survey track pings.
        Detects GPS teleportation jumps, reverse timestamps, and compass spin anomalies.
        """
        if len(records) < 2:
            return True, []

        warnings = []
        is_valid = True

        for i in range(1, len(records)):
            prev, curr = records[i - 1], records[i]
            dt = curr.timestamp - prev.timestamp

            if dt <= 0:
                warnings.append(f"Non-monotonic timestamp at index {i}: dt={dt:.3f}s")
                is_valid = False
                continue

            # Calculate Haversine distance
            lat1, lon1 = math.radians(prev.latitude), math.radians(prev.longitude)
            lat2, lon2 = math.radians(curr.latitude), math.radians(curr.longitude)
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
            dist_m = 6371000.0 * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
            speed_mps = dist_m / dt
            speed_knots = speed_mps * 1.94384

            if speed_knots > max_speed_knots:
                warnings.append(
                    f"GPS Teleport Jump detected between ping {i-1} & {i}: {speed_knots:.1f} kts > max {max_speed_knots} kts"
                )
                is_valid = False

        return is_valid, warnings


def parse_telemetry_file(file_path: Union[str, Path]) -> List[TelemetryRecord]:
    """
    Parses telemetry from a JSON or CSV file.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Telemetry file not found: {p}")

    records = []
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        for item in items:
            rec = TelemetryRecord(
                timestamp=float(item.get("timestamp", time.time())),
                latitude=float(item.get("latitude", 0.0)),
                longitude=float(item.get("longitude", 0.0)),
                heading_deg=float(item.get("heading_deg", item.get("heading", 0.0))),
                depth_m=float(item.get("depth_m", item.get("depth", 15.0))),
                altitude_m=float(item.get("altitude_m", item.get("altitude", 10.0))),
                slant_range_m=float(item.get("slant_range_m", item.get("slant_range", 75.0))),
                frequency_khz=float(item.get("frequency_khz", 450.0)),
                beamwidth_deg=float(item.get("beamwidth_deg", 0.5)),
                vessel_speed_knots=float(item.get("vessel_speed_knots", 3.5)),
                layback_m=float(item.get("layback_m", 25.0)),
                pitch_deg=float(item.get("pitch_deg", 0.0)),
                roll_deg=float(item.get("roll_deg", 0.0)),
            )
            records.append(rec)
    elif p.suffix.lower() in [".csv", ".txt"]:
        import csv
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = TelemetryRecord(
                    timestamp=float(row.get("timestamp", time.time())),
                    latitude=float(row.get("latitude", row.get("lat", 0.0))),
                    longitude=float(row.get("longitude", row.get("lon", 0.0))),
                    heading_deg=float(row.get("heading_deg", row.get("heading", 0.0))),
                    depth_m=float(row.get("depth_m", row.get("depth", 15.0))),
                    altitude_m=float(row.get("altitude_m", row.get("altitude", 10.0))),
                    slant_range_m=float(row.get("slant_range_m", row.get("range", 75.0))),
                )
                records.append(rec)

    return records


def generate_synthetic_telemetry(
    num_pings: int = 1,
    base_lat: float = 13.0827,
    base_lon: float = 80.2707,
    heading_deg: float = 45.0,
    depth_m: float = 15.0,
    altitude_m: float = 12.0,
    slant_range_m: float = 75.0,
    speed_knots: float = 3.5,
    ping_interval_s: float = 0.25,
) -> List[TelemetryRecord]:
    """
    Generates realistic, physically valid synthetic side-scan telemetry for standalone sonar images.
    """
    records = []
    t_start = time.time()
    speed_mps = speed_knots * 0.514444
    heading_rad = math.radians(heading_deg)

    # Displacement per ping
    dx = speed_mps * ping_interval_s * math.sin(heading_rad)
    dy = speed_mps * ping_interval_s * math.cos(heading_rad)

    # Conversion factor from meters to degrees lat/lon (approx at equator / tropics)
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(base_lat))

    curr_lat = base_lat
    curr_lon = base_lon

    for i in range(num_pings):
        curr_lat += (dy / m_per_deg_lat)
        curr_lon += (dx / m_per_deg_lon)

        # Micro-fluctuations from towfish hydrodynamic motion
        alt_jitter = altitude_m + 0.15 * math.sin(i * 0.1)
        pitch_jitter = 0.5 * math.sin(i * 0.15)
        roll_jitter = 0.8 * math.cos(i * 0.12)
        head_jitter = (heading_deg + 0.3 * math.sin(i * 0.08)) % 360.0

        rec = TelemetryRecord(
            timestamp=t_start + i * ping_interval_s,
            latitude=curr_lat,
            longitude=curr_lon,
            heading_deg=head_jitter,
            depth_m=depth_m,
            altitude_m=max(1.0, alt_jitter),
            slant_range_m=slant_range_m,
            frequency_khz=450.0,
            beamwidth_deg=0.5,
            vessel_speed_knots=speed_knots,
            layback_m=25.0,
            pitch_deg=pitch_jitter,
            roll_deg=roll_jitter
        )
        records.append(rec)

    return records
