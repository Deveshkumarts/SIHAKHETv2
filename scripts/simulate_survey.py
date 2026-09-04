"""
Autonomous Side-Scan Sonar Multi-Track Survey Simulator.
Simulates a multi-kilometer marine survey with realistic towfish hydrodynamics,
generates debris sightings with acoustic shadows, executes the full Akhet pipeline,
and populates the spatial database for rich GIS heatmap demonstration.
"""

import time
import math
import sys
from pathlib import Path
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from utils.telemetry_parser import generate_synthetic_telemetry
from utils.sonar_calibration import calibrate_side_scan_sonar
from utils.geolocation import project_pixel_to_latlon, spatial_clustering_deduplication
from utils.db_store import SurveyDatabase
from utils.feedback_loop import ActiveLearningManager


def run_survey_simulation(
    num_pings: int = 35,
    base_lat: float = 13.0827,
    base_lon: float = 80.2707,
    survey_speed_knots: float = 4.0,
    heading_deg: float = 65.0,
    altitude_m: float = 10.0,
    slant_range_m: float = 75.0,
    mission_id: str = "mission_bay_of_bengal_01"
):
    print(f"🌊 Starting Autonomous Survey Simulator: {mission_id}")
    print(f"   Trajectory: {num_pings} pings | Base: ({base_lat}°N, {base_lon}°E) | Heading: {heading_deg}°")

    db = SurveyDatabase()
    db.record_mission(mission_id, mission_name="Bay of Bengal Coastal Debris Survey", vessel_name="RV Akhet Explorer")
    al_mgr = ActiveLearningManager()

    # Generate sequential towfish trajectory
    telemetry_stream = generate_synthetic_telemetry(
        num_pings=num_pings,
        base_lat=base_lat,
        base_lon=base_lon,
        speed_knots=survey_speed_knots,
        heading_deg=heading_deg,
        altitude_m=altitude_m,
        slant_range_m=slant_range_m
    )

    all_mission_detections = []
    classes_pool = [
        "bottle", "plastic_bag", "tire", "can", "net",
        "metal_container", "pipe", "rope", "wood", "valve"
    ]

    for ping_idx, telem in enumerate(telemetry_stream):
        # 1. Synthesize acoustic sonar image
        img = np.random.normal(85, 12, (512, 1024)).clip(0, 255).astype(np.uint8)
        # Add central water-column band
        img[:, 480:544] = np.random.normal(15, 4, (512, 64)).clip(0, 255).astype(np.uint8)

        # Probabilistically inject 1-2 debris items into this ping
        ping_dets = []
        if ping_idx % 3 == 0 or ping_idx % 5 == 0:
            num_items = np.random.choice([1, 2], p=[0.7, 0.3])
            for _ in range(num_items):
                is_starboard = (np.random.rand() > 0.5)
                if is_starboard:
                    cx = int(np.random.uniform(580, 950))
                    # Shadow is to the right
                    img[240:270, cx:cx+20] = 235  # Highlight
                    img[240:270, cx+20:cx+60] = 12 # Acoustic Shadow
                else:
                    cx = int(np.random.uniform(70, 440))
                    # Shadow is to the left
                    img[240:270, cx:cx+20] = 235  # Highlight
                    img[240:270, cx-40:cx] = 12   # Acoustic Shadow

                cy = int(np.random.uniform(100, 412))
                cname = str(np.random.choice(classes_pool))
                conf = float(np.random.uniform(0.75, 0.96))

                # Ray-trace to WGS-84 coordinate
                geo = project_pixel_to_latlon(
                    u_col=cx, v_row=cy, image_shape=img.shape, telemetry=telem
                )

                # Uncertainty metrics
                variance = round(float(np.random.exponential(0.012)), 4)
                unc_flag = "LOW" if variance < 0.02 else ("MODERATE" if variance < 0.045 else "HIGH")

                det_record = {
                    "class_name": cname,
                    "conf": round(conf, 3),
                    "bbox": [cx - 15, cy - 15, cx + 15, cy + 15],
                    "latitude": geo.latitude,
                    "longitude": geo.longitude,
                    "ground_range_m": geo.ground_range_m,
                    "channel": geo.channel,
                    "error_ellipse_a": geo.error_ellipse_semi_major_m,
                    "error_ellipse_b": geo.error_ellipse_semi_minor_m,
                    "error_ellipse_phi": geo.error_ellipse_orientation_deg,
                    "uncertainty_variance": variance,
                    "uncertainty_flag": unc_flag
                }
                ping_dets.append(det_record)
                all_mission_detections.append(det_record)

                if unc_flag == "HIGH":
                    crop = img[max(0, cy-30):min(512, cy+30), max(0, cx-30):min(1024, cx+30)]
                    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
                    al_mgr.enqueue_for_review(det_record, crop_bgr, reason=f"Epistemic Variance: {variance}")

    # Spatial deduplication across overlapping pings
    deduped = spatial_clustering_deduplication(all_mission_detections, distance_threshold_m=4.5)
    db.save_detections(deduped, mission_id=mission_id)

    print(f"✅ Simulation Complete!")
    print(f"   • Total Pings Processed: {num_pings}")
    print(f"   • Total Sighted Targets: {len(all_mission_detections)}")
    print(f"   • Deduplicated Unique Debris Clusters: {len(deduped)}")
    print(f"   • Saved to Spatial Database ({db.db_path})")
    return deduped


if __name__ == "__main__":
    run_survey_simulation()
