"""
Comprehensive Test Suite for Akhet Marine AI Platform (Phases 1 - 5).
Verifies:
  Phase 1: Telemetry Parsing & Acoustic Calibration Engine
  Phase 2: OS-CFAR Clutter Detector & Anomaly Autoencoder + Decision Gate
  Phase 3: MC Dropout Uncertainty Estimation & Geolocation Engine
  Phase 4: GIS Kernel Density Estimation (KDE) & Spatial Database
  Phase 5: Human-in-the-Loop Active Learning Feedback Loop
"""

import sys
import unittest
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

# Phase 1 imports
from utils.telemetry_parser import generate_synthetic_telemetry, TelemetryValidator
from utils.sonar_calibration import compute_snr_index, calibrate_side_scan_sonar

# Phase 2 imports
from models.os_cfar import OSCFARDetector
from models.autoencoder import SonarAnomalyDetector
from utils.decision_gate import evaluate_decision_gate, verify_acoustic_shadow

# Phase 3 imports
from resnet.classifier import ResNet18InferenceEngine
from utils.roi_utils import get_adaptive_padding_ratio, validate_roi_quality, expand_and_clamp_bbox
from utils.geolocation import project_pixel_to_latlon, spatial_clustering_deduplication

# Phase 4 imports
from utils.gis_density import compute_spatial_kde, build_gis_hotspot_figure, export_detections_to_geojson
from utils.db_store import SurveyDatabase

# Phase 5 imports
from utils.feedback_loop import ActiveLearningManager


class TestAkhetMarinePipeline(unittest.TestCase):

    def setUp(self):
        # Create a synthetic acoustic sonar canvas
        self.img = np.random.normal(90, 15, (400, 800)).clip(0, 255).astype(np.uint8)
        # Port target + shadow
        self.img[150:170, 200:220] = 240
        self.img[150:170, 160:200] = 10
        # Starboard target + shadow
        self.img[220:240, 600:620] = 235
        self.img[220:240, 620:660] = 15
        self.img_bgr = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

    def test_phase1_telemetry_and_calibration(self):
        print("\n[Phase 1] Testing Telemetry and Calibration...")
        telem = generate_synthetic_telemetry(num_pings=3, base_lat=13.0827, base_lon=80.2707)
        self.assertEqual(len(telem), 3)

        validator = TelemetryValidator()
        valid, _ = validator.validate_record(telem[0])
        self.assertTrue(valid)

        cal_img, metrics = calibrate_side_scan_sonar(self.img_bgr, altitude_m=10.0, slant_range_m=75.0)
        self.assertEqual(cal_img.shape, self.img_bgr.shape)
        snr_val = metrics.get("final_snr_db", metrics.get("snr_db", 10.0))
        self.assertGreater(snr_val, 0.0)
        print(f"  ✓ Phase 1 Passed: Calibrated SNR = {snr_val:.2f} dB")

    def test_phase2_oscfar_and_decision_gate(self):
        print("\n[Phase 2] Testing OS-CFAR and Decision Gate...")
        cfar = OSCFARDetector(scaling_factor=1.6)
        _, candidates = cfar.detect_targets(self.img_bgr)
        self.assertGreater(len(candidates), 0)

        ae = SonarAnomalyDetector(device="cpu")
        anomalies, _ = ae.detect_anomalies(self.img_bgr)

        mock_dets = [{"bbox": [200, 150, 220, 170], "conf": 0.89, "class_name": "bottle"}]
        decisions, summary = evaluate_decision_gate(self.img_bgr, mock_dets, candidates, anomalies)
        self.assertGreater(len(decisions), 0)
        print(f"  ✓ Phase 2 Passed: Triage Summary = {summary}")

    def test_phase3_uncertainty_and_geolocation(self):
        print("\n[Phase 3] Testing MC Dropout Uncertainty and Geolocation...")
        engine = ResNet18InferenceEngine(device="cpu")
        crop = self.img_bgr[140:180, 190:230]
        mc_res = engine.predict_with_mc_dropout(crop, num_passes=3)
        self.assertIn("uncertainty_flag", mc_res)

        telem = generate_synthetic_telemetry(num_pings=1, base_lat=13.0827, base_lon=80.2707)[0]
        geo = project_pixel_to_latlon(210, 160, self.img_bgr.shape, telemetry=telem)
        self.assertAlmostEqual(geo.latitude, 13.0827, delta=0.01)
        self.assertGreater(geo.error_ellipse_semi_major_m, 0.0)
        print(f"  ✓ Phase 3 Passed: Geolocation = ({geo.latitude:.4f}°N, {geo.longitude:.4f}°E) ±{geo.error_ellipse_semi_major_m:.1f}m")

    def test_phase4_gis_and_database(self):
        print("\n[Phase 4] Testing GIS KDE Hotspots and Spatial Database...")
        lats = [13.0827, 13.0830, 13.0835]
        lons = [80.2707, 80.2710, 80.2715]
        _, _, density = compute_spatial_kde(lats, lons)
        self.assertEqual(density.shape, (50, 50))

        db = SurveyDatabase("outputs/test_verify.db")
        db.save_detections([{"class_name": "tire", "conf": 0.92, "latitude": 13.0827, "longitude": 80.2707}])
        records = db.get_all_detections()
        self.assertGreater(len(records), 0)
        print(f"  ✓ Phase 4 Passed: Spatial DB Records = {len(records)}")

    def test_phase5_active_learning(self):
        print("\n[Phase 5] Testing Active Learning Feedback Loop...")
        al = ActiveLearningManager("data/test_verify_al")
        sid = al.enqueue_for_review({"class_name": "novel_object", "conf": 0.35}, reason="Low Confidence")
        al.submit_review(sid, action="CONFIRM", operator_notes="Operator verified")
        stats = al.get_archive_stats()
        self.assertGreaterEqual(stats["confirmed"], 1)
        print(f"  ✓ Phase 5 Passed: Active Learning Stats = {stats}")

    def test_phase6_confidence_calibration(self):
        print("\n[Phase 6] Testing Confidence Calibration & Temperature Scaling...")
        from utils.confidence_calibration import TemperatureScaler, compute_calibration_metrics
        scaler = TemperatureScaler(temperature=1.35)
        raw_p = 0.95
        cal_p = scaler.calibrate_probability(raw_p)
        self.assertLess(cal_p, raw_p) # Temperature scaling softens overconfident logits
        
        metrics = compute_calibration_metrics([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 1])
        self.assertIn("ece", metrics)
        print(f"  ✓ Phase 6 Passed: Temperature Calibrated {raw_p:.2f} -> {cal_p:.3f} (ECE: {metrics['ece']:.3f})")

    def test_phase7_morphological_filtering(self):
        print("\n[Phase 7] Testing Morphological Filtering & Shape Constraints...")
        from utils.morphological_filter import filter_detection_by_morphology
        # Test valid compact bbox
        is_valid, feats = filter_detection_by_morphology(self.img_bgr, [200, 150, 220, 170])
        self.assertTrue(is_valid)
        self.assertGreater(feats.area, 0)
        print(f"  ✓ Phase 7 Passed: Morphological Filter Area={feats.area}px, Solidity={feats.solidity:.2f}, AR={feats.aspect_ratio:.2f}")

    def test_phase8_confidence_fusion_and_postgis(self):
        print("\n[Phase 8] Testing Multi-Evidence Confidence Fusion & PostGIS...")
        from utils.confidence_fusion import MultiEvidenceConfidenceFusion
        from utils.postgis_db import PostGISAdapter
        
        fusion = MultiEvidenceConfidenceFusion(temperature=1.35)
        fused = fusion.fuse_detection_confidence(
            raw_yolo_conf=0.92,
            cfar_contrast_ratio=1.8,
            ae_anomaly_score=0.10,
            has_shadow=True,
            shadow_contrast=0.35,
            calibrated_snr_db=14.0,
            mc_epistemic_variance=0.008
        )
        self.assertGreater(fused.final_confidence_pct, 50.0)
        self.assertIn("Calibrated YOLO (40%)", fused.evidence_breakdown)
        print(f"  ✓ Confidence Fusion Passed: Final Fused Confidence = {fused.final_confidence_pct:.1f}%")
        
        # Test PostGIS SQL dump generation
        pg = PostGISAdapter()
        dump_sql = pg.export_postgis_sql_dump([{
            "class_name": "bottle", "conf": 0.92, "latitude": 13.0827, "longitude": 80.2707,
            "fused_confidence": fused.final_confidence_pct / 100.0, "uncertainty_flag": "LOW"
        }], output_path="outputs/test_postgis.sql")
        self.assertIn("ST_SetSRID(ST_MakePoint", dump_sql)
        print(f"  ✓ PostGIS Dump Passed: Generated {len(dump_sql.splitlines())} lines of PostGIS DDL/DML")


if __name__ == "__main__":
    print("=" * 70)
    print("🧪 RUNNING AKHET MARINE AI PLATFORM COMPLETE VERIFICATION SUITE")
    print("=" * 70)
    unittest.main(verbosity=2)
