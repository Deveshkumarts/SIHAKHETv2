"""
Automated Test Suite for Raw Sonar Ingestion (.xtf / .jsf) and
Seabed Clutter Regime Segmentation (K-Means & SA-CFAR).
"""

import sys
import unittest
import tempfile
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import cv2

from utils.sonar_raw_ingestion import (
    XTFReader,
    JSFReader,
    generate_synthetic_xtf,
    ingest_raw_sonar_file,
)
from models.clutter_segmentation import (
    SeabedClutterSegmenter,
    REGIME_NADIR,
    REGIME_SMOOTH_SAND,
    REGIME_RIPPLED_SEABED,
    REGIME_ROCKY_CLUTTER,
    REGIME_NAMES,
)
from models.os_cfar import OSCFARDetector, SACFARDetector


class TestRawIngestionAndClutter(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

        # Synthesize acoustic test image (400 x 600)
        # Center = Nadir (dark column)
        # Left = Port side (smooth sand + rippled zone)
        # Right = Starboard side (rocky clutter + bright target with shadow)
        self.img = np.random.normal(80, 12, (300, 600)).clip(10, 240).astype(np.uint8)

        # Nadir center column
        self.img[:, 280:320] = np.random.normal(12, 4, (300, 40)).clip(2, 25).astype(np.uint8)

        # Left: sand ripples
        x_grid = np.arange(280)
        ripple = (np.sin(x_grid * 0.4) * 20.0).astype(np.int16)
        self.img[:, :280] = np.clip(self.img[:, :280].astype(np.int16) + ripple, 0, 255).astype(np.uint8)

        # Right: high variance rocky scatterers
        self.img[:, 450:580] = np.random.normal(110, 35, (300, 130)).clip(30, 240).astype(np.uint8)

        # Debris target + shadow on starboard
        self.img[120:140, 380:400] = 250   # Highlight
        self.img[120:140, 400:435] = 8     # Shadow

        self.img_bgr = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_xtf_generation_and_ingestion(self):
        """Tests end-to-end binary Triton XTF generation, ping decoding, and waterfall reconstruction."""
        xtf_path = self.test_dir / "test_mission.xtf"

        # 1. Generate compliant binary XTF file
        generated_file = generate_synthetic_xtf(
            output_path=xtf_path,
            num_pings=64,
            samples_per_channel=256,
            base_lat=13.0827,
            base_lon=80.2707,
            inject_target=True
        )
        self.assertTrue(generated_file.exists())
        self.assertGreater(generated_file.stat().st_size, 1024)

        # 2. Parse using universal dispatcher
        waterfall_bgr, telemetry, meta = ingest_raw_sonar_file(generated_file)

        # Validate waterfall dimensions
        self.assertIsNotNone(waterfall_bgr)
        self.assertEqual(len(waterfall_bgr.shape), 3)
        self.assertEqual(waterfall_bgr.shape[0], 64)  # 64 pings = 64 rows
        self.assertEqual(waterfall_bgr.shape[1], 256 * 2 + 6)  # Port + Nadir + Stbd

        # Validate telemetry
        self.assertEqual(len(telemetry), 64)
        first_ping = telemetry[0]
        self.assertAlmostEqual(first_ping.latitude, 13.0827, places=3)
        self.assertAlmostEqual(first_ping.longitude, 80.2707, places=3)
        self.assertGreater(first_ping.altitude_m, 0.0)
        self.assertGreater(first_ping.slant_range_m, 0.0)

        # Validate metadata
        self.assertEqual(meta["format"], "Triton XTF")
        self.assertEqual(meta["num_pings"], 64)
        self.assertEqual(meta["samples_per_channel"], 256)

    def test_clutter_regime_segmentation(self):
        """Tests K-Means acoustic clutter segmentation across 4 physical seabed regimes."""
        segmenter = SeabedClutterSegmenter(num_clusters=4)
        result = segmenter.segment_clutter(self.img_bgr)

        # Verify output structures
        self.assertEqual(result.regime_map.shape, self.img.shape)
        self.assertEqual(result.colored_overlay_bgr.shape, self.img_bgr.shape)
        self.assertEqual(result.blended_view_bgr.shape, self.img_bgr.shape)

        # Verify regime labels
        unique_labels = np.unique(result.regime_map)
        for lbl in unique_labels:
            self.assertIn(lbl, [REGIME_NADIR, REGIME_SMOOTH_SAND, REGIME_RIPPLED_SEABED, REGIME_ROCKY_CLUTTER])

        # Verify percentages sum to ~100%
        total_pct = sum(result.regime_percentages.values())
        self.assertAlmostEqual(total_pct, 100.0, delta=1.5)

        # Verify nadir column is captured at center
        mid_col = self.img.shape[1] // 2
        nadir_label = result.regime_map[:, mid_col]
        nadir_ratio = np.count_nonzero(nadir_label == REGIME_NADIR) / len(nadir_label)
        self.assertGreater(nadir_ratio, 0.60)

    def test_sa_cfar_adaptive_detection(self):
        """Tests Seabed-Adaptive CFAR with regime-modulated scaling and false-alarm suppression."""
        sa_cfar = SACFARDetector(
            guard_size=4,
            ref_size=12,
            min_target_area=10,
            max_target_area=5000,
        )

        det_mask, candidates, clutter_res = sa_cfar.detect_adaptive_targets(self.img_bgr)

        self.assertIsNotNone(det_mask)
        self.assertGreater(len(candidates), 0)

        # Verify candidate metadata
        found_target = False
        for c in candidates:
            self.assertIn("regime_id", c)
            self.assertIn("regime_name", c)
            self.assertIn("applied_scaling_factor", c)
            self.assertEqual(c["source"], "SA-CFAR")

            # Check if injected target is found around [120:140, 380:400]
            bbox = c["bbox"]
            if 360 <= bbox[0] <= 410 and 110 <= bbox[1] <= 150:
                found_target = True

        self.assertTrue(found_target, "Injected debris target was detected by SA-CFAR")

        # Verify Nadir suppression: NO candidates should have centroid inside the nadir zone
        mid_col = self.img.shape[1] // 2
        for c in candidates:
            cx = c["centroid"][0]
            # Should not have centroid in narrow nadir column (290-310)
            self.assertFalse(290 <= cx <= 310, f"Candidate centroid {cx} was unexpectedly inside nadir column")


if __name__ == "__main__":
    unittest.main()
