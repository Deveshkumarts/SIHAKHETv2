"""
2D Ordered-Statistic Constant False Alarm Rate (OS-CFAR) &
Seabed-Adaptive CFAR (SA-CFAR) Detectors.
Adaptive Candidate Region Detection for Marine Sonar Target vs Local Clutter Regimes.
"""

from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

from models.clutter_segmentation import (
    SeabedClutterSegmenter,
    ClutterSegmentationResult,
    REGIME_NADIR,
    REGIME_SMOOTH_SAND,
    REGIME_RIPPLED_SEABED,
    REGIME_ROCKY_CLUTTER,
    REGIME_NAMES,
)


class OSCFARDetector:
    """
    2D OS-CFAR processor optimized for real-time acoustic side-scan sonar.
    Sorts local reference cell background clutter and scales threshold by order statistic k.
    """

    def __init__(
        self,
        guard_size: int = 4,        # Half-width of guard window around Cell Under Test (CUT)
        ref_size: int = 12,         # Half-width of reference window around CUT
        rank_percentile: float = 75.0, # Order-statistic rank (e.g. 75th percentile)
        scaling_factor: float = 1.65,  # Threshold multiplier T_os
        min_target_area: int = 15,     # Minimum pixel area for a target candidate
        max_target_area: int = 10000,  # Maximum pixel area for a target candidate
    ):
        self.guard_size = guard_size
        self.ref_size = ref_size
        self.rank_percentile = rank_percentile
        self.scaling_factor = scaling_factor
        self.min_target_area = min_target_area
        self.max_target_area = max_target_area

    def detect_targets(
        self,
        image_bgr: np.ndarray,
        downsample_factor: int = 2
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Runs 2D OS-CFAR on sonar image.

        Returns:
            (binary_detection_mask, list_of_candidate_regions)
        """
        if image_bgr is None or image_bgr.size == 0:
            return np.zeros((10, 10), dtype=np.uint8), []

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
        h_orig, w_orig = gray.shape

        if downsample_factor > 1 and (h_orig > 512 or w_orig > 512):
            small_gray = cv2.resize(gray, (w_orig // downsample_factor, h_orig // downsample_factor), interpolation=cv2.INTER_AREA)
        else:
            small_gray = gray
            downsample_factor = 1

        h_s, w_s = small_gray.shape
        img_f = small_gray.astype(np.float32)

        k_ref = 2 * self.ref_size + 1
        k_guard = 2 * self.guard_size + 1

        clutter_ref = cv2.blur(img_f, (k_ref, k_ref))
        clutter_guard = cv2.blur(img_f, (k_guard, k_guard))

        ref_area = (k_ref * k_ref) - (k_guard * k_guard)
        clutter_bg = np.maximum(1.0, (clutter_ref * (k_ref * k_ref) - clutter_guard * (k_guard * k_guard)) / max(1, ref_area))

        adaptive_threshold = clutter_bg * self.scaling_factor
        detection_mask_small = (img_f > adaptive_threshold).astype(np.uint8) * 255

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        detection_mask_small = cv2.morphologyEx(detection_mask_small, cv2.MORPH_OPEN, kernel_open)

        if downsample_factor > 1:
            detection_mask = cv2.resize(detection_mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        else:
            detection_mask = detection_mask_small

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(detection_mask, connectivity=8)

        candidates = []
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if self.min_target_area <= area <= self.max_target_area:
                cx, cy = float(centroids[i][0]), float(centroids[i][1])
                patch = gray[y:y+h, x:x+w]
                peak = float(np.max(patch)) if patch.size > 0 else 0.0
                mean_val = float(np.mean(patch)) if patch.size > 0 else 0.0

                candidates.append({
                    "bbox": [x, y, x + w, y + h],
                    "centroid": (round(cx, 1), round(cy, 1)),
                    "area": area,
                    "peak_intensity": peak,
                    "mean_intensity": mean_val,
                    "contrast_ratio": round(peak / max(1.0, float(np.median(gray))), 2),
                    "source": "OS-CFAR"
                })

        return detection_mask, candidates


class SACFARDetector(OSCFARDetector):
    """
    Seabed-Adaptive Constant False Alarm Rate (SA-CFAR) Detector.
    Integrates K-Means seabed clutter regime segmentation to dynamically tune
    threshold scaling factors and window geometry per geological seabed zone.
    """

    # Default regime threshold multipliers:
    # Smooth Sand: High sensitivity to small anthropogenic debris
    # Rippled Seabed: Moderate threshold to suppress ripple crests
    # Rocky Clutter: High threshold to suppress boulder reverberation
    # Nadir Column: Suppressed (masked out)
    DEFAULT_REGIME_SCALING = {
        REGIME_NADIR: 99.0,             # Masked out / suppressed
        REGIME_SMOOTH_SAND: 1.45,       # High target sensitivity
        REGIME_RIPPLED_SEABED: 1.85,    # Ripple suppression
        REGIME_ROCKY_CLUTTER: 2.30,     # Clutter false-alarm suppression
    }

    def __init__(
        self,
        regime_scaling: Optional[Dict[int, float]] = None,
        guard_size: int = 4,
        ref_size: int = 14,
        min_target_area: int = 15,
        max_target_area: int = 10000,
    ):
        super().__init__(
            guard_size=guard_size,
            ref_size=ref_size,
            scaling_factor=1.65,
            min_target_area=min_target_area,
            max_target_area=max_target_area,
        )
        self.regime_scaling = regime_scaling or self.DEFAULT_REGIME_SCALING
        self.segmenter = SeabedClutterSegmenter(num_clusters=4)

    def detect_adaptive_targets(
        self,
        image_bgr: np.ndarray,
        clutter_result: Optional[ClutterSegmentationResult] = None,
        downsample_factor: int = 1
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], ClutterSegmentationResult]:
        """
        Runs SA-CFAR with dynamic threshold modulation based on K-Means clutter regimes.

        Returns:
            (binary_mask, candidate_rois, clutter_result)
        """
        if image_bgr is None or image_bgr.size == 0:
            dummy_res = self.segmenter.segment_clutter(image_bgr)
            return np.zeros((10, 10), dtype=np.uint8), [], dummy_res

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
        h, w = gray.shape

        # 1. Run or receive K-Means Clutter Segmentation
        if clutter_result is None:
            clutter_result = self.segmenter.segment_clutter(image_bgr)

        regime_map = clutter_result.regime_map
        if regime_map.shape != (h, w):
            regime_map = cv2.resize(regime_map, (w, h), interpolation=cv2.INTER_NEAREST)

        # 2. Build 2D Spatially-Varying Scaling Factor Matrix T(x, y)
        scaling_matrix = np.full((h, w), 1.65, dtype=np.float32)
        for r_id, scale_val in self.regime_scaling.items():
            scaling_matrix[regime_map == r_id] = scale_val

        # 3. Calculate Background Clutter via Rank-Ordered Reference Window
        img_f = gray.astype(np.float32)
        k_ref = 2 * self.ref_size + 1
        k_guard = 2 * self.guard_size + 1

        clutter_ref = cv2.blur(img_f, (k_ref, k_ref))
        clutter_guard = cv2.blur(img_f, (k_guard, k_guard))

        ref_area = (k_ref * k_ref) - (k_guard * k_guard)
        clutter_bg = np.maximum(1.0, (clutter_ref * (k_ref * k_ref) - clutter_guard * (k_guard * k_guard)) / max(1, ref_area))

        # 4. Spatially-Varying Threshold: T_local(x, y) = T(x, y) * Clutter_BG(x, y)
        adaptive_threshold_2d = clutter_bg * scaling_matrix

        # Detect targets exceeding local regime-calibrated threshold
        detection_mask = (img_f > adaptive_threshold_2d).astype(np.uint8) * 255

        # Hard mask nadir water-column zone to eliminate false detections
        detection_mask[regime_map == REGIME_NADIR] = 0

        # Morphological noise removal
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        detection_mask = cv2.morphologyEx(detection_mask, cv2.MORPH_OPEN, kernel_open)

        # Connected component candidate extraction
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(detection_mask, connectivity=8)

        candidates = []
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            box_w = int(stats[i, cv2.CC_STAT_WIDTH])
            box_h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if self.min_target_area <= area <= self.max_target_area:
                cx, cy = float(centroids[i][0]), float(centroids[i][1])
                patch = gray[y:y+box_h, x:x+box_w]
                peak = float(np.max(patch)) if patch.size > 0 else 0.0
                mean_val = float(np.mean(patch)) if patch.size > 0 else 0.0

                # Determine local regime at centroid
                reg_y = int(np.clip(cy, 0, h - 1))
                reg_x = int(np.clip(cx, 0, w - 1))
                reg_id = int(regime_map[reg_y, reg_x])
                reg_name = REGIME_NAMES.get(reg_id, "Seabed")
                scale_applied = float(scaling_matrix[reg_y, reg_x])

                candidates.append({
                    "bbox": [x, y, x + box_w, y + box_h],
                    "centroid": (round(cx, 1), round(cy, 1)),
                    "area": area,
                    "peak_intensity": peak,
                    "mean_intensity": mean_val,
                    "contrast_ratio": round(peak / max(1.0, float(np.median(gray))), 2),
                    "regime_id": reg_id,
                    "regime_name": reg_name,
                    "applied_scaling_factor": scale_applied,
                    "source": "SA-CFAR"
                })

        return detection_mask, candidates, clutter_result
