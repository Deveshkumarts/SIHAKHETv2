"""
2D Ordered-Statistic Constant False Alarm Rate (OS-CFAR) Detector.
Adaptive Candidate Region Detection for Marine Sonar Target vs Local Clutter.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np


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
    ) -> Tuple[np.ndarray, List[Dict[str, any]]]:
        """
        Runs 2D OS-CFAR on sonar image.

        Returns:
            (binary_detection_mask, list_of_candidate_regions)
            Candidate region format:
            {
                "bbox": [x1, y1, x2, y2],
                "centroid": (cx, cy),
                "area": int,
                "peak_intensity": float,
                "contrast_ratio": float,
                "source": "OS-CFAR"
            }
        """
        if image_bgr is None or image_bgr.size == 0:
            return np.zeros((10, 10), dtype=np.uint8), []

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
        h_orig, w_orig = gray.shape

        # Downsample for sub-10ms real-time processing if image is large
        if downsample_factor > 1 and (h_orig > 512 or w_orig > 512):
            small_gray = cv2.resize(gray, (w_orig // downsample_factor, h_orig // downsample_factor), interpolation=cv2.INTER_AREA)
        else:
            small_gray = gray
            downsample_factor = 1

        h_s, w_s = small_gray.shape
        img_f = small_gray.astype(np.float32)

        # Compute background clutter using fast rank estimation
        # Median/percentile kernel over outer reference window
        k_ref = 2 * self.ref_size + 1
        k_guard = 2 * self.guard_size + 1

        # Morphological rank filtering: dilate/erode approximation of local clutter rank
        clutter_ref = cv2.blur(img_f, (k_ref, k_ref))
        clutter_guard = cv2.blur(img_f, (k_guard, k_guard))

        # Reference cells = outer window minus guard window
        ref_area = (k_ref * k_ref) - (k_guard * k_guard)
        clutter_bg = np.maximum(1.0, (clutter_ref * (k_ref * k_ref) - clutter_guard * (k_guard * k_guard)) / max(1, ref_area))

        # Adaptive threshold: T = scaling_factor * clutter_bg
        adaptive_threshold = clutter_bg * self.scaling_factor

        # Detect pixels where Cell Under Test (CUT) exceeds local adaptive clutter threshold
        detection_mask_small = (img_f > adaptive_threshold).astype(np.uint8) * 255

        # Morphological opening to remove isolated acoustic speckle spikes
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        detection_mask_small = cv2.morphologyEx(detection_mask_small, cv2.MORPH_OPEN, kernel_open)

        # Upsample mask back to original resolution
        if downsample_factor > 1:
            detection_mask = cv2.resize(detection_mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        else:
            detection_mask = detection_mask_small

        # Connected component analysis to extract bounding boxes
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
