"""
Acoustic Seabed Clutter Regime Segmentation Engine.
Implements:
  1. Multi-scale statistical acoustic feature extraction (Mean, Variance, Roughness, Gradient)
  2. K-Means (K=4) unsupervised physical seabed regime classification:
       - Regime 0: Nadir Water Column (Acoustic Void)
       - Regime 1: Smooth Sand / Silt (Low Clutter)
       - Regime 2: Rippled Seabed (Periodic Acoustic Texture)
       - Regime 3: Rocky / High Reverberation Clutter (Complex Scatterers)
  3. Colorized Clutter Regime Overlay for Maritime Tactical Visualization
  4. Regime-aware spatial statistics for false-alarm suppression
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# SEABED REGIME CONSTANTS & METADATA
# ─────────────────────────────────────────────────────────────────────────────
REGIME_NADIR = 0
REGIME_SMOOTH_SAND = 1
REGIME_RIPPLED_SEABED = 2
REGIME_ROCKY_CLUTTER = 3

REGIME_NAMES = {
    REGIME_NADIR: "Nadir Water Column",
    REGIME_SMOOTH_SAND: "Smooth Sand / Silt",
    REGIME_RIPPLED_SEABED: "Rippled Seabed",
    REGIME_ROCKY_CLUTTER: "Rocky / High Clutter",
}

# Distinct tactical BGR colors for visualization:
# Nadir: Navy Blue [30, 16, 6]
# Smooth Sand: Ochre Yellow [115, 163, 212]
# Rippled Seabed: Vibrant Cyan [216, 180, 0]
# Rocky Clutter: Coral Orange/Red [81, 111, 231]
REGIME_COLORS_BGR = {
    REGIME_NADIR: (30, 16, 6),
    REGIME_SMOOTH_SAND: (115, 163, 212),
    REGIME_RIPPLED_SEABED: (216, 180, 0),
    REGIME_ROCKY_CLUTTER: (81, 111, 231),
}


@dataclass
class ClutterSegmentationResult:
    regime_map: np.ndarray             # 2D int array (H, W) values 0..3
    colored_overlay_bgr: np.ndarray    # 2D BGR image (H, W, 3)
    blended_view_bgr: np.ndarray       # Sonar image alpha-blended with regime colors
    regime_percentages: Dict[str, float]
    regime_stats: Dict[int, Dict[str, float]]
    dominant_regime: str


class SeabedClutterSegmenter:
    """
    Unsupervised Seabed Acoustic Clutter Classifier using K-Means and Physical Features.
    """

    def __init__(self, num_clusters: int = 4, patch_radius: int = 7):
        self.num_clusters = num_clusters
        self.patch_radius = patch_radius

    def extract_acoustic_features(self, gray_img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extracts multi-scale acoustic backscatter descriptors:
          1. Local mean intensity (mu)
          2. Local acoustic standard deviation (sigma)
          3. Local surface roughness ratio (sigma / (mu + eps))
          4. Sobel spatial gradient magnitude (edge transition frequency)
          5. Nadir proximity weight
        """
        h, w = gray_img.shape
        img_f = gray_img.astype(np.float32)

        k_size = (2 * self.patch_radius + 1, 2 * self.patch_radius + 1)

        # 1. Local mean
        local_mean = cv2.blur(img_f, k_size)

        # 2. Local variance: E[X^2] - (E[X])^2
        img_sq = img_f ** 2
        local_sq_mean = cv2.blur(img_sq, k_size)
        local_var = np.maximum(0.0, local_sq_mean - (local_mean ** 2))
        local_std = np.sqrt(local_var)

        # 3. Acoustic Roughness (Coefficient of Variation)
        roughness = local_std / (local_mean + 1e-4)

        # 4. Spatial gradient magnitude
        grad_x = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_smooth = cv2.blur(grad_mag, (5, 5))

        # 5. Proximity to center nadir line
        mid_col = w / 2.0
        cols = np.arange(w, dtype=np.float32)
        nadir_dist_1d = np.abs(cols - mid_col) / max(1.0, mid_col)  # 0 at nadir, 1 at edges
        nadir_dist_2d = np.tile(nadir_dist_1d, (h, 1))

        # Downsample feature matrices for ultra-fast K-Means clustering (sub-15ms)
        step = 2 if (h > 256 or w > 256) else 1
        sub_mean = local_mean[::step, ::step].flatten()
        sub_std = local_std[::step, ::step].flatten()
        sub_rough = roughness[::step, ::step].flatten()
        sub_grad = grad_smooth[::step, ::step].flatten()
        sub_nadir = nadir_dist_2d[::step, ::step].flatten()

        # Normalize features to [0, 1] standard scale
        def norm(arr):
            mn, mx = np.min(arr), np.max(arr)
            return (arr - mn) / max(1e-5, mx - mn)

        features = np.column_stack([
            norm(sub_mean),
            norm(sub_std),
            norm(sub_rough),
            norm(sub_grad),
            norm(sub_nadir)
        ]).astype(np.float32)

        return features, (local_mean.shape, step)

    def segment_clutter(self, image_bgr: np.ndarray) -> ClutterSegmentationResult:
        """
        Runs unsupervised K-Means clutter segmentation on sonar image.
        Maps raw clusters to the 4 physical seabed regimes.
        """
        if image_bgr is None or image_bgr.size == 0:
            dummy_map = np.zeros((10, 10), dtype=np.uint8)
            return ClutterSegmentationResult(
                regime_map=dummy_map,
                colored_overlay_bgr=image_bgr,
                blended_view_bgr=image_bgr,
                regime_percentages={},
                regime_stats={},
                dominant_regime=REGIME_NAMES[REGIME_SMOOTH_SAND]
            )

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if len(image_bgr.shape) == 3 else image_bgr
        h, w = gray.shape

        features, (orig_shape, step) = self.extract_acoustic_features(gray)

        # OpenCV K-Means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.2)
        flags = cv2.KMEANS_PP_CENTERS

        _, labels, centers = cv2.kmeans(features, self.num_clusters, None, criteria, 3, flags)
        labels = labels.flatten()

        # Physical acoustic ranking of cluster centers:
        # centers columns: [mean, std, roughness, gradient, nadir_dist]
        cluster_ranking = []
        for c_idx in range(self.num_clusters):
            c = centers[c_idx]
            c_mean = c[0]
            c_std = c[1]
            c_rough = c[2]
            c_grad = c[3]
            c_nadir = c[4]
            cluster_ranking.append({
                "cluster_id": c_idx,
                "mean": c_mean,
                "std": c_std,
                "rough": c_rough,
                "grad": c_grad,
                "nadir_dist": c_nadir,
                # Nadir score: low mean + low std + near center
                "nadir_score": (1.0 - c_mean) * 1.5 + (1.0 - c_nadir) * 1.5,
                # Rock score: high roughness + high std + high gradient
                "rock_score": c_rough * 1.5 + c_std * 1.2 + c_grad * 1.0,
            })

        # 1. Identify Nadir cluster (highest nadir_score)
        nadir_cluster = max(cluster_ranking, key=lambda x: x["nadir_score"])["cluster_id"]

        remaining = [c for c in cluster_ranking if c["cluster_id"] != nadir_cluster]

        # 2. Identify Rocky Clutter (highest rock_score among remaining)
        rock_cluster = max(remaining, key=lambda x: x["rock_score"])["cluster_id"]

        remaining = [c for c in remaining if c["cluster_id"] != rock_cluster]

        # 3. Sort last two by roughness: lower = Smooth Sand, higher = Rippled Seabed
        if len(remaining) >= 2:
            remaining.sort(key=lambda x: x["rough"])
            sand_cluster = remaining[0]["cluster_id"]
            ripple_cluster = remaining[1]["cluster_id"]
        elif len(remaining) == 1:
            sand_cluster = remaining[0]["cluster_id"]
            ripple_cluster = sand_cluster
        else:
            sand_cluster = nadir_cluster
            ripple_cluster = nadir_cluster

        # Map cluster IDs to physical regimes
        cluster_to_regime = {
            nadir_cluster: REGIME_NADIR,
            sand_cluster: REGIME_SMOOTH_SAND,
            ripple_cluster: REGIME_RIPPLED_SEABED,
            rock_cluster: REGIME_ROCKY_CLUTTER,
        }

        mapped_labels = np.array([cluster_to_regime.get(lbl, REGIME_SMOOTH_SAND) for lbl in labels], dtype=np.uint8)

        # Reshape to 2D grid and upsample if downsampled
        sub_h = int(np.ceil(h / step))
        sub_w = int(np.ceil(w / step))
        regime_grid_small = mapped_labels[:sub_h * sub_w].reshape(sub_h, sub_w)

        if step > 1:
            regime_map = cv2.resize(regime_grid_small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            regime_map = regime_grid_small

        # Spatial consistency filtering (remove salt-and-pepper clutter artifacts)
        regime_map = cv2.medianBlur(regime_map, 5)

        # Generate colorized overlay
        colored_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for r_id, color in REGIME_COLORS_BGR.items():
            colored_overlay[regime_map == r_id] = color

        # Alpha blend with original sonar image
        alpha = 0.42
        blended = cv2.addWeighted(colored_overlay, alpha, image_bgr, 1.0 - alpha, 0)

        # Compute regime distribution percentages
        total_px = regime_map.size
        percentages = {}
        regime_stats = {}

        for r_id, r_name in REGIME_NAMES.items():
            cnt = int(np.count_nonzero(regime_map == r_id))
            pct = round((cnt / total_px) * 100.0, 1)
            percentages[r_name] = pct

            mask = (regime_map == r_id)
            if cnt > 0:
                regime_pixels = gray[mask]
                regime_stats[r_id] = {
                    "mean_intensity": round(float(np.mean(regime_pixels)), 1),
                    "std_intensity": round(float(np.std(regime_pixels)), 1),
                    "pixel_count": cnt,
                    "coverage_pct": pct,
                }

        # Determine dominant seabed type
        seabed_regimes = {k: v for k, v in percentages.items() if k != REGIME_NAMES[REGIME_NADIR]}
        dominant = max(seabed_regimes.items(), key=lambda x: x[1])[0] if seabed_regimes else REGIME_NAMES[REGIME_SMOOTH_SAND]

        return ClutterSegmentationResult(
            regime_map=regime_map,
            colored_overlay_bgr=colored_overlay,
            blended_view_bgr=blended,
            regime_percentages=percentages,
            regime_stats=regime_stats,
            dominant_regime=dominant
        )
