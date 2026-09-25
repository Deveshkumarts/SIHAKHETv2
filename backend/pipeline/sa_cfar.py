"""Stage 6 - SA-CFAR candidate generation (seabed-adaptive CFAR on K-Means clutter regimes)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from models.os_cfar import SACFARDetector


def pad_bbox(bbox, pad: int, shape) -> List[int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    return [int(max(0, x1 - pad)), int(max(0, y1 - pad)), int(min(w, x2 + pad)), int(min(h, y2 + pad))]


def merge_candidates(cands: List[Dict[str, Any]], gap_px: int) -> List[Dict[str, Any]]:
    """Merge candidates whose boxes touch or lie within `gap_px` of each other (one physical object that CFAR
    fragmented). Union box, summed area, max contrast; the dominant fragment keeps its regime label."""
    if len(cands) < 2 or gap_px < 0:
        return cands
    parent = list(range(len(cands)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def near(a, b):
        return not (a[2] + gap_px < b[0] or b[2] + gap_px < a[0] or a[3] + gap_px < b[1] or b[3] + gap_px < a[1])

    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            if near(cands[i]["bbox"], cands[j]["bbox"]):
                parent[find(i)] = find(j)
    groups: Dict[int, List[int]] = {}
    for i in range(len(cands)):
        groups.setdefault(find(i), []).append(i)
    merged = []
    for idx in groups.values():
        members = [cands[i] for i in idx]
        lead = max(members, key=lambda c: c["area"])
        if len(members) == 1:
            merged.append(dict(lead))
            continue
        x1, y1 = min(m["bbox"][0] for m in members), min(m["bbox"][1] for m in members)
        x2, y2 = max(m["bbox"][2] for m in members), max(m["bbox"][3] for m in members)
        m = dict(lead)
        m.update({"bbox": [x1, y1, x2, y2], "centroid": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                  "area": int(sum(c["area"] for c in members)),
                  "peak_intensity": max(c["peak_intensity"] for c in members),
                  "contrast_ratio": max(c["contrast_ratio"] for c in members), "merged_fragments": len(members)})
        merged.append(m)
    return merged


def run_sa_cfar(image_bgr: np.ndarray, cfg, waterfall: bool = True) -> Tuple[np.ndarray, List[Dict[str, Any]], Any]:
    """waterfall=True  : real side-scan waterfall -> the central corridor may be nadir water column and is suppressed.
    waterfall=False : plain image / cropped object -> there is no nadir; do not suppress the image centre."""
    c = cfg.sa_cfar
    det = SACFARDetector(guard_size=c.guard_size, ref_size=c.ref_size,
                         min_target_area=c.min_target_area, max_target_area=c.max_target_area)
    clutter_in = None
    if not waterfall:
        from models.clutter_segmentation import REGIME_NADIR, REGIME_SMOOTH_SAND
        clutter_in = det.segmenter.segment_clutter(image_bgr)
        clutter_in.regime_map[clutter_in.regime_map == REGIME_NADIR] = REGIME_SMOOTH_SAND
    mask, cands, clutter = det.detect_adaptive_targets(image_bgr, clutter_result=clutter_in)
    cands = merge_candidates(cands, int(cfg.get("sa_cfar.merge_gap_px", 6)))
    for i, cand in enumerate(cands):
        cand["candidate_id"] = i
        cand["roi"] = pad_bbox(cand["bbox"], int(c.candidate_pad_px), image_bgr.shape)
    return mask, cands, clutter
