"""
YOLO11-Seg dataset builder for the SIH 27-class side-scan sonar set.

Facts about the source data (verified, not assumed):
  * Every label row in SIH_Dataset_27class is a bounding box (5 columns) --
    there are NO pixel-level ground-truth masks.
  * 10 of the 12 shipwreck surveys are split across train/val/test, i.e. the
    original split leaks survey information.

What this module does about it:
  1. Re-splits shipwreck images by survey (whole surveys go to one split).
     Other classes are single-source object crops with no survey identifier,
     so their original split is kept and this is stated in the report.
  2. Generates WEAK PSEUDO-MASKS from each box with GrabCut (falls back to an
     inscribed ellipse when GrabCut degenerates). These are pseudo-labels:
     mask metrics computed against them measure agreement with the pseudo-mask
     generator, not true pixel accuracy. The report says so explicitly.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

SPLITS = ("train", "val", "test")
SHIPWRECK_PREFIX = "Shipwrecks_"


def survey_group(stem: str) -> Optional[str]:
    """Survey/wreck id for shipwreck frames (e.g. Shipwrecks_DM_Wilson_07 -> Shipwrecks_DM_Wilson)."""
    if stem.startswith(SHIPWRECK_PREFIX):
        return re.sub(r"_\d+$", "", stem)
    return None


def read_yolo_boxes(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    for line in label_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) == 5:
            rows.append((int(p[0]), *map(float, p[1:])))
    return rows


def leakage_report(src_root: Path) -> Dict:
    groups: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in SPLITS:
        for img in (src_root / s / "images").glob("*.png"):
            g = survey_group(img.stem)
            if g:
                groups[g][s] += 1
    leaking = {g: dict(c) for g, c in groups.items() if len(c) > 1}
    return {"survey_groups": len(groups), "groups_spanning_multiple_splits": len(leaking), "leaking": leaking}


def assign_survey_splits(groups: Dict[str, int], seed: int = 0, val_frac: float = 0.10, test_frac: float = 0.12) -> Dict[str, str]:
    """Deterministic greedy assignment of whole surveys to splits by image count."""
    rng = np.random.default_rng(seed)
    total = sum(groups.values())
    names = sorted(groups)
    rng.shuffle(names)
    target = {"test": test_frac * total, "val": val_frac * total}
    have = {"train": 0, "val": 0, "test": 0}
    out: Dict[str, str] = {}
    for g in names:                                 # shuffled order -> seed-dependent, reproducible
        for s in ("test", "val"):
            if have[s] < target[s] and have[s] + groups[g] <= target[s] * 1.6:
                out[g] = s
                have[s] += groups[g]
                break
    for g in names:
        out.setdefault(g, "train")
    return out


def _ellipse(x1, y1, x2, y2, w, h, n: int = 16) -> np.ndarray:
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = max(1, (x2 - x1) / 2), max(1, (y2 - y1) / 2)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([cx + a * np.cos(t), cy + b * np.sin(t)], 1).astype(np.float32)
    pts[:, 0] /= w
    pts[:, 1] /= h
    return np.clip(pts, 0, 1)


def box_to_pseudo_polygon(img_bgr: np.ndarray, cx: float, cy: float, bw: float, bh: float) -> Tuple[np.ndarray, str]:
    """Return (normalized polygon Nx2, method) for one box. Method: 'grabcut' or 'ellipse'."""
    h, w = img_bgr.shape[:2]
    x1 = int(max(0, round((cx - bw / 2) * w)))
    y1 = int(max(0, round((cy - bh / 2) * h)))
    x2 = int(min(w, round((cx + bw / 2) * w)))
    y2 = int(min(h, round((cy + bh / 2) * h)))
    rw, rh = x2 - x1, y2 - y1
    if rw < 4 or rh < 4:
        return _ellipse(x1, y1, x2, y2, w, h), "ellipse"

    poly = None
    try:
        # GrabCut on a padded, size-capped crop (source frames reach 5581x1728 px;
        # running it on the full frame is orders of magnitude slower and pointless).
        pad = max(6, int(0.25 * max(rw, rh)))
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
        crop = img_bgr[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[:2]
        s = min(1.0, 160.0 / max(ch, cw))
        if s < 1.0:
            crop = cv2.resize(crop, (max(8, int(cw * s)), max(8, int(ch * s))), interpolation=cv2.INTER_AREA)
        rx1, ry1 = int((x1 - cx1) * s), int((y1 - cy1) * s)
        rrw, rrh = max(4, int(rw * s)), max(4, int(rh * s))
        rrw, rrh = min(rrw, crop.shape[1] - rx1), min(rrh, crop.shape[0] - ry1)

        mask = np.zeros(crop.shape[:2], np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(crop, mask, (rx1, ry1, rrw, rrh), bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        fg[:ry1], fg[ry1 + rrh:], fg[:, :rx1], fg[:, rx1 + rrw:] = 0, 0, 0, 0
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            ratio = cv2.contourArea(c) / float(rrw * rrh)
            if 0.12 <= ratio <= 0.97:                       # reject degenerate masks
                c = cv2.approxPolyDP(c, 0.015 * cv2.arcLength(c, True), True)
                if len(c) >= 3:
                    poly = c.reshape(-1, 2).astype(np.float32)
                    poly = poly / s                          # back to full-frame pixels
                    poly[:, 0] += cx1
                    poly[:, 1] += cy1
    except cv2.error:
        poly = None
    if poly is None:
        return _ellipse(x1, y1, x2, y2, w, h), "ellipse"
    poly[:, 0] /= w
    poly[:, 1] /= h
    return np.clip(poly, 0, 1), "grabcut"


def build_seg_dataset(src_root, dst_root, seed: int = 0, limit: Optional[int] = None) -> Dict:
    src_root, dst_root = Path(src_root), Path(dst_root)
    names = yaml.safe_load((src_root / "data.yaml").read_text())["names"]
    if dst_root.exists():
        shutil.rmtree(dst_root)
    for s in SPLITS:
        (dst_root / s / "images").mkdir(parents=True)
        (dst_root / s / "labels").mkdir(parents=True)

    frames = []
    for s in SPLITS:
        for img in sorted((src_root / s / "images").glob("*.png")):
            lab = src_root / s / "labels" / (img.stem + ".txt")
            if lab.exists():
                frames.append((s, img, lab))
    if limit:
        frames = frames[:limit]

    ship_groups: Dict[str, int] = defaultdict(int)
    for _, img, _ in frames:
        g = survey_group(img.stem)
        if g:
            ship_groups[g] += 1
    ship_split = assign_survey_splits(dict(ship_groups), seed=seed)

    n_grabcut = n_ellipse = 0
    ratios: List[float] = []
    per_split: Dict[str, int] = defaultdict(int)
    for orig_split, img_path, lab_path in frames:
        g = survey_group(img_path.stem)
        split = ship_split[g] if g else orig_split
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lines = []
        for cls, cx, cy, bw, bh in read_yolo_boxes(lab_path):
            poly, method = box_to_pseudo_polygon(img, cx, cy, bw, bh)
            if method == "grabcut":
                n_grabcut += 1
            else:
                n_ellipse += 1
            area = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1)))
            ratios.append(float(area / max(1e-9, bw * bh)))
            lines.append(f"{cls} " + " ".join(f"{v:.5f}" for v in poly.reshape(-1)))
        if not lines:
            continue
        dst_img = dst_root / split / "images" / img_path.name
        try:
            os.link(img_path, dst_img)
        except OSError:
            shutil.copy2(img_path, dst_img)
        (dst_root / split / "labels" / (img_path.stem + ".txt")).write_text("\n".join(lines) + "\n")
        per_split[split] += 1

    data = {"path": str(dst_root.resolve()), "train": "train/images", "val": "val/images", "test": "test/images",
            "nc": len(names), "names": names}
    (dst_root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    # verify the new shipwreck split really is survey-disjoint
    check: Dict[str, set] = defaultdict(set)
    for s in SPLITS:
        for img in (dst_root / s / "images").glob("Shipwrecks_*.png"):
            check[survey_group(img.stem)].add(s)
    still_leaking = {g: sorted(v) for g, v in check.items() if len(v) > 1}

    report = {
        "source": str(src_root),
        "label_type": "PSEUDO-MASKS derived from boxes (no ground-truth masks exist in the source data)",
        "leakage_before": leakage_report(src_root),
        "shipwreck_groups_leaking_after": still_leaking,
        "shipwreck_survey_split": ship_split,
        "non_shipwreck_split": "original split kept; object crops carry no survey id",
        "instances": n_grabcut + n_ellipse,
        "grabcut_masks": n_grabcut,
        "ellipse_fallback_masks": n_ellipse,
        "images_per_split": dict(per_split),
        "mean_mask_to_box_area_ratio": round(float(np.mean(ratios)), 3) if ratios else None,
        "class_names": names,
    }
    (dst_root / "dataset_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="SIH_Dataset_27class")
    ap.add_argument("--dst", default="SIH_Dataset_27class_seg")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rep = build_seg_dataset(a.src, a.dst, a.seed)
    rep.pop("class_names", None)
    rep["leakage_before"].pop("leaking", None)
    print(json.dumps(rep, indent=2))
