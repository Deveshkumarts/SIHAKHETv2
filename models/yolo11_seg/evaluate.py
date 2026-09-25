"""
Evaluate YOLO11-Seg on the held-out test split:   python -m models.yolo11_seg.evaluate

Reports  Precision, Recall, F1, mAP@50, mAP@50:95 (box AND mask), Mask IoU, latency, FPS, model size and
memory/GPU use, plus per-class AP.

CAVEAT (also written into the JSON): the source data has no ground-truth masks. "Mask" metrics are measured
against PSEUDO-MASKS (GrabCut / ellipse from boxes), so they quantify agreement with those pseudo-labels,
not true pixel accuracy. Box metrics are against real human-drawn boxes.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from backend.config import ROOT_DIR, load_config


def _poly_mask(poly_norm, h, w):
    m = np.zeros((h, w), np.uint8)
    pts = np.array(poly_norm, np.float32).reshape(-1, 2) * [w, h]
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m


def _box_iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def mask_iou_report(model, data_root: Path, split: str, imgsz: int, device: str, conf: float, limit=None):
    imgs = sorted((data_root / split / "images").glob("*.png"))
    if limit:
        imgs = imgs[:limit]
    ious, matched, total_gt = [], 0, 0
    for p in imgs:
        lab = data_root / split / "labels" / (p.stem + ".txt")
        if not lab.exists():
            continue
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        gts = []
        for line in lab.read_text().strip().splitlines():
            v = line.split()
            cls, pts = int(v[0]), np.array(v[1:], np.float32).reshape(-1, 2)
            gts.append((cls, [pts[:, 0].min() * w, pts[:, 1].min() * h, pts[:, 0].max() * w, pts[:, 1].max() * h], pts))
        total_gt += len(gts)
        res = model.predict(im, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        if res.masks is None:
            continue
        used = set()
        for k, box in enumerate(res.boxes):
            pc, pb = int(box.cls[0]), box.xyxy[0].cpu().numpy().tolist()
            best, bj = 0.0, None
            for j, (gc, gb, _) in enumerate(gts):
                if j in used or gc != pc:
                    continue
                iou = _box_iou(pb, gb)
                if iou > best:
                    best, bj = iou, j
            if bj is None or best < 0.5:
                continue
            used.add(bj)
            pm = np.zeros((h, w), np.uint8)
            cv2.fillPoly(pm, [np.round(res.masks.xy[k]).astype(np.int32)], 1)
            gm = _poly_mask(gts[bj][2], h, w)
            inter, union = int((pm & gm).sum()), int((pm | gm).sum())
            ious.append(inter / union if union else 0.0)
            matched += 1
    return {"n_images": len(imgs), "n_gt_instances": total_gt, "n_matched": matched,
            "mean_mask_iou": float(np.mean(ious)) if ious else None,
            "frac_mask_iou_ge_0.5": float(np.mean(np.array(ious) >= 0.5)) if ious else None,
            "reference": "pseudo-masks (no ground-truth masks exist)"}


def latency_report(model, images, imgsz, device, n_warm=5):
    for im in images[:n_warm]:
        model.predict(im, imgsz=imgsz, device=device, verbose=False)
    ts = []
    for im in images:
        t0 = time.perf_counter()
        model.predict(im, imgsz=imgsz, device=device, verbose=False)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    return {"n_images": len(ts), "mean_ms": float(ts.mean()), "p50_ms": float(np.percentile(ts, 50)),
            "p95_ms": float(np.percentile(ts, 95)), "fps": float(1000.0 / ts.mean()), "batch": 1}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="SIH_Dataset_27class_seg")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--latency-images", type=int, default=100)
    ap.add_argument("--mask-iou-limit", type=int, default=None)
    a = ap.parse_args()

    import psutil
    import torch
    from ultralytics import YOLO

    cfg = load_config()
    weights = cfg.path("yolo_seg.weights")
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")
    data_root = ROOT_DIR / a.data
    proc = psutil.Process(os.getpid())
    rss0 = proc.memory_info().rss
    model = YOLO(str(weights))
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    val = model.val(data=str(data_root / "data.yaml"), split=a.split, imgsz=a.imgsz, device=a.device, batch=16,
                    conf=0.001, iou=0.6, plots=False, verbose=False, project=str(ROOT_DIR / "outputs" / "evaluation"),
                    name="yolo11_seg_val", exist_ok=True)

    def block(m):
        p, r = float(m.mp), float(m.mr)
        return {"precision": p, "recall": r, "f1": float(2 * p * r / (p + r)) if p + r else 0.0,
                "map50": float(m.map50), "map50_95": float(m.map)}

    names = model.names
    per_class = {names[int(c)]: {"ap50": float(val.box.ap50[i]), "ap50_95": float(val.box.ap[i]),
                                 "mask_ap50": float(val.seg.ap50[i]), "mask_ap50_95": float(val.seg.ap[i])}
                 for i, c in enumerate(val.box.ap_class_index)}

    test_imgs = [cv2.imread(str(p)) for p in sorted((data_root / a.split / "images").glob("*.png"))[: a.latency_images]]
    lat = latency_report(model, test_imgs, a.imgsz, a.device)
    miou = mask_iou_report(model, data_root, a.split, a.imgsz, a.device, a.conf, a.mask_iou_limit)

    n_params = sum(p.numel() for p in model.model.parameters())
    report = {
        "weights": str(weights.relative_to(ROOT_DIR)), "split": a.split, "imgsz": a.imgsz, "device": a.device,
        "dataset": json.loads((data_root / "dataset_report.json").read_text()) if (data_root / "dataset_report.json").exists() else None,
        "box": block(val.box), "mask": block(val.seg), "mask_iou": miou, "latency": lat,
        "model": {"size_mb": round(weights.stat().st_size / 1e6, 2), "parameters": int(n_params)},
        "memory": {"process_rss_delta_mb": round((proc.memory_info().rss - rss0) / 1e6, 1),
                   "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if a.device.startswith("cuda") else None,
                   "gpu_note": None if a.device.startswith("cuda") else "CPU evaluation - no GPU utilisation to report"},
        "per_class": per_class,
        "caveat": "Mask metrics are computed against pseudo-masks derived from boxes; box metrics use real annotations.",
    }
    out = ROOT_DIR / "outputs" / "evaluation" / "yolo11_seg_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("box", "mask", "mask_iou", "latency", "model", "memory")}, indent=2))


if __name__ == "__main__":
    main()
