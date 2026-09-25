"""
Train YOLO11-Seg on the survey-disjoint pseudo-mask dataset.

    python -m models.yolo11_seg.train --epochs 25 --imgsz 320 --device cpu

Outputs: outputs/training/<name>/ (ultralytics run dir) and the best weights copied to
weights/yolo11n_seg_sih27.pt (path comes from configs/pipeline.yaml -> yolo_seg.weights).

IMPORTANT: the masks are pseudo-labels (GrabCut / ellipse from boxes). Mask metrics from this
training therefore measure agreement with those pseudo-masks, not true pixel accuracy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from backend.config import ROOT_DIR, load_config


def train(args) -> Path:
    from ultralytics import YOLO

    cfg = load_config()
    data = ROOT_DIR / args.data / "data.yaml"
    if not data.exists():
        raise SystemExit(f"{data} missing - run: python -m models.yolo11_seg.dataset")

    model = YOLO(args.model)
    t0 = time.time()
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device,
        workers=args.workers, patience=args.patience, project=str(ROOT_DIR / "outputs" / "training"),
        name=args.name, exist_ok=True, seed=cfg.pipeline.seed, deterministic=True, cache=args.cache,
        fraction=args.fraction, plots=True, verbose=True,
        # sonar-appropriate augmentation: no colour jitter (single-channel intensity), no vertical flip
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.25, fliplr=0.5, flipud=0.0, mosaic=1.0, degrees=0.0, close_mosaic=3,
    )
    run_dir = ROOT_DIR / "outputs" / "training" / args.name
    best = run_dir / "weights" / "best.pt"
    dst = cfg.path("yolo_seg.weights")
    if best.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst)
    (run_dir / "train_meta.json").write_text(json.dumps({
        "args": vars(args), "wall_seconds": round(time.time() - t0, 1), "label_type": "pseudo-masks",
        "weights": str(dst), "config_hash": cfg.hash}, indent=2))
    print("best weights ->", dst)
    return dst


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="SIH_Dataset_27class_seg")
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--cache", default="ram")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--name", default="yolo11n_seg_sih27")
    train(ap.parse_args())
