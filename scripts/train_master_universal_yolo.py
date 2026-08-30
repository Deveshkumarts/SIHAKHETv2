"""
Training Script for Master Universal 24-Class Marine & Sonar Object Detector (YOLO11).
Trains across all 24 unified classes (Sonar + Debris/Turntable).
"""

import sys
from pathlib import Path
from ultralytics import YOLO

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def train_master():
    model = YOLO("yolo11s.pt")
    print("\n" + "=" * 75)
    print(" 🚀 STARTING MASTER UNIVERSAL 24-CLASS YOLO11 TRAINING")
    print("=" * 75)

    results = model.train(
        data="Master_Universal_24class_YOLO/data.yaml",
        epochs=40,
        imgsz=640,
        batch=16,
        device="0",
        workers=2,
        project="runs/detect/outputs/training",
        name="yolo11s_master_universal_24class",
        exist_ok=True,
        seed=42,
        patience=15,
        optimizer="auto",
        mixup=0.1,
        close_mosaic=10,
        verbose=True,
    )
    print("=" * 75)
    print(" ✅ MASTER UNIVERSAL TRAINING COMPLETE!")
    print("=" * 75 + "\n")
    return results


if __name__ == "__main__":
    train_master()
