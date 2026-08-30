"""
Train YOLO11s on the Unified 7-Class Sonar Dataset (1,867 images).
Classes:
  [0] shipwreck
  [1] airplane
  [2] mine
  [3] human
  [4] engineering platform
  [5] pipeline or cable
  [6] underwater residual mound
"""

import sys
import json
import time
from pathlib import Path
from ultralytics import YOLO

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.device_utils import select_device


def train_unified_yolo(
    data_yaml: str = "Unified_Sonar_Dataset/data.yaml",
    weights: str = "yolo11s.pt",
    epochs: int = 50,
    imgsz: int = 832,
    batch_size: int = 8,
    project: str = "outputs/training",
    name: str = "yolo11s_unified_7class",
    device_str: str = "0",
    conf_thresh: float = 0.15
):
    selected_device = select_device(device_str)

    print("\n" + "=" * 80)
    print(" 🚀 TRAINING YOLO11s ON UNIFIED 7-CLASS UNDERWATER SONAR DATASET")
    print("=" * 80)
    print(f"📦 Dataset:       {data_yaml} (1,867 sonar images)")
    print(f"📦 Base Model:    {weights}")
    print(f"🎯 Target Device: {selected_device}")
    print(f"⚙️ Config:        {epochs} epochs | batch {batch_size} | {imgsz}x{imgsz} px | conf {conf_thresh}")
    print("=" * 80 + "\n")

    model = YOLO(weights)

    # Train
    train_results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=selected_device,
        project=project,
        name=name,
        exist_ok=True,
        plots=True,
        save=True,
        amp=True,
        workers=2,
        seed=42,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.005,
        hsv_s=0.3,
        hsv_v=0.3,
    )

    best_weights = Path(project) / name / "weights" / "best.pt"
    print("\n" + "=" * 80)
    print(" 🔍 RUNNING TEST-SET EVALUATION ON UNIFIED MODEL")
    print("=" * 80)

    best_model = YOLO(str(best_weights))
    val_results = best_model.val(
        data=data_yaml,
        split="test",
        imgsz=imgsz,
        conf=conf_thresh,
        iou=0.5,
        device=selected_device,
        project=project,
        name=f"{name}_test_eval",
        exist_ok=True,
        save_json=True
    )

    metrics = val_results.results_dict
    prec = metrics.get("metrics/precision(B)", 0.0)
    rec = metrics.get("metrics/recall(B)", 0.0)
    map50 = metrics.get("metrics/mAP50(B)", 0.0)
    map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
    speed_ms = val_results.speed.get("inference", 0.0)
    fps = 1000.0 / max(1e-6, speed_ms)

    print("\n" + "=" * 80)
    print(" 🏆 UNIFIED 7-CLASS MODEL TEST METRICS")
    print("=" * 80)
    print(f"   • Precision (P):        {prec*100:6.2f}%")
    print(f"   • Recall (R):           {rec*100:6.2f}%")
    print(f"   • mAP@50:               {map50*100:6.2f}%")
    print(f"   • mAP@50-95:            {map50_95*100:6.2f}%")
    print(f"   • Inference Latency:    {speed_ms:6.2f} ms (~{fps:.1f} FPS)")
    print("=" * 80 + "\n")

    summary = {
        "model": name,
        "classes": val_results.names,
        "epochs": epochs,
        "imgsz": imgsz,
        "conf_threshold": conf_thresh,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "latency_ms": round(speed_ms, 2),
        "fps": round(fps, 1),
        "best_weights": str(best_weights)
    }

    out_file = Path(project) / name / "unified_evaluation_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    train_unified_yolo()
