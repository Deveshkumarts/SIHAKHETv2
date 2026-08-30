"""
Comprehensive Model Evaluation Script for Sea Debris YOLO11.
Evaluates precision, recall, mAP@50, mAP@50-95, per-class breakdown, latency benchmarks,
and exports summary reports and metric charts.

Usage:
    python scripts/evaluate.py --model models/best.pt --data dataset/data.yaml --split val
    python scripts/evaluate.py --model outputs/training/baseline_yolo11s_640px/weights/best.pt --split test
"""

import argparse
import sys
import os
import json
from pathlib import Path
import pandas as pd

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.device_utils import get_device_info, select_device
from utils.dataset_utils import parse_data_yaml
from utils.visualization import plot_metrics_summary


def run_evaluation(
    model_path: str,
    data_yaml: str = "dataset/data.yaml",
    split: str = "val",
    imgsz: int = 640,
    batch: int = 16,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.6,
    device_str: str = "auto",
    save_dir: str = "outputs/evaluation",
    save_name: str = None
) -> dict:
    mpath = Path(model_path).resolve()
    if not mpath.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {mpath}")

    data_cfg = parse_data_yaml(data_yaml)
    class_names = data_cfg["names_dict"]

    selected_device = select_device(device_str)
    save_run_name = save_name or f"eval_{mpath.stem}_{split}_{imgsz}px"
    out_dir = Path(save_dir) / save_run_name if not save_name or not save_name.startswith("outputs/") else Path(save_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(" 📊 YOLO11 Sea Debris Model Evaluation")
    print("=" * 75)
    print(f"📦 Model:  {mpath}")
    print(f"📂 Data:   {data_yaml} (Split: {split})")
    print(f"🎯 Device: {selected_device} | Resolution: {imgsz}px | Conf: {conf_thresh} | IoU: {iou_thresh}")

    from ultralytics import YOLO
    model = YOLO(str(mpath))

    # Run validation
    print(f"\n🔍 Running evaluation on '{split}' set...")
    metrics = model.val(
        data=str(Path(data_yaml).resolve()),
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=conf_thresh,
        iou=iou_thresh,
        device=selected_device,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        plots=True,
        save_json=True
    )

    # Extract overall metrics
    p = float(metrics.box.mp) if hasattr(metrics.box, "mp") else 0.0
    r = float(metrics.box.mr) if hasattr(metrics.box, "mr") else 0.0
    map50 = float(metrics.box.map50) if hasattr(metrics.box, "map50") else 0.0
    map50_95 = float(metrics.box.map) if hasattr(metrics.box, "map") else 0.0

    # Extract speed metrics
    speed = metrics.speed  # dictionary with preprocess, inference, loss, postprocess in ms
    inf_speed = speed.get("inference", 0.0)
    fps = 1000.0 / inf_speed if inf_speed > 0 else 0.0

    print("\n" + "=" * 75)
    print(" 🏆 OVERALL DETECTION METRICS")
    print("=" * 75)
    print(f"   • Precision (P):       {p:>7.4f} ({p*100:.2f}%)")
    print(f"   • Recall (R):          {r:>7.4f} ({r*100:.2f}%)")
    print(f"   • mAP@50:              {map50:>7.4f} ({map50*100:.2f}%)")
    print(f"   • mAP@50-95:           {map50_95:>7.4f} ({map50_95*100:.2f}%)")
    print(f"   • Inference Latency:   {inf_speed:>7.2f} ms/image (~{fps:.1f} FPS)")
    print("=" * 75)

    # Extract per-class breakdown
    class_metrics = {}
    print("\n🏷️ PER-CLASS BREAKDOWN:")
    print(f"{'Class ID':<9} | {'Class Name':<18} | {'Precision':<10} | {'Recall':<10} | {'mAP@50':<10} | {'mAP@50-95':<10}")
    print("-" * 75)

    try:
        maps_per_class = metrics.box.maps if hasattr(metrics.box, "maps") and len(metrics.box.maps) > 0 else []
        ap50_per_class = metrics.box.ap50 if hasattr(metrics.box, "ap50") and len(metrics.box.ap50) > 0 else []
        p_per_class = metrics.box.p if hasattr(metrics.box, "p") and len(metrics.box.p) > 0 else []
        r_per_class = metrics.box.r if hasattr(metrics.box, "r") and len(metrics.box.r) > 0 else []

        for i, (cid, cname) in enumerate(class_names.items()):
            cp = float(p_per_class[i]) if i < len(p_per_class) else p
            cr = float(r_per_class[i]) if i < len(r_per_class) else r
            cmap50 = float(ap50_per_class[i]) if i < len(ap50_per_class) else map50
            cmap = float(maps_per_class[i]) if i < len(maps_per_class) else map50_95

            class_metrics[cname] = {
                "class_id": cid,
                "precision": round(cp, 4),
                "recall": round(cr, 4),
                "map50": round(cmap50, 4),
                "map50_95": round(cmap, 4),
            }
            print(f"[{cid}]      | {cname:<18} | {cp:<10.4f} | {cr:<10.4f} | {cmap50:<10.4f} | {cmap:<10.4f}")
    except Exception as e:
        print(f"[Note] Per-class extraction fallback: {e}")

    print("=" * 75)

    eval_report = {
        "model": str(mpath),
        "split": split,
        "imgsz": imgsz,
        "conf_threshold": conf_thresh,
        "iou_threshold": iou_thresh,
        "device": selected_device,
        "overall": {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "inference_speed_ms": round(inf_speed, 2),
            "fps": round(fps, 2),
        },
        "per_class": class_metrics,
        "speed_profile_ms": speed,
    }

    report_json_path = out_dir / "evaluation_summary.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    chart_path = out_dir / "metrics_summary_chart.png"
    plot_metrics_summary(eval_report["overall"], class_metrics, chart_path)

    print(f"\n✅ Evaluation report saved to: {report_json_path.resolve()}")
    print(f"📊 Summary chart saved to:   {chart_path.resolve()}\n")

    return eval_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO11 Sea Debris Detection Model.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for evaluation")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cpu', '0')")
    parser.add_argument("--save-dir", type=str, default="outputs/evaluation", help="Directory to save evaluation artifacts")
    parser.add_argument("--name", type=str, default=None, help="Evaluation run subdirectory name")
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        device_str=args.device,
        save_dir=args.save_dir,
        save_name=args.name
    )


if __name__ == "__main__":
    main()

