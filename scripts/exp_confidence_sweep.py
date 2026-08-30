"""
Experiment 2: Confidence Threshold Sweep.
Evaluates baseline YOLO11s model across confidence thresholds [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
to determine the optimal operating point for precision, recall, and F1 score.
"""

import sys
import json
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate import run_evaluation


def main():
    model_path = "outputs/training/yolo11s_combined_dataset/weights/best.pt"
    data_yaml = "Combined_Dataset/data.yaml"
    conf_list = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]

    sweep_results = []

    print("\n" + "=" * 80)
    print(" 🔬 EXPERIMENT 2: CONFIDENCE THRESHOLD SWEEP (0.15 - 0.60)")
    print("=" * 80)

    for conf in conf_list:
        exp_name = f"exp_conf_{int(conf * 100):02d}"
        print(f"\n▶ Evaluating Conf Threshold: {conf:.2f} ...")
        
        eval_data = run_evaluation(
            model_path=model_path,
            data_yaml=data_yaml,
            split="test",
            imgsz=640,
            conf_thresh=conf,
            iou_thresh=0.5,
            device_str="0",
            save_name=f"outputs/experiments/{exp_name}"
        )

        # Ensure summary json is copied to outputs/experiments/{exp_name}/
        exp_dir = Path("outputs/experiments") / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2)

        overall = eval_data.get("overall", {})
        p = overall.get("precision", 0)
        r = overall.get("recall", 0)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        sweep_results.append({
            "conf": conf,
            "precision": p,
            "recall": r,
            "f1": f1,
            "map50": overall.get("map50", 0),
            "map50_95": overall.get("map50_95", 0),
            "fps": overall.get("fps", 0)
        })

    print("\n" + "=" * 80)
    print(" 📊 CONFIDENCE SWEEP RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Conf':<8} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10} | {'mAP@50':<10} | {'mAP@50-95':<10}")
    print("-" * 80)
    for res in sweep_results:
        print(
            f"{res['conf']:<8.2f} | {res['precision']*100:<9.2f}% | {res['recall']*100:<9.2f}% | "
            f"{res['f1']*100:<9.2f}% | {res['map50']*100:<9.2f}% | {res['map50_95']*100:<9.2f}%"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
