"""
Controlled Experiment Runner for YOLO11 Underwater Sonar Detector.
Executes training runs for resolution, augmentation, and preprocessing experiments,
and automatically runs test-set evaluation post-training.
"""

import argparse
import sys
import json
import shutil
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate import run_evaluation
from utils.sonar_preprocess import create_preprocessed_dataset


def run_experiment(
    exp_name: str,
    imgsz: int = 640,
    batch: int = 16,
    epochs: int = 50,
    data_yaml: str = "Combined_Dataset/data.yaml",
    model_name: str = "yolo11s.pt",
    extra_aug: dict = None,
    preprocessed_mode: str = None
) -> dict:
    from ultralytics import YOLO

    exp_dir = Path("outputs/experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 85)
    print(f" 🚀 RUNNING EXPERIMENT: {exp_name}")
    print("=" * 85)
    print(f"📦 Model: {model_name} | Resolution: {imgsz}px | Batch: {batch} | Epochs: {epochs}")

    # Handle preprocessing if requested
    actual_data_yaml = data_yaml
    if preprocessed_mode:
        prep_dataset_dir = f"Combined_Dataset_preprocessed_{preprocessed_mode}"
        print(f"⚙️ Applying '{preprocessed_mode}' preprocessing to dataset -> {prep_dataset_dir}...")
        actual_data_yaml = create_preprocessed_dataset(data_yaml, prep_dataset_dir, mode=preprocessed_mode)
        print(f"✅ Preprocessed dataset created: {actual_data_yaml}")

    # Set up training arguments
    aug_params = {
        "hsv_h": 0.015, "hsv_s": 0.5, "hsv_v": 0.4,
        "degrees": 10.0, "translate": 0.1, "scale": 0.5,
        "shear": 2.0, "perspective": 0.0001, "flipud": 0.0,
        "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.1,
        "copy_paste": 0.1, "erasing": 0.2,
        "optimizer": "auto", "lr0": 0.01, "lrf": 0.01,
        "patience": 30, "amp": True, "workers": 4,
        "plots": True, "verbose": True
    }
    if extra_aug:
        aug_params.update(extra_aug)

    # Train model
    print(f"\n🔥 Starting training run '{exp_name}'...")
    model = YOLO(model_name)
    results = model.train(
        data=str(Path(actual_data_yaml).resolve()),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=0,
        project=str(Path("outputs/experiments").resolve()),
        name=exp_name,
        exist_ok=True,
        **aug_params
    )

    best_weights = exp_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Best weights not found at: {best_weights}")

    # Evaluate on Test Set
    print(f"\n🔍 Running test-set evaluation for '{exp_name}'...")
    eval_summary = run_evaluation(
        model_path=str(best_weights),
        data_yaml=actual_data_yaml,
        split="test",
        imgsz=imgsz,
        conf_thresh=0.15,  # Use optimal conf threshold discovered in Exp 2
        iou_thresh=0.5,
        device_str="0",
        save_dir=str(exp_dir.parent),
        save_name=exp_name
    )

    # Save summary JSON directly inside experiment folder
    with open(exp_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    return eval_summary


def main():
    parser = argparse.ArgumentParser(description="Run controlled YOLO11 experiments.")
    parser.add_argument("--exp", type=str, required=True, choices=["resolution_832", "resolution_1024", "sonar_aug", "sonar_prep"], help="Experiment identifier")
    args = parser.parse_args()

    if args.exp == "resolution_832":
        run_experiment(
            exp_name="exp_resolution_832",
            imgsz=832,
            batch=8,
            epochs=50
        )
    elif args.exp == "resolution_1024":
        run_experiment(
            exp_name="exp_resolution_1024",
            imgsz=1024,
            batch=4,
            epochs=50
        )
    elif args.exp == "sonar_aug":
        run_experiment(
            exp_name="exp_sonar_aug",
            imgsz=832,
            batch=8,
            epochs=50,
            extra_aug={
                "hsv_h": 0.005,     # Minimal hue variation (greyscale sonar)
                "hsv_s": 0.3,
                "scale": 0.7,       # Strong multiscale jitter for small objects
                "mosaic": 1.0,
                "erasing": 0.25,    # Simulate acoustic shadow cutouts
                "mixup": 0.15
            }
        )
    elif args.exp == "sonar_prep":
        run_experiment(
            exp_name="exp_sonar_prep",
            imgsz=832,
            batch=8,
            epochs=50,
            preprocessed_mode="clahe_denoise"
        )


if __name__ == "__main__":
    main()
