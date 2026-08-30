"""
YOLO11 Fine-Tuning Pipeline for Marine / Sea Debris Detection.
Loads configuration, validates dataset integrity, applies marine-specific augmentations,
and logs reproducibility metadata alongside training artifacts.

Usage:
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --model yolo11s.pt --epochs 50 --batch 16 --imgsz 640
"""

import argparse
import sys
import os
import json
import yaml
from pathlib import Path
from datetime import datetime

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.device_utils import get_device_info, select_device
from utils.dataset_utils import validate_yolo_dataset, parse_data_yaml


def load_config(config_path: str) -> dict:
    """Load training config YAML with fallback defaults."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        print(f"[Warning] Config file '{config_path}' not found. Using default parameters.")
        return {}
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11 model on Marine Debris dataset.")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to training config YAML")
    parser.add_argument("--model", type=str, default=None, help="YOLO11 checkpoint (e.g. yolo11s.pt, yolo11n.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Image size (640, 832, 1024)")
    parser.add_argument("--device", type=str, default=None, help="Device ('auto', 'cpu', '0', etc.)")
    parser.add_argument("--name", type=str, default=None, help="Run experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--skip-validation", action="store_true", help="Skip pre-training dataset health audit")
    args = parser.parse_args()

    # 1. Load config file and merge CLI overrides
    file_cfg = load_config(args.config)

    model_name = args.model or file_cfg.get("model", "yolo11s.pt")
    data_yaml = args.data or file_cfg.get("data", "dataset/data.yaml")
    epochs = args.epochs if args.epochs is not None else file_cfg.get("epochs", 100)
    batch_size = args.batch if args.batch is not None else file_cfg.get("batch", 16)
    imgsz = args.imgsz if args.imgsz is not None else file_cfg.get("imgsz", 640)
    requested_device = args.device or file_cfg.get("device", "auto")
    run_name = args.name or file_cfg.get("name", f"sea_debris_{Path(model_name).stem}_{imgsz}px")
    project_dir = file_cfg.get("project", "outputs/training")

    print("\n" + "=" * 75)
    print(" 🚀 YOLO11 Sea Debris Detection Fine-Tuning Pipeline")
    print("=" * 75)

    # 2. Hardware & Device Selection
    device_info = get_device_info()
    selected_device = select_device(requested_device)
    print(f"💻 Python: {device_info['python_version']} | PyTorch: {device_info['torch_version']}")
    print(f"🎯 Target Device: {selected_device} ({device_info['gpu_name']})")
    print(f"📦 Model: {model_name} | Image Size: {imgsz} | Batch Size: {batch_size} | Epochs: {epochs}")

    # 3. Pre-training Dataset Validation
    if not args.skip_validation:
        print("\n🔍 Validating dataset integrity before starting...")
        report = validate_yolo_dataset(data_yaml)
        if not report["is_healthy"]:
            print("\n❌ Dataset validation failed! Fix corrupt images or invalid annotations before training.")
            sys.exit(1)
        print(f"✅ Dataset validated: {report['total_images']} images, {report['num_classes']} classes, {report['total_boxes']} objects.")

    # 4. Import Ultralytics YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[Error] Ultralytics is not installed. Please run: pip install ultralytics")
        sys.exit(1)

    # 5. Initialize Model
    print(f"\n📥 Loading pretrained checkpoint: {model_name}...")
    model = YOLO(model_name)

    # 6. Extract Augmentations & Hyperparameters from config
    aug_params = {
        "hsv_h": file_cfg.get("hsv_h", 0.015),
        "hsv_s": file_cfg.get("hsv_s", 0.5),
        "hsv_v": file_cfg.get("hsv_v", 0.4),
        "degrees": file_cfg.get("degrees", 10.0),
        "translate": file_cfg.get("translate", 0.1),
        "scale": file_cfg.get("scale", 0.5),
        "shear": file_cfg.get("shear", 2.0),
        "perspective": file_cfg.get("perspective", 0.0001),
        "flipud": file_cfg.get("flipud", 0.0),
        "fliplr": file_cfg.get("fliplr", 0.5),
        "mosaic": file_cfg.get("mosaic", 1.0),
        "mixup": file_cfg.get("mixup", 0.1),
        "copy_paste": file_cfg.get("copy_paste", 0.1),
        "erasing": file_cfg.get("erasing", 0.2),
        "optimizer": file_cfg.get("optimizer", "auto"),
        "lr0": file_cfg.get("lr0", 0.01),
        "lrf": file_cfg.get("lrf", 0.01),
        "patience": file_cfg.get("patience", 30),
        "amp": file_cfg.get("amp", True) if selected_device != "cpu" else False,
        "workers": file_cfg.get("workers", 4),
        "plots": file_cfg.get("plots", True),
        "verbose": file_cfg.get("verbose", True),
    }

    # 7. Prepare Run Directory & Experiment Metadata
    save_run_dir = Path(project_dir) / run_name
    save_run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "data_yaml": str(Path(data_yaml).resolve()),
        "epochs": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "selected_device": selected_device,
        "hardware": device_info,
        "hyperparameters": aug_params,
    }

    metadata_path = save_run_dir / "experiment_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 8. Start Fine-Tuning
    print("\n🔥 Starting YOLO11 Fine-Tuning...")
    try:
        results = model.train(
            data=str(Path(data_yaml).resolve()),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=selected_device,
            project=str(Path(project_dir).resolve()),
            name=run_name,
            exist_ok=True,
            resume=args.resume,
            **aug_params
        )
    except Exception as e:
        print(f"\n[Training Error] {e}")
        if "out of memory" in str(e).lower():
            print("\n💡 Tip: CUDA Out-Of-Memory encountered. Try reducing batch size (e.g. --batch 8 or --batch 4) or image size (--imgsz 640).")
        sys.exit(1)

    # 9. Print Training Summary
    best_pt = save_run_dir / "weights" / "best.pt"
    last_pt = save_run_dir / "weights" / "last.pt"

    print("\n" + "=" * 75)
    print(" 🎉 TRAINING COMPLETE")
    print("=" * 75)
    print(f"📁 Output Run Directory: {save_run_dir.resolve()}")
    if best_pt.exists():
        print(f"⭐ Best Model Weights:    {best_pt.resolve()}")
    if last_pt.exists():
        print(f"💾 Last Model Weights:    {last_pt.resolve()}")
    print(f"📋 Experiment Metadata:  {metadata_path.resolve()}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
