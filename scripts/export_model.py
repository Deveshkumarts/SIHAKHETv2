"""
Model Export Script for YOLO11 Marine Debris Detector.
Exports trained PyTorch weights (.pt) to deployment-ready formats: ONNX, TensorRT, TorchScript, OpenVINO.

Usage:
    # Standard ONNX export with FP16 and dynamic axes
    python scripts/export_model.py --model models/best.pt --format onnx --dynamic

    # TensorRT Engine export (requires NVIDIA TensorRT & CUDA)
    python scripts/export_model.py --model models/best.pt --format engine --half
"""

import argparse
import sys
import shutil
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.device_utils import select_device, get_device_info


def main():
    parser = argparse.ArgumentParser(description="Export YOLO11 Marine Debris Model for Deployment.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained PyTorch weights (.pt)")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "engine", "torchscript", "openvino", "tflite"], help="Target export format")
    parser.add_argument("--imgsz", type=int, default=640, help="Input resolution (height, width)")
    parser.add_argument("--half", action="store_true", help="Export with FP16 half precision (faster on modern GPUs)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic batch size and image dimensions (ONNX)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (12-17)")
    parser.add_argument("--device", type=str, default="auto", help="Device for export ('0', 'cpu', 'auto')")
    parser.add_argument("--save-dir", type=str, default="outputs/exported", help="Directory to save exported models")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"[Error] Model weights file not found: {model_path}")
        sys.exit(1)

    selected_device = select_device(args.device)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" 🚀 YOLO11 Model Export Engine")
    print("=" * 70)
    print(f"📦 Source Weights: {model_path}")
    print(f"🎯 Target Format:  {args.format.upper()}")
    print(f"📐 Resolution:     {args.imgsz}px | FP16 Half: {args.half} | Dynamic: {args.dynamic}")
    print(f"💻 Device:         {selected_device}")

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    try:
        exported_path_str = model.export(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
            opset=args.opset,
            device=selected_device
        )
    except Exception as e:
        print(f"\n[Export Error] {e}")
        if args.format == "engine":
            print("\n💡 Note: TensorRT ('engine') export requires NVIDIA TensorRT and matching CUDA libraries.")
        sys.exit(1)

    exported_file = Path(exported_path_str)
    destination = out_dir / exported_file.name

    # If exported into same folder as weights, copy/move to outputs/exported
    if exported_file.resolve() != destination.resolve():
        shutil.copy(str(exported_file), str(destination))

    size_mb = destination.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 70)
    print(" 🎉 EXPORT SUCCESSFUL")
    print("=" * 70)
    print(f"📁 Exported Artifact: {destination.resolve()}")
    print(f"📊 File Size:         {size_mb:.2f} MB")
    print(f"🔧 Format:            {args.format.upper()} (Opset: {args.opset if args.format == 'onnx' else 'N/A'})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
