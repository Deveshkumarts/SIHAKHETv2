"""
Image Inference Script for Sea Debris Detection with YOLO11.
Supports single image files, image folders, confidence filtering,
high-contrast bounding box drawing, and JSON prediction outputs.

Usage:
    python scripts/predict.py --model models/best.pt --source dataset/images/val/sample.jpg --conf 0.40
    python scripts/predict.py --model outputs/training/baseline_yolo11s_640px/weights/best.pt --source test_images/
"""

import argparse
import sys
import os
import json
import cv2
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.device_utils import select_device
from utils.visualization import draw_detections, draw_fps_and_stats
from utils.dataset_utils import VALID_IMAGE_EXTENSIONS


def main():
    parser = argparse.ArgumentParser(description="Run YOLO11 Marine Debris inference on images.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Path to image file or directory")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence score threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cpu', '0')")
    parser.add_argument("--save-dir", type=str, default="outputs/predictions", help="Directory to save predictions")
    parser.add_argument("--save-json", action="store_true", default=True, help="Save structured detection JSON")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"[Error] Model checkpoint not found: {model_path}")
        sys.exit(1)

    source_path = Path(args.source).resolve()
    if not source_path.exists():
        print(f"[Error] Source path not found: {source_path}")
        sys.exit(1)

    # Gather images
    if source_path.is_file():
        image_files = [source_path]
    else:
        image_files = [p for p in source_path.rglob("*") if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS]

    if not image_files:
        print(f"[Error] No valid images found at: {source_path}")
        sys.exit(1)

    selected_device = select_device(args.device)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print(" 🌊 YOLO11 Sea Debris Image Detection")
    print("=" * 70)
    print(f"📦 Model:  {model_path.name}")
    print(f"📂 Images: {len(image_files)} image(s) from {source_path}")
    print(f"🎯 Params: Conf >= {args.conf} | IoU: {args.iou} | Resolution: {args.imgsz}px")

    from ultralytics import YOLO
    model = YOLO(str(model_path))
    class_names = model.names

    all_predictions = []
    total_debris_detected = 0

    for i, img_path in enumerate(image_files, start=1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[Warning] Failed to read {img_path.name}")
            continue

        results = model.predict(
            source=image,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=selected_device,
            verbose=False
        )[0]

        boxes_xyxy = []
        scores = []
        class_ids = []
        detections_list = []

        for box in results.boxes:
            c_score = float(box.conf[0])
            cid = int(box.cls[0])
            coords = box.xyxy[0].cpu().numpy().tolist()

            boxes_xyxy.append(coords)
            scores.append(c_score)
            class_ids.append(cid)

            cname = class_names.get(cid, f"cls_{cid}")
            detections_list.append({
                "class_id": cid,
                "class_name": cname,
                "confidence": round(c_score, 4),
                "box_xyxy": [round(c, 2) for c in coords]
            })

        count = len(boxes_xyxy)
        total_debris_detected += count

        annotated = draw_detections(image, boxes_xyxy, scores, class_ids, class_names)
        out_img_path = out_dir / f"pred_{img_path.name}"
        cv2.imwrite(str(out_img_path), annotated)

        print(f"[{i}/{len(image_files)}] {img_path.name} -> {count} debris item(s) detected.")

        all_predictions.append({
            "image": str(img_path.name),
            "detection_count": count,
            "detections": detections_list,
            "output_image": str(out_img_path.name)
        })

    if args.save_json:
        json_out_path = out_dir / "predictions_summary.json"
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": model_path.name,
                "conf_threshold": args.conf,
                "total_images_processed": len(image_files),
                "total_debris_detected": total_debris_detected,
                "results": all_predictions
            }, f, indent=2)

    print("=" * 70)
    print(f"✅ Processed {len(image_files)} image(s). Total debris found: {total_debris_detected}")
    print(f"📁 Predictions saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
