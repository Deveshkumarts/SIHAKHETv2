"""
Error Analysis and Diagnostic Workflow for Marine Debris Detection.
Identifies and categorizes:
1. False Positives (background/reflections/rocks detected as debris)
2. False Negatives (un-detected marine debris)
3. Low-Confidence Detections (uncertain predictions)

Usage:
    python scripts/analyze_errors.py --model models/best.pt --data dataset/data.yaml --split val --conf 0.35 --iou 0.5
"""

import argparse
import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dataset_utils import parse_data_yaml, check_single_annotation, VALID_IMAGE_EXTENSIONS
from utils.visualization import draw_bounding_box, get_class_color
from utils.device_utils import select_device


def box_iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) of two boxes in (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def main():
    parser = argparse.ArgumentParser(description="Analyze False Positives, False Negatives, and Low-Confidence Detections.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU matching threshold between GT and Prediction")
    parser.add_argument("--low-conf-range", type=float, nargs=2, default=[0.15, 0.40], help="Low-confidence range [min, max]")
    parser.add_argument("--max-images", type=int, default=50, help="Maximum error images to save per category")
    parser.add_argument("--save-dir", type=str, default="outputs/error_analysis", help="Base directory to save error analysis")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"[Error] Model weights not found: {model_path}")
        sys.exit(1)

    cfg = parse_data_yaml(args.data)
    root = cfg["resolved_root"]
    class_names = cfg["names_dict"]
    num_classes = cfg["num_classes"]

    split_rel = cfg.get(args.split)
    images_dir = (root / split_rel).resolve()
    if "images" in images_dir.parts:
        parts = list(images_dir.parts)
        idx = parts.index("images")
        parts[idx] = "labels"
        labels_dir = Path(*parts)
    else:
        labels_dir = images_dir.parent / "labels" / images_dir.name

    from ultralytics import YOLO
    model = YOLO(str(model_path))

    fp_dir = Path(args.save_dir) / "false_positives"
    fn_dir = Path(args.save_dir) / "false_negatives"
    lc_dir = Path(args.save_dir) / "low_confidence"
    loc_dir = Path(args.save_dir) / "poor_localization"
    for d in [fp_dir, fn_dir, lc_dir, loc_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(" 🔬 YOLO11 Sea Debris Error Analysis & Diagnostic Engine")
    print("=" * 75)
    print(f"📦 Model: {model_path.name} | Conf Threshold: {args.conf} | Matching IoU: {args.iou}")
    print(f"📂 Analyzing '{args.split}' split from: {images_dir}\n")

    image_files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
    if not image_files:
        print("[Warning] No images found in dataset split.")
        sys.exit(0)

    total_gt_boxes = 0
    total_pred_boxes = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_poor_loc = 0
    total_low_conf = 0

    fn_by_class = {cid: 0 for cid in range(num_classes)}
    gt_by_class = {cid: 0 for cid in range(num_classes)}
    fn_by_size = {"very_small": 0, "small": 0, "medium": 0, "large": 0}
    gt_by_size = {"very_small": 0, "small": 0, "medium": 0, "large": 0}

    saved_fp = 0
    saved_fn = 0
    saved_lc = 0
    saved_loc = 0

    for img_path in image_files:
        orig_img = cv2.imread(str(img_path))
        if orig_img is None:
            continue
        h, w = orig_img.shape[:2]

        # 1. Parse Ground Truth
        label_file = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if label_file.exists():
            _, _, parsed = check_single_annotation(label_file, num_classes=num_classes)
            for cid, cx, cy, bw, bh in parsed:
                x1 = (cx - bw / 2.0) * w
                y1 = (cy - bh / 2.0) * h
                x2 = (cx + bw / 2.0) * w
                y2 = (cy + bh / 2.0) * h
                
                box_area = bw * bh
                if box_area < 0.002:
                    size_cat = "very_small"
                elif box_area < 0.01:
                    size_cat = "small"
                elif box_area < 0.05:
                    size_cat = "medium"
                else:
                    size_cat = "large"

                gt_boxes.append({"cls": cid, "box": np.array([x1, y1, x2, y2]), "size_cat": size_cat, "norm_wh": (bw, bh)})
                gt_by_class[cid] += 1
                gt_by_size[size_cat] += 1
        total_gt_boxes += len(gt_boxes)

        # 2. Run Inference with low confidence threshold to capture low-conf predictions
        results = model.predict(
            source=str(img_path),
            conf=min(args.low_conf_range[0], args.conf),
            verbose=False
        )[0]

        pred_boxes = []
        low_conf_preds = []
        for box in results.boxes:
            conf_val = float(box.conf[0])
            cid = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy()

            if args.low_conf_range[0] <= conf_val < args.conf:
                low_conf_preds.append({"cls": cid, "conf": conf_val, "box": xyxy})

            if conf_val >= args.conf:
                pred_boxes.append({"cls": cid, "conf": conf_val, "box": xyxy})

        total_pred_boxes += len(pred_boxes)
        total_low_conf += len(low_conf_preds)

        # 3. Match Predictions to Ground Truth
        matched_gt = set()
        matched_pred = set()
        poor_loc_indices = []

        for p_idx, pred in enumerate(pred_boxes):
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gt in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                if pred["cls"] == gt["cls"]:
                    iou_val = box_iou_xyxy(pred["box"], gt["box"])
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = g_idx

            if best_iou >= args.iou and best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
                total_tp += 1
            elif 0.1 <= best_iou < args.iou and best_gt_idx != -1:
                # Poor localization
                poor_loc_indices.append((p_idx, best_gt_idx, best_iou))
                total_poor_loc += 1

        # False positives: predictions that didn't match any GT
        fp_indices = [i for i in range(len(pred_boxes)) if i not in matched_pred]
        total_fp += len(fp_indices)

        # False negatives: GT boxes that were never matched
        fn_indices = [i for i in range(len(gt_boxes)) if i not in matched_gt]
        total_fn += len(fn_indices)

        for f_idx in fn_indices:
            gt = gt_boxes[f_idx]
            fn_by_class[gt["cls"]] += 1
            fn_by_size[gt["size_cat"]] += 1

        # Save Visualizations
        # 1. False Positives
        if len(fp_indices) > 0 and saved_fp < args.max_images:
            vis = orig_img.copy()
            for g_idx, gt in enumerate(gt_boxes):
                cname = class_names.get(gt["cls"], str(gt["cls"]))
                draw_bounding_box(vis, gt["box"], f"GT: {cname}", (0, 255, 0), line_thickness=1)
            for f_idx in fp_indices:
                p = pred_boxes[f_idx]
                cname = class_names.get(p["cls"], str(p["cls"]))
                draw_bounding_box(vis, p["box"], f"FP: {cname} {p['conf']:.2f}", (0, 0, 255), line_thickness=3)
            cv2.imwrite(str(fp_dir / f"fp_{img_path.name}"), vis)
            saved_fp += 1

        # 2. False Negatives
        if len(fn_indices) > 0 and saved_fn < args.max_images:
            vis = orig_img.copy()
            for f_idx in fn_indices:
                gt = gt_boxes[f_idx]
                cname = class_names.get(gt["cls"], str(gt["cls"]))
                draw_bounding_box(vis, gt["box"], f"MISSED (FN): {cname} [{gt['size_cat']}]", (0, 165, 255), line_thickness=3)
            for p in pred_boxes:
                cname = class_names.get(p["cls"], str(p["cls"]))
                draw_bounding_box(vis, p["box"], f"Pred: {cname} {p['conf']:.2f}", (0, 255, 0), line_thickness=1)
            cv2.imwrite(str(fn_dir / f"fn_{img_path.name}"), vis)
            saved_fn += 1

        # 3. Poor Localization
        if len(poor_loc_indices) > 0 and saved_loc < args.max_images:
            vis = orig_img.copy()
            for p_idx, g_idx, iou_val in poor_loc_indices:
                gt = gt_boxes[g_idx]
                pred = pred_boxes[p_idx]
                cname = class_names.get(gt["cls"], str(gt["cls"]))
                draw_bounding_box(vis, gt["box"], f"GT: {cname}", (0, 255, 0), line_thickness=2)
                draw_bounding_box(vis, pred["box"], f"PoorLoc IoU={iou_val:.2f}", (255, 165, 0), line_thickness=2)
            cv2.imwrite(str(loc_dir / f"loc_{img_path.name}"), vis)
            saved_loc += 1

        # 4. Low Confidence Detections
        if len(low_conf_preds) > 0 and saved_lc < args.max_images:
            vis = orig_img.copy()
            for lc in low_conf_preds:
                cname = class_names.get(lc["cls"], str(lc["cls"]))
                draw_bounding_box(vis, lc["box"], f"UNCERTAIN: {cname} {lc['conf']:.2f}", (255, 0, 255), line_thickness=2)
            cv2.imwrite(str(lc_dir / f"lc_{img_path.name}"), vis)
            saved_lc += 1

    # Print Summary Report
    precision = total_tp / total_pred_boxes if total_pred_boxes > 0 else 0.0
    recall = total_tp / total_gt_boxes if total_gt_boxes > 0 else 0.0

    print("=" * 75)
    print(" 📋 ERROR ANALYSIS DIAGNOSTIC SUMMARY")
    print("=" * 75)
    print(f"   • Total Ground Truth Targets: {total_gt_boxes}")
    print(f"   • Total Detections (>= {args.conf}):  {total_pred_boxes}")
    print(f"   • True Positives (TP):        {total_tp}")
    print(f"   • False Positives (FP):       {total_fp}  (Spurious background/rock detections)")
    print(f"   • False Negatives (FN):       {total_fn}  (Missed targets)")
    print(f"   • Poor Localization (IoU 0.1-0.5): {total_poor_loc}")
    print(f"   • Low-Confidence Predictions: {total_low_conf} (Conf {args.low_conf_range[0]}-{args.conf})")
    print(f"   • Precision:                  {precision:.4f} ({precision*100:.2f}%)")
    print(f"   • Recall:                     {recall:.4f} ({recall*100:.2f}%)")

    print("\n🔍 False Negatives (Missed Objects) by Class:")
    for cid in range(num_classes):
        cname = class_names.get(cid, str(cid))
        missed = fn_by_class[cid]
        total_c = max(gt_by_class[cid], 1)
        miss_rate = missed / total_c * 100
        print(f"   • [{cid}] {cname:<12} : {missed:>3} / {gt_by_class[cid]:>3} missed ({miss_rate:5.1f}% Miss Rate)")

    print("\n📐 False Negatives (Missed Objects) by Size:")
    for size_cat, missed in fn_by_size.items():
        total_s = max(gt_by_size[size_cat], 1)
        miss_rate = missed / total_s * 100
        print(f"   • {size_cat:<12} : {missed:>3} / {gt_by_size[size_cat]:>3} missed ({miss_rate:5.1f}% Miss Rate)")

    print("=" * 75)
    print(f"\n🖼️ Diagnostic Visuals Exported:")
    print(f"   - False Positives:     {fp_dir.resolve()}")
    print(f"   - False Negatives:     {fn_dir.resolve()}")
    print(f"   - Poor Localization:   {loc_dir.resolve()}")
    print(f"   - Low Confidence:      {lc_dir.resolve()}\n")

    summary_json = {
        "conf_threshold": args.conf,
        "matching_iou": args.iou,
        "total_gt_boxes": total_gt_boxes,
        "total_pred_boxes": total_pred_boxes,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "poor_localization": total_poor_loc,
        "low_confidence_count": total_low_conf,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fn_by_class": {class_names.get(c, str(c)): fn_by_class[c] for c in fn_by_class},
        "gt_by_class": {class_names.get(c, str(c)): gt_by_class[c] for c in gt_by_class},
        "fn_by_size": fn_by_size,
        "gt_by_size": gt_by_size,
    }

    report_path = Path(args.save_dir) / "error_analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)


if __name__ == "__main__":
    main()

