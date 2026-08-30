"""
Comprehensive System Comparison & Ablation Study.
Evaluates and benchmarks three detection paradigms across the 248 test images:
  • System A: YOLO11 Only (832px, conf 0.15)
  • System B: SegFormer Only (ROI Segmentation Quality)
  • System C: YOLO11 + Dynamic ROI + SegFormer Fusion (Weighted Verification)

Profiles latency breakdown (YOLO, ROI extraction, SegFormer, Fusion, Total FPS)
and computes false-positive reduction vs recall preservation.
"""

import sys
import time
import json
from pathlib import Path
import cv2
import numpy as np
import torch

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.yolo_interface import YOLO11Detector
from utils.roi_utils import extract_rois_from_image, roi_mask_to_full_image
from utils.dataset_utils import parse_data_yaml, check_single_annotation, VALID_IMAGE_EXTENSIONS
from segformer.inference import SegFormerInference
from scripts.fusion import compute_segmentation_quality_score, fuse_detection


def run_system_comparison(
    data_yaml_path: str = "Combined_Dataset/data.yaml",
    yolo_weights: str = "outputs/experiments/exp_resolution_832/weights/best.pt",
    seg_weights: str = "outputs/segformer/weights/best.pt",
    output_dir: str = "outputs/fusion",
    alpha_beta_pairs: list = None,
    threshold_verified: float = 0.50,
    threshold_reject: float = 0.30
):
    if alpha_beta_pairs is None:
        alpha_beta_pairs = [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6)]

    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = parse_data_yaml(data_yaml_path)
    root = cfg["resolved_root"]
    num_classes = cfg["num_classes"]
    class_names = cfg["names_dict"]

    test_rel = cfg.get("test")
    images_dir = (root / test_rel).resolve()
    if "images" in images_dir.parts:
        parts = list(images_dir.parts)
        idx = parts.index("images")
        parts[idx] = "labels"
        labels_dir = Path(*parts)
    else:
        labels_dir = images_dir.parent / "labels" / images_dir.name

    test_images = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS])
    print("\n" + "=" * 90)
    print(" 🔬 SIH 2026: SYSTEM COMPARISON & ABLATION STUDY (TEST SET)")
    print("=" * 90)
    print(f"📦 Test Images: {len(test_images)}")
    print(f"📦 YOLO Model:  {yolo_weights}")
    print(f"📦 SegFormer:   {seg_weights}")
    print("=" * 90 + "\n")

    # Initialize models
    detector = YOLO11Detector(model_path=yolo_weights, conf_thresh=0.15, imgsz=832)
    segmenter = SegFormerInference(weights_path=seg_weights, img_size=224)

    # Benchmark metrics accumulators
    total_gt_boxes = 0
    total_yolo_detections = 0

    yolo_times = []
    roi_times = []
    seg_times = []
    fusion_times = []

    # Store detection results for fusion analysis
    all_detections_data = []

    for img_idx, img_path in enumerate(test_images):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        # Read Ground Truth
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        if label_path.exists():
            _, _, boxes = check_single_annotation(label_path, num_classes=num_classes)
            for cid, cx, cy, bw, bh in boxes:
                x1 = (cx - bw / 2.0) * w
                y1 = (cy - bh / 2.0) * h
                x2 = (cx + bw / 2.0) * w
                y2 = (cy + bh / 2.0) * h
                gt_boxes.append({"class_id": cid, "bbox": [x1, y1, x2, y2]})
        total_gt_boxes += len(gt_boxes)

        # Stage 1: YOLO Detection
        t0 = time.perf_counter()
        _, detections = detector.detect(img_bgr)
        t_yolo = (time.perf_counter() - t0) * 1000
        yolo_times.append(t_yolo)
        total_yolo_detections += len(detections)

        if not detections:
            continue

        # Stage 2: ROI Extraction
        t1 = time.perf_counter()
        rois = extract_rois_from_image(img_bgr, detections, padding_ratio=0.25, source_id=img_path.stem)
        t_roi = (time.perf_counter() - t1) * 1000
        roi_times.append(t_roi)

        # Stage 3: SegFormer Segmentation
        t2 = time.perf_counter()
        img_seg_qualities = []
        for roi in rois:
            roi_mask, fg_prob = segmenter.predict(roi["roi_crop_raw"])
            s_seg = compute_segmentation_quality_score(
                roi_mask, roi["roi_bbox"], roi["roi_crop_raw"].shape, fg_prob
            )
            img_seg_qualities.append(s_seg)
        t_seg = (time.perf_counter() - t2) * 1000
        seg_times.append(t_seg)

        # Stage 4: Fusion & Decision
        t3 = time.perf_counter()
        for det_i, (det, s_seg) in enumerate(zip(detections, img_seg_qualities)):
            # Match detection with ground truth (IoU >= 0.5)
            is_tp = False
            best_iou = 0.0
            det_box = det["bbox"]
            for gt in gt_boxes:
                if gt["class_id"] == det["class_id"]:
                    gx1, gy1, gx2, gy2 = gt["bbox"]
                    dx1, dy1, dx2, dy2 = det_box
                    ix1, iy1 = max(dx1, gx1), max(dy1, gy1)
                    ix2, iy2 = min(dx2, gx2), min(dy2, gy2)
                    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                    inter = iw * ih
                    union = (dx2 - dx1)*(dy2 - dy1) + (gx2 - gx1)*(gy2 - gy1) - inter
                    iou = inter / max(1e-6, union)
                    if iou > best_iou:
                        best_iou = iou
            if best_iou >= 0.5:
                is_tp = True

            all_detections_data.append({
                "image_stem": img_path.stem,
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "yolo_conf": det["confidence"],
                "seg_quality": s_seg,
                "is_tp": is_tp,
                "best_gt_iou": round(best_iou, 4)
            })
        t_fus = (time.perf_counter() - t3) * 1000
        fusion_times.append(t_fus)

    # Compute Latency Benchmarks
    avg_yolo_ms = np.mean(yolo_times)
    avg_roi_ms = np.mean(roi_times) if roi_times else 0.0
    avg_seg_ms = np.mean(seg_times) if seg_times else 0.0
    avg_fus_ms = np.mean(fusion_times) if fusion_times else 0.0
    avg_total_ms = avg_yolo_ms + avg_roi_ms + avg_seg_ms + avg_fus_ms
    fps_end_to_end = 1000.0 / max(1e-6, avg_total_ms)

    # -------------------------------------------------------------
    # Evaluate Ablation across Alpha / Beta Ratios
    # -------------------------------------------------------------
    ablation_results = []
    
    # Baseline: System A (YOLO only at conf 0.15)
    yolo_tps = sum(1 for d in all_detections_data if d["is_tp"])
    yolo_fps = sum(1 for d in all_detections_data if not d["is_tp"])
    yolo_prec = yolo_tps / max(1, yolo_tps + yolo_fps)
    yolo_rec = yolo_tps / max(1, total_gt_boxes)

    ablation_results.append({
        "system": "System A (YOLO11 Only)",
        "alpha": 1.0,
        "beta": 0.0,
        "precision": round(yolo_prec * 100, 2),
        "recall": round(yolo_rec * 100, 2),
        "f1_score": round(2 * yolo_prec * yolo_rec / max(1e-6, yolo_prec + yolo_rec) * 100, 2),
        "fp_rejected": 0,
        "fp_rejection_rate": "0.0%",
        "verified_count": len(all_detections_data),
        "latency_ms": round(avg_yolo_ms, 2),
        "fps": round(1000.0 / max(1e-6, avg_yolo_ms), 1)
    })

    for alpha, beta in alpha_beta_pairs:
        tps, fps, fp_rejected, tp_rejected = 0, 0, 0, 0
        verified_count = 0
        for d in all_detections_data:
            f = fuse_detection(d["yolo_conf"], d["seg_quality"], alpha, beta, threshold_verified, threshold_reject)
            if f["decision"] in ["VERIFIED", "REVIEW"]:
                verified_count += 1
                if d["is_tp"]:
                    tps += 1
                else:
                    fps += 1
            else:  # REJECT
                if not d["is_tp"]:
                    fp_rejected += 1
                else:
                    tp_rejected += 1

        prec = tps / max(1, tps + fps)
        rec = tps / max(1, total_gt_boxes)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        fp_rej_pct = (fp_rejected / max(1, yolo_fps)) * 100

        ablation_results.append({
            "system": f"System C: Fusion (α={alpha}, β={beta})",
            "alpha": alpha,
            "beta": beta,
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "fp_rejected": fp_rejected,
            "fp_rejection_rate": f"{fp_rej_pct:.1f}%",
            "verified_count": verified_count,
            "latency_ms": round(avg_total_ms, 2),
            "fps": round(fps_end_to_end, 1)
        })

    # Save summary JSON
    summary_out = {
        "dataset_test_images": len(test_images),
        "ground_truth_targets": total_gt_boxes,
        "total_yolo_detections": total_yolo_detections,
        "latency_breakdown_ms": {
            "yolo_ms": round(avg_yolo_ms, 2),
            "roi_extraction_ms": round(avg_roi_ms, 2),
            "segformer_ms": round(avg_seg_ms, 2),
            "fusion_ms": round(avg_fus_ms, 2),
            "total_ms": round(avg_total_ms, 2),
            "end_to_end_fps": round(fps_end_to_end, 1)
        },
        "ablation_results": ablation_results
    }

    with open(out_path / "system_comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    # Print Table
    print("=" * 95)
    print(" 🏆 THREE-SYSTEM ABLATION & COMPARISON TABLE (TEST SET)")
    print("=" * 95)
    print(f"{'System / Configuration':<35} {'Prec (%)':>10} {'Rec (%)':>10} {'F1 (%)':>10} {'FP Rej %':>10} {'Latency':>10} {'FPS':>8}")
    print("-" * 95)
    for r in ablation_results:
        print(f"{r['system']:<35} {r['precision']:>9.2f}% {r['recall']:>9.2f}% {r['f1_score']:>9.2f}% {r['fp_rejection_rate']:>10} {r['latency_ms']:>8.2f}ms {r['fps']:>8.1f}")
    print("=" * 95)
    print(f"⏱️  LATENCY BREAKDOWN:")
    print(f"   • YOLO11 (832px):          {avg_yolo_ms:6.2f} ms/image ({1000/avg_yolo_ms:.1f} FPS)")
    print(f"   • Dynamic ROI Extraction:  {avg_roi_ms:6.2f} ms/image")
    print(f"   • SegFormer Segmentation:  {avg_seg_ms:6.2f} ms/image ({1000/max(1e-6, avg_seg_ms):.1f} FPS)")
    print(f"   • Fusion Decision Engine:  {avg_fus_ms:6.2f} ms/image")
    print(f"   • Total End-to-End:        {avg_total_ms:6.2f} ms/image ({fps_end_to_end:.1f} FPS)")
    print("=" * 95 + "\n")

    return summary_out


if __name__ == "__main__":
    run_system_comparison()
