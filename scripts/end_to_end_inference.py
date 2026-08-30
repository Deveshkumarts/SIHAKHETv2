"""
End-to-End Inference Pipeline: YOLO11 → Dynamic ROI → SegFormer → Fusion.
Runs the complete sonar anomaly detection pipeline on a single image and produces
a multi-panel diagnostic visualization.
"""

import sys
import time
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.yolo_interface import YOLO11Detector
from utils.roi_utils import extract_rois_from_image, roi_mask_to_full_image
from scripts.fusion import compute_segmentation_quality_score, fuse_detection


YOLO_WEIGHTS = "outputs/experiments/exp_resolution_832/weights/best.pt"
SEG_WEIGHTS = "outputs/segformer/weights/best.pt"
CLASS_COLORS = {
    "shipwreck": (255, 165, 0),  # Orange
    "airplane":  (0, 128, 255),  # Blue
    "mine":      (0, 0, 255),    # Red
    "human":     (0, 255, 0),    # Green
}
DECISION_COLORS = {
    "VERIFIED": (0, 255, 0),     # Green
    "REVIEW":   (0, 165, 255),   # Orange
    "REJECT":   (0, 0, 255),     # Red
}


def run_pipeline(
    image_path: str,
    yolo_conf: float = 0.15,
    padding_ratio: float = 0.25,
    alpha: float = 0.6,
    beta: float = 0.4,
    threshold_verified: float = 0.55,
    threshold_reject: float = 0.35,
    output_dir: str = "outputs/fusion",
    seg_available: bool = True,
) -> dict:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    img_name = Path(image_path).stem
    timings = {}

    # ─── STAGE 1: YOLO11 DETECTION ─────────────────────────────────────────
    t0 = time.perf_counter()
    detector = YOLO11Detector(model_path=YOLO_WEIGHTS, conf_thresh=yolo_conf, imgsz=832)
    image_bgr, detections = detector.detect(image_path)
    timings["yolo_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    if not detections:
        print(f"⚠️  No YOLO detections found in: {image_path}")
        return {"detections": [], "timings": timings}

    print(f"\n🔍 YOLO11 found {len(detections)} detection(s) in {timings['yolo_ms']} ms")

    # ─── STAGE 2: DYNAMIC ROI EXTRACTION ───────────────────────────────────
    t1 = time.perf_counter()
    rois = extract_rois_from_image(image_bgr, detections, padding_ratio=padding_ratio, source_id=img_name)
    timings["roi_ms"] = round((time.perf_counter() - t1) * 1000, 2)
    print(f"✂️  ROI extraction: {timings['roi_ms']} ms")

    # ─── STAGE 3: SEGFORMER SEGMENTATION ───────────────────────────────────
    full_image_masks = []
    seg_qualities = []

    if seg_available and Path(SEG_WEIGHTS).exists():
        from segformer.inference import SegFormerInference
        t2 = time.perf_counter()
        segmenter = SegFormerInference(weights_path=SEG_WEIGHTS, img_size=224, device_str="0")
        seg_time = 0.0
        for roi in rois:
            ts = time.perf_counter()
            roi_mask_224, fg_prob = segmenter.predict(roi["roi_crop_raw"])
            seg_time += (time.perf_counter() - ts) * 1000

            # Project ROI mask back to full image
            full_mask = roi_mask_to_full_image(
                roi_mask_224.astype(np.float32) / 255.0,
                roi["roi_bbox"],
                image_bgr.shape
            )
            full_image_masks.append(full_mask)

            s_seg = compute_segmentation_quality_score(
                roi_mask_224, roi["roi_bbox"], roi["roi_crop_raw"].shape, fg_prob
            )
            seg_qualities.append(s_seg)

        timings["segformer_ms"] = round(seg_time, 2)
        print(f"🧠 SegFormer segmentation: {timings['segformer_ms']} ms ({len(rois)} ROIs)")
    else:
        full_image_masks = [None] * len(rois)
        seg_qualities = [0.0] * len(rois)
        timings["segformer_ms"] = 0.0

    # ─── STAGE 4: YOLO + SEGFORMER FUSION ──────────────────────────────────
    t3 = time.perf_counter()
    final_results = []
    for i, (det, roi) in enumerate(zip(detections, rois)):
        fusion = fuse_detection(
            yolo_conf=det["confidence"],
            seg_quality=seg_qualities[i],
            alpha=alpha, beta=beta,
            threshold_verified=threshold_verified,
            threshold_reject=threshold_reject,
        )
        final_results.append({
            "class_id": det["class_id"],
            "class_name": det["class_name"],
            "confidence": det["confidence"],
            "bbox": det["bbox"],
            "roi_bbox": roi["roi_bbox"],
            "seg_quality": seg_qualities[i],
            "fusion_score": fusion["fusion_score"],
            "decision": fusion["decision"],
        })
    timings["fusion_ms"] = round((time.perf_counter() - t3) * 1000, 2)
    timings["total_ms"] = round(sum(timings.values()), 2)
    timings["total_fps"] = round(1000.0 / max(1.0, timings["total_ms"]), 1)

    print(f"🔗 Fusion: {timings['fusion_ms']} ms | TOTAL: {timings['total_ms']} ms ({timings['total_fps']} FPS)")

    # ─── VISUALIZATION ──────────────────────────────────────────────────────
    vis = _build_visualization(image_bgr, final_results, full_image_masks)
    vis_path = output_path / f"{img_name}_pipeline_result.jpg"
    cv2.imwrite(str(vis_path), vis)
    print(f"\n🖼️  Visualization saved: {vis_path}")

    # Save JSON result
    json_path = output_path / f"{img_name}_result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"image": image_path, "detections": final_results, "timings": timings}, f, indent=2)

    _print_result_table(final_results, timings)
    return {"detections": final_results, "timings": timings}


def _build_visualization(image_bgr: np.ndarray, results: list, masks: list) -> np.ndarray:
    """Build multi-panel diagnostic visualization."""
    vis = image_bgr.copy()

    # Overlay segmentation masks first
    for result, full_mask in zip(results, masks):
        if full_mask is None:
            continue
        cname = result["class_name"]
        color = CLASS_COLORS.get(cname, (255, 255, 255))
        color_overlay = np.zeros_like(vis)
        color_overlay[full_mask > 0] = color
        vis = cv2.addWeighted(vis, 1.0, color_overlay, 0.35, 0)

    # Draw bounding boxes and labels
    for result in results:
        x1, y1, x2, y2 = [int(c) for c in result["bbox"]]
        cname = result["class_name"]
        decision = result["decision"]
        box_color = DECISION_COLORS.get(decision, (255, 255, 255))

        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

        label = (
            f"[{cname}] YOLO:{result['confidence']:.2f} "
            f"Seg:{result['seg_quality']:.2f} Fusion:{result['fusion_score']:.2f} {decision}"
        )
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(vis, label, (x1 + 2, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return vis


def _print_result_table(results: list, timings: dict):
    print("\n" + "=" * 90)
    print(" 🏆 PIPELINE RESULTS")
    print("=" * 90)
    print(f"{'Class':<12} {'YOLO Conf':>10} {'Seg Quality':>12} {'Fusion Score':>13} {'Decision':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['class_name']:<12} {r['confidence']:>10.3f} {r['seg_quality']:>12.3f} "
              f"{r['fusion_score']:>13.3f} {r['decision']:>10}")
    print("=" * 90)
    print(f"⏱️  Timings → YOLO: {timings['yolo_ms']}ms | ROI: {timings['roi_ms']}ms | "
          f"SegFormer: {timings['segformer_ms']}ms | Fusion: {timings['fusion_ms']}ms")
    print(f"📊 Total: {timings['total_ms']}ms ({timings['total_fps']} FPS)")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Sonar Detection Pipeline")
    parser.add_argument("image", type=str, help="Path to sonar image")
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO confidence threshold")
    parser.add_argument("--padding", type=float, default=0.25, help="ROI padding ratio")
    parser.add_argument("--alpha", type=float, default=0.6, help="YOLO weight in fusion")
    parser.add_argument("--beta", type=float, default=0.4, help="SegFormer weight in fusion")
    parser.add_argument("--output-dir", type=str, default="outputs/fusion", help="Output directory")
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        yolo_conf=args.conf,
        padding_ratio=args.padding,
        alpha=args.alpha,
        beta=args.beta,
        output_dir=args.output_dir,
    )
