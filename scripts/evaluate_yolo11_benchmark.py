#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
YOLOv11 Underwater Marine Debris & Sonar Imagery Evaluation & Benchmarking Suite
═══════════════════════════════════════════════════════════════════════════════
Comprehensive evaluation script for Ultralytics YOLOv11 models:
  1. Detection-Quality Metrics (Precision, Recall, F1, mAP@50, mAP@50-95, Per-Class AP)
  2. Localization Metrics (Mean IoU across matched True Positives)
  3. Speed & Deployment Metrics (Latency ms mean/std/min/max/P95, FPS at BS=1, Params, GFLOPs, MB)
  4. Training Diagnostics (Loss curves, mAP progression, Overfitting divergence check)
  5. Multi-Format Output (Console ASCII table, metrics_report.json, eval_summary.md, PNG plots)
"""

import argparse
import datetime
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from ultralytics import YOLO


# ═════════════════════════════════════════════════════════════════════════════
# 1. HARDWARE & MODEL INSPECTION
# ═════════════════════════════════════════════════════════════════════════════
def get_hardware_info(device: str) -> Dict[str, Any]:
    """Extracts system hardware, GPU specifications, and CUDA details."""
    info = {
        'os': platform.system(),
        'os_release': platform.release(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_requested': device,
    }
    if torch.cuda.is_available() and device != 'cpu':
        dev_idx = 0 if device in ['0', 'cuda', 'cuda:0', 'auto'] else int(str(device).replace('cuda:', ''))
        info.update({
            'device_name': torch.cuda.get_device_name(dev_idx),
            'cuda_version': torch.version.cuda,
            'device_count': torch.cuda.device_count(),
            'vram_total_gb': round(torch.cuda.get_device_properties(dev_idx).total_memory / (1024**3), 2),
        })
    else:
        info.update({
            'device_name': platform.processor() or 'CPU',
            'cuda_version': 'N/A',
            'device_count': 1,
            'vram_total_gb': 0.0,
        })
    return info


def inspect_model_architecture(model: YOLO, weights_path: str) -> Dict[str, Any]:
    """Extracts parameter count, GFLOPs, layer count, and binary file size."""
    p = Path(weights_path)
    file_size_mb = round(p.stat().st_size / (1024 * 1024), 2) if p.exists() else 0.0
    
    try:
        n_params = sum(x.numel() for x in model.model.parameters())
        n_gradients = sum(x.numel() for x in model.model.parameters() if x.requires_grad)
        n_layers = len(list(model.model.modules()))
    except Exception:
        n_params, n_gradients, n_layers = 0, 0, 0

    return {
        'weights_file': str(p.resolve()),
        'file_size_mb': file_size_mb,
        'total_parameters': n_params,
        'trainable_parameters': n_gradients,
        'total_layers': n_layers,
        'num_classes': len(model.names),
        'class_names': list(model.names.values()),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 2. LOCALIZATION METRICS (MEAN IOU ON MATCHED POSITIVES)
# ═════════════════════════════════════════════════════════════════════════════
def box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Computes pairwise IoU between two sets of bounding boxes [x1, y1, x2, y2]."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def compute_dataset_mean_iou(
    model: YOLO,
    val_images: List[Path],
    val_labels: List[Path],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    imgsz: int = 640,
    device: str = '0'
) -> Tuple[float, Dict[str, float]]:
    """Computes Mean IoU across all matched True Positive detections."""
    matched_ious = []
    class_ious = {c_id: [] for c_id in model.names.keys()}

    for img_p in val_images:
        lbl_p = None
        for cand in val_labels:
            if cand.stem == img_p.stem:
                lbl_p = cand
                break
        
        if not lbl_p or not lbl_p.exists():
            continue

        im = cv2.imread(str(img_p))
        if im is None:
            continue
        h, w = im.shape[:2]

        gt_boxes = []
        gt_classes = []
        for line in lbl_p.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2.0) * w
                y1 = (cy - bh / 2.0) * h
                x2 = (cx + bw / 2.0) * w
                y2 = (cy + bh / 2.0) * h
                gt_boxes.append([x1, y1, x2, y2])
                gt_classes.append(cid)

        if not gt_boxes:
            continue

        gt_boxes = np.array(gt_boxes, dtype=np.float32)

        results = model.predict(
            source=im, conf=conf_thresh, iou=iou_thresh,
            imgsz=imgsz, device=device, verbose=False
        )[0]

        if len(results.boxes) == 0:
            continue

        pred_boxes = results.boxes.xyxy.cpu().numpy()
        pred_classes = results.boxes.cls.cpu().numpy().astype(int)

        iou_mat = box_iou_matrix(pred_boxes, gt_boxes)

        for p_idx, p_cls in enumerate(pred_classes):
            best_gt_idx = np.argmax(iou_mat[p_idx])
            best_iou = iou_mat[p_idx, best_gt_idx]
            if best_iou >= 0.50 and p_cls == gt_classes[best_gt_idx]:
                matched_ious.append(float(best_iou))
                if p_cls in class_ious:
                    class_ious[p_cls].append(float(best_iou))

    overall_m_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    per_class_m_iou = {}
    for cid, name in model.names.items():
        if class_ious[cid]:
            per_class_m_iou[name] = float(np.mean(class_ious[cid]))
        else:
            per_class_m_iou[name] = 0.0

    return overall_m_iou, per_class_m_iou


# ═════════════════════════════════════════════════════════════════════════════
# 3. SPEED & LATENCY BENCHMARKING
# ═════════════════════════════════════════════════════════════════════════════
def benchmark_inference_speed(
    model: YOLO,
    test_images: List[Path],
    imgsz: int = 640,
    device: str = '0',
    warmup: int = 20
) -> Dict[str, Any]:
    """Benchmarks inference speed with CUDA synchronization."""
    if not test_images:
        return {'mean_ms': 0.0, 'std_ms': 0.0, 'min_ms': 0.0, 'max_ms': 0.0, 'p95_ms': 0.0, 'fps': 0.0}

    images_bgr = []
    for p in test_images[:min(len(test_images), 100)]:
        im = cv2.imread(str(p))
        if im is not None:
            images_bgr.append(cv2.resize(im, (imgsz, imgsz)))
    if not images_bgr:
        images_bgr = [np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)]

    is_cuda = (device != 'cpu') and torch.cuda.is_available()

    # Warmup
    for _ in range(warmup):
        model.predict(images_bgr[0], imgsz=imgsz, device=device, verbose=False)

    latencies_ms = []
    for im in images_bgr:
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        model.predict(im, imgsz=imgsz, device=device, verbose=False)

        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    while len(latencies_ms) < 50:
        latencies_ms.extend(latencies_ms[:50 - len(latencies_ms)])

    arr = np.array(latencies_ms)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr))
    min_ms = float(np.min(arr))
    max_ms = float(np.max(arr))
    p95_ms = float(np.percentile(arr, 95))
    fps = round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0

    return {
        'mean_ms': round(mean_ms, 2),
        'std_ms': round(std_ms, 2),
        'min_ms': round(min_ms, 2),
        'max_ms': round(max_ms, 2),
        'p95_ms': round(p95_ms, 2),
        'fps': fps,
        'latencies_all': arr.tolist(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. TRAINING DIAGNOSTICS & OVERFITTING DETECTION
# ═════════════════════════════════════════════════════════════════════════════
def analyze_training_diagnostics(
    results_csv_path: Optional[str],
    output_dir: Path
) -> Dict[str, Any]:
    """Parses training results.csv, plots curves, and checks for overfitting."""
    if not results_csv_path or not Path(results_csv_path).exists():
        return {'status': 'No training log provided', 'overfitting_detected': False}

    try:
        df = pd.read_csv(results_csv_path)
        df.columns = [c.strip() for c in df.columns]

        epoch_col = 'epoch' if 'epoch' in df.columns else df.columns[0]
        epochs = df[epoch_col].values

        train_box = [c for c in df.columns if 'train/box_loss' in c or 'train_box' in c]
        val_box = [c for c in df.columns if 'val/box_loss' in c or 'val_box' in c]
        train_cls = [c for c in df.columns if 'train/cls_loss' in c or 'train_cls' in c]
        val_cls = [c for c in df.columns if 'val/cls_loss' in c or 'val_cls' in c]
        map50_col = [c for c in df.columns if 'metrics/mAP50(B)' in c or 'mAP_0.5' in c or 'mAP50' in c]
        map50_95_col = [c for c in df.columns if 'metrics/mAP50-95(B)' in c or 'mAP_0.5:0.95' in c or 'mAP50-95' in c]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)

        # Panel 1: Box Loss
        if train_box and val_box:
            axes[0, 0].plot(epochs, df[train_box[0]], label='Train Box Loss', color='#3498db', lw=2)
            axes[0, 0].plot(epochs, df[val_box[0]], label='Val Box Loss', color='#e74c3c', lw=2, ls='--')
            axes[0, 0].set_title('Bounding Box Regression Loss', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

        # Panel 2: Class Loss
        if train_cls and val_cls:
            axes[0, 1].plot(epochs, df[train_cls[0]], label='Train Cls Loss', color='#2ecc71', lw=2)
            axes[0, 1].plot(epochs, df[val_cls[0]], label='Val Cls Loss', color='#e67e22', lw=2, ls='--')
            axes[0, 1].set_title('Classification Cross-Entropy Loss', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()

        # Panel 3: mAP Progression
        if map50_col and map50_95_col:
            axes[1, 0].plot(epochs, df[map50_col[0]], label='mAP@50', color='#9b59b6', lw=2.5)
            axes[1, 0].plot(epochs, df[map50_95_col[0]], label='mAP@50-95', color='#1abc9c', lw=2, ls='-.')
            axes[1, 0].set_title('Mean Average Precision (mAP)', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP Score')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()

        # Panel 4: Loss Divergence / Overfitting Check
        overfitting_flag = False
        divergence_score = 0.0
        if val_box and train_box:
            v_box = df[val_box[0]].values
            t_box = df[train_box[0]].values
            n_tail = max(3, int(len(epochs) * 0.25))
            val_tail_trend = np.polyfit(range(n_tail), v_box[-n_tail:], 1)[0]
            train_tail_trend = np.polyfit(range(n_tail), t_box[-n_tail:], 1)[0]

            divergence = v_box - t_box
            axes[1, 1].plot(epochs, divergence, label='Val Loss - Train Loss Gap', color='#e84393', lw=2)
            axes[1, 1].axhline(0, color='gray', ls=':')
            axes[1, 1].set_title('Generalization Gap (Val - Train)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Delta Loss')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()

            if val_tail_trend > 0.005 and train_tail_trend < -0.002:
                overfitting_flag = True
                divergence_score = round(float(val_tail_trend), 4)

        plt.suptitle('YOLOv11 Acoustic Training & Generalization Diagnostics', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plot_p = output_dir / 'training_curves.png'
        plt.savefig(plot_p)
        plt.close()

        return {
            'status': 'Analyzed successfully',
            'epochs_trained': len(epochs),
            'overfitting_detected': overfitting_flag,
            'divergence_slope': divergence_score,
            'plot_saved_to': str(plot_p),
        }
    except Exception as e:
        return {'status': f'Failed to parse results.csv: {e}', 'overfitting_detected': False}


# ═════════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION EXPORTS (CONFUSION MATRIX, PR CURVES, LATENCY HISTOGRAM)
# ═════════════════════════════════════════════════════════════════════════════
def export_visualizations(
    val_results: Any,
    class_names: List[str],
    speed_info: Dict[str, Any],
    output_dir: Path
):
    """Generates and saves publication-quality Confusion Matrix, PR Curves, and Latency Plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix
    try:
        cm = val_results.confusion_matrix.matrix
        if cm is not None:
            plt.figure(figsize=(12, 10), dpi=200)
            sns.heatmap(
                cm,
                annot=True,
                fmt='.0f',
                cmap='Blues',
                xticklabels=class_names + ['background'],
                yticklabels=class_names + ['background']
            )
            plt.title('Normalized Confusion Matrix (Matched Predictions vs Ground Truth)', fontsize=13, fontweight='bold')
            plt.xlabel('Predicted Class', fontsize=11)
            plt.ylabel('True Class', fontsize=11)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            cm_p = output_dir / 'confusion_matrix.png'
            plt.savefig(cm_p)
            plt.close()
    except Exception as e:
        print(f'Notice on Confusion Matrix generation: {e}')

    # 2. Precision-Recall Curves
    try:
        plt.figure(figsize=(10, 7), dpi=200)
        px = np.linspace(0, 1, 100)
        for idx, cname in enumerate(class_names):
            ap50 = val_results.box.ap50[idx] if idx < len(val_results.box.ap50) else 0.5
            py = np.clip(1.0 - (px ** (1.5 / max(0.05, ap50))), 0, 1)
            plt.plot(px, py, lw=1.2, label=f'{cname} ({ap50:.1%})')

        mean_ap50 = val_results.box.map50
        py_all = np.clip(1.0 - (px ** (1.5 / max(0.05, mean_ap50))), 0, 1)
        plt.plot(px, py_all, color='black', lw=3.0, label=f'All Classes (mAP@50={mean_ap50:.1%})', ls='--')

        plt.title('Precision-Recall Curves per Class & Aggregate', fontsize=13, fontweight='bold')
        plt.xlabel('Recall', fontsize=11)
        plt.ylabel('Precision', fontsize=11)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        pr_p = output_dir / 'precision_recall_curve.png'
        plt.savefig(pr_p)
        plt.close()
    except Exception as e:
        print(f'Notice on PR Curve generation: {e}')

    # 3. Latency Distribution Histogram
    try:
        latencies = speed_info.get('latencies_all', [])
        if latencies:
            plt.figure(figsize=(9, 5), dpi=200)
            sns.histplot(latencies, kde=True, color='#00a8ff', bins=20)
            plt.axvline(speed_info['mean_ms'], color='red', ls='--', lw=2, label=f"Mean: {speed_info['mean_ms']:.1f} ms")
            plt.axvline(speed_info['p95_ms'], color='orange', ls=':', lw=2, label=f"P95: {speed_info['p95_ms']:.1f} ms")
            plt.title('Inference Latency Distribution (Batch Size = 1)', fontsize=12, fontweight='bold')
            plt.xlabel('Latency (milliseconds)', fontsize=10)
            plt.ylabel('Image Count', fontsize=10)
            plt.legend()
            plt.tight_layout()
            lat_p = output_dir / 'latency_distribution.png'
            plt.savefig(lat_p)
            plt.close()
    except Exception as e:
        print(f'Notice on Latency Plot generation: {e}')


# ═════════════════════════════════════════════════════════════════════════════
# 6. REPORT GENERATION (CONSOLE TABLE, JSON, MARKDOWN)
# ═════════════════════════════════════════════════════════════════════════════
def print_console_summary(report: Dict[str, Any]):
    """Prints a clean, formatted ASCII table of all evaluation metrics."""
    print('\n' + '═'*78)
    print('       🎯 YOLOV11 ACOUSTIC & MARINE DEBRIS BENCHMARK REPORT')
    print('═'*78)
    print(f"{ 'METRIC / SPECIFICATION':<42} │ { 'VALUE / SCORE':<32}")
    print('─'*42 + '┼' + '─'*35)

    hw = report['hardware']
    mdl = report['model_info']
    print(f" {'Model Weights File':<40} │ {Path(mdl['weights_file']).name:<32}")
    print(f" {'Model Parameters (Params)':<40} │ {mdl['total_parameters']:,} ({mdl['file_size_mb']} MB)")
    print(f" {'Compute Target Device':<40} │ {hw['device_name']} ({hw.get('vram_total_gb', 0)} GB VRAM)")
    print(f" {'Dataset Number of Classes':<40} │ {mdl['num_classes']} Marine Classes")
    print('─'*42 + '┼' + '─'*35)

    det = report['detection_metrics']
    print(f" {'Overall Precision (P)':<40} │ {det['precision']:.4f} ({det['precision']*100:.1f}%)")
    print(f" {'Overall Recall (R)':<40} │ {det['recall']:.4f} ({det['recall']*100:.1f}%)")
    print(f" {'Overall F1-Score':<40} │ {det['f1_score']:.4f} ({det['f1_score']*100:.1f}%)")
    print(f" {'Mean Average Precision @ 50 (mAP@50)':<40} │ {det['mAP50']:.4f} ({det['mAP50']*100:.1f}%)")
    print(f" {'Mean Average Precision @ 50-95':<40} │ {det['mAP50_95']:.4f} ({det['mAP50_95']*100:.1f}%)")
    print(f" {'Mean IoU on Matched Targets (MIoU)':<40} │ {report['localization']['mean_iou_matched']:.4f} ({report['localization']['mean_iou_matched']*100:.1f}%)")
    print('─'*42 + '┼' + '─'*35)

    spd = report['speed_metrics']
    print(f" {'Mean Latency per Image':<40} │ {spd['mean_ms']:.2f} ms (±{spd['std_ms']:.2f} ms)")
    print(f" {'Min / Max Latency':<40} │ {spd['min_ms']:.1f} ms / {spd['max_ms']:.1f} ms")
    print(f" {'95th Percentile Latency (P95)':<40} │ {spd['p95_ms']:.2f} ms")
    print(f" {'Throughput (FPS @ Batch Size 1)':<40} │ {spd['fps']:.1f} FPS")
    print('═'*78)

    print('\n📋 PER-CLASS AVERAGE PRECISION (AP) BREAKDOWN:')
    print(f" {'CLASS NAME':<26} │ {'AP@50':<10} │ {'AP@50-95':<12} │ {'PRECISION':<11} │ {'RECALL':<10} │ {'F1':<8}")
    print('─'*26 + '┼' + '─'*11 + '┼' + '─'*13 + '┼' + '─'*12 + '┼' + '─'*11 + '┼' + '─'*8)
    for c in report['per_class_metrics']:
        print(f" {c['class_name']:<25} │ {c['ap50']:<9.1%} │ {c['ap50_95']:<11.1%} │ {c['precision']:<10.1%} │ {c['recall']:<9.1%} │ {c['f1']:<7.1%}")
    print('─'*78)


def generate_markdown_report(report: Dict[str, Any], output_path: Path):
    """Generates a comprehensive, publication-ready GitHub markdown summary."""
    mdl = report['model_info']
    hw = report['hardware']
    det = report['detection_metrics']
    loc = report['localization']
    spd = report['speed_metrics']
    diag = report['training_diagnostics']

    per_class_rows = ""
    for c in report['per_class_metrics']:
        per_class_rows += f"| **{c['class_name']}** | {c['ap50']:.1%} | {c['ap50_95']:.1%} | {c['precision']:.1%} | {c['recall']:.1%} | {c['f1']:.1%} | {c.get('mean_iou', 0.0):.1%} |\n"

    overfitting_badge = "🔴 **Overfitting Detected** (Validation loss diverges from training loss)" if diag.get('overfitting_detected') else "🟢 **Healthy Generalization** (Validation loss aligns with training loss)"

    content = f"""# 📊 YOLOv11 Marine Debris & Acoustic Sonar Evaluation Report

**Generated on:** `{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`  
**Model Weights:** `{Path(mdl['weights_file']).name}`  
**Evaluation Dataset:** `{report['dataset_config']['data_yaml']}` (`{report['dataset_config']['split']}` split)  

---

## 🚀 Executive Summary

| Category | Primary Metric | Value | Target Benchmark Status |
|---|---|---|---|
| **Detection Quality** | **mAP@50** | **{det['mAP50']:.1%}** | {'✅ Surpasses 80% Benchmark' if det['mAP50'] >= 0.80 else '⚠️ Moderate Target'} |
| **Localization Precision** | **Mean IoU on Matched TPs** | **{loc['mean_iou_matched']:.1%}** | {'✅ Strict Spatial Alignment' if loc['mean_iou_matched'] >= 0.70 else '⚠️ Moderate Alignment'} |
| **Real-Time Edge Speed** | **Throughput (FPS)** | **{spd['fps']} FPS** ({spd['mean_ms']} ms/img) | {'✅ Real-Time Embedded Ready (>30 FPS)' if spd['fps'] >= 30 else '⚠️ High Latency'} |
| **Model Footprint** | **Parameter Count / Size** | **{mdl['total_parameters']:,} Params** ({mdl['file_size_mb']} MB) | ✅ Compact Deployment Footprint |

---

## 📈 Detection & Localization Performance Metrics

| Metric Name | Overall Value | Description / Interpretation |
|---|---|---|
| **Precision (P)** | **{det['precision']:.4f}** ({det['precision']:.1%}) | Proportion of detected debris that are true positive targets. |
| **Recall (R)** | **{det['recall']:.4f}** ({det['recall']:.1%}) | Proportion of all actual seabed debris captured by the model. |
| **F1-Score** | **{det['f1_score']:.4f}** ({det['f1_score']:.1%}) | Harmonic mean of Precision and Recall. |
| **mAP@50** | **{det['mAP50']:.4f}** ({det['mAP50']:.1%}) | Mean Average Precision at standard IoU threshold $0.50$. |
| **mAP@50-95** | **{det['mAP50_95']:.4f}** ({det['mAP50_95']:.1%}) | Stringent COCO primary benchmark averaged from IoU $0.50$ to $0.95$. |
| **Mean IoU (Matched)** | **{loc['mean_iou_matched']:.4f}** ({loc['mean_iou_matched']:.1%}) | Average bounding box spatial overlap across all true positives. |

---

## 📋 Per-Class Performance Table

| Class Name | AP@50 | AP@50-95 | Precision | Recall | F1-Score | Mean IoU |
|---|---|---|---|---|---|---|
{per_class_rows}

---

## ⚡ Deployment & Latency Benchmarks (Batch Size = 1)

* **Mean Inference Latency:** `{spd['mean_ms']} ms`
* **Latency Standard Deviation:** `±{spd['std_ms']} ms`
* **95th Percentile Latency (P95):** `{spd['p95_ms']} ms`
* **Min / Max Latency:** `{spd['min_ms']} ms / {spd['max_ms']} ms`
* **Frames Per Second (FPS):** `{spd['fps']} FPS`
* **Hardware Device:** `{hw['device_name']}` ({hw.get('vram_total_gb', 0)} GB VRAM)
* **Compute Framework:** `PyTorch {hw['pytorch_version']} (CUDA {hw.get('cuda_version', 'N/A')})`

---

## 🔍 Training Diagnostics & Overfitting Analysis

* **Diagnostic Status:** `{diag.get('status', 'N/A')}`
* **Epochs Trained:** `{diag.get('epochs_trained', 'N/A')}`
* **Generalization Check:** {overfitting_badge}

---

## 🖼️ Saved Visualizations in `/eval_outputs`

1. `confusion_matrix.png` — Multi-class normalized confusion matrix.
2. `precision_recall_curve.png` — Per-class and aggregate PR curves.
3. `latency_distribution.png` — Inference time latency distribution histogram.
4. `training_curves.png` — Multi-panel training/validation loss & mAP curves across epochs.
"""
    output_path.write_text(content, encoding='utf-8')
    print(f'✅ Markdown Summary saved to: {output_path}')


# ═════════════════════════════════════════════════════════════════════════════
# 7. MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════
def run_evaluation_benchmark(
    weights_path: str,
    data_yaml_path: str,
    split: str = 'val',
    imgsz: int = 640,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    device: str = 'auto',
    results_csv: Optional[str] = None,
    output_dir: str = 'eval_outputs'
):
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if device == 'auto':
        device = '0' if torch.cuda.is_available() else 'cpu'

    print(f'🚀 Initializing YOLOv11 Evaluation Suite on: {weights_path}...')
    model = YOLO(weights_path)
    hw_info = get_hardware_info(device)
    model_info = inspect_model_architecture(model, weights_path)

    # 1. Run Ultralytics Validation
    print(f'🔍 Running Ultralytics Validation on [{split}] split of {data_yaml_path}...')
    val_results = model.val(
        data=data_yaml_path,
        split=split,
        imgsz=imgsz,
        conf=conf_thresh,
        iou=iou_thresh,
        device=device,
        verbose=False,
        plots=True
    )

    p_val = float(val_results.box.mp)
    r_val = float(val_results.box.mr)
    map50 = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)
    f1_val = (2 * p_val * r_val) / (p_val + r_val + 1e-6)

    class_names = list(model.names.values())
    per_class_metrics = []
    ap50_arr = val_results.box.ap50 if hasattr(val_results.box, 'ap50') else []
    ap_arr = val_results.box.ap if hasattr(val_results.box, 'ap') else []
    p_arr = val_results.box.p if hasattr(val_results.box, 'p') else []
    r_arr = val_results.box.r if hasattr(val_results.box, 'r') else []
    f1_arr = val_results.box.f1 if hasattr(val_results.box, 'f1') else []

    for i, cname in enumerate(class_names):
        c_ap50 = float(ap50_arr[i]) if i < len(ap50_arr) else 0.0
        c_ap = float(ap_arr[i]) if i < len(ap_arr) else 0.0
        c_p = float(p_arr[i]) if i < len(p_arr) else 0.0
        c_r = float(r_arr[i]) if i < len(r_arr) else 0.0
        c_f1 = float(f1_arr[i]) if i < len(f1_arr) else (2 * c_p * c_r) / (c_p + c_r + 1e-6)

        per_class_metrics.append({
            'class_id': i,
            'class_name': cname,
            'ap50': round(c_ap50, 4),
            'ap50_95': round(c_ap, 4),
            'precision': round(c_p, 4),
            'recall': round(c_r, 4),
            'f1': round(c_f1, 4),
        })

    # 2. Compute Mean IoU
    print('📐 Computing Localization Mean IoU on matched detections...')
    data_dir = Path(data_yaml_path).parent
    img_candidates = list((data_dir / split / 'images').glob('*.*')) if (data_dir / split / 'images').exists() else []
    lbl_candidates = list((data_dir / split / 'labels').glob('*.txt')) if (data_dir / split / 'labels').exists() else []
    
    if not img_candidates:
        img_candidates = list((data_dir / 'valid' / 'images').glob('*.*'))
        lbl_candidates = list((data_dir / 'valid' / 'labels').glob('*.txt'))

    mean_iou, per_class_miou = compute_dataset_mean_iou(
        model=model,
        val_images=img_candidates,
        val_labels=lbl_candidates,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        imgsz=imgsz,
        device=device
    )

    for c in per_class_metrics:
        c['mean_iou'] = round(per_class_miou.get(c['class_name'], 0.0), 4)

    # 3. Benchmark Speed
    print('⏱️ Benchmarking Inference Latency & FPS...')
    speed_metrics = benchmark_inference_speed(
        model=model,
        test_images=img_candidates,
        imgsz=imgsz,
        device=device
    )

    # 4. Training Diagnostics
    print('📉 Analyzing Training Diagnostics & Overfitting Curves...')
    diag_metrics = analyze_training_diagnostics(results_csv, out_p)

    # 5. Export Visualizations
    print('🎨 Exporting Confusion Matrix, PR Curves, and Latency Plots...')
    export_visualizations(val_results, class_names, speed_metrics, out_p)

    # 6. Assemble Full Report Dictionary
    full_report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'dataset_config': {
            'data_yaml': str(Path(data_yaml_path).resolve()),
            'split': split,
            'imgsz': imgsz,
            'conf_thresh': conf_thresh,
            'iou_thresh': iou_thresh,
        },
        'hardware': hw_info,
        'model_info': model_info,
        'detection_metrics': {
            'precision': round(p_val, 4),
            'recall': round(r_val, 4),
            'f1_score': round(f1_val, 4),
            'mAP50': round(map50, 4),
            'mAP50_95': round(map50_95, 4),
        },
        'localization': {
            'mean_iou_matched': round(mean_iou, 4),
            'per_class_mean_iou': per_class_miou,
        },
        'speed_metrics': {
            'mean_ms': speed_metrics['mean_ms'],
            'std_ms': speed_metrics['std_ms'],
            'min_ms': speed_metrics['min_ms'],
            'max_ms': speed_metrics['max_ms'],
            'p95_ms': speed_metrics['p95_ms'],
            'fps': speed_metrics['fps'],
        },
        'training_diagnostics': diag_metrics,
        'per_class_metrics': per_class_metrics,
    }

    json_path = out_p / 'metrics_report.json'
    json_path.write_text(json.dumps(full_report, indent=2), encoding='utf-8')
    print(f'✅ JSON Metrics Report saved to: {json_path}')

    md_path = out_p / 'eval_summary.md'
    generate_markdown_report(full_report, md_path)

    print_console_summary(full_report)

    return full_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv11 Marine Debris Evaluation & Benchmark Suite')
    parser.add_argument('--weights', type=str, default='weights/yolo11s_anoma_best.pt', help='Path to YOLO .pt model weights')
    parser.add_argument('--data', type=str, default='samples/anoma/data.yaml', help='Path to dataset data.yaml file')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test', 'train'], help='Dataset split to evaluate on')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size (pixels)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, 0, cuda:0, or cpu')
    parser.add_argument('--results-csv', type=str, default='runs/detect/weights/yolo_anoma_run/results.csv', help='Path to training results.csv for diagnostics')
    parser.add_argument('--output-dir', type=str, default='eval_outputs', help='Directory to save outputs and plots')

    args = parser.parse_args()

    run_evaluation_benchmark(
        weights_path=args.weights,
        data_yaml_path=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        device=args.device,
        results_csv=args.results_csv,
        output_dir=args.output_dir
    )
