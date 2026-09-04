# 📊 YOLOv11 Marine Debris & Acoustic Sonar Evaluation Report

**Generated on:** `2026-09-01 11:03:21`  
**Model Weights:** `yolo11s_anoma_best.pt`  
**Evaluation Dataset:** `C:\Users\CMRMuthuthiyagarajan\Downloads\SIHAKHETv2-main\SIHAKHETv2-main\samples\anoma\data.yaml` (`val` split)  

---

## 🚀 Executive Summary

| Category | Primary Metric | Value | Target Benchmark Status |
|---|---|---|---|
| **Detection Quality** | **mAP@50** | **34.6%** | ⚠️ Moderate Target |
| **Localization Precision** | **Mean IoU on Matched TPs** | **78.6%** | ✅ Strict Spatial Alignment |
| **Real-Time Edge Speed** | **Throughput (FPS)** | **84.13 FPS** (11.89 ms/img) | ✅ Real-Time Embedded Ready (>30 FPS) |
| **Model Footprint** | **Parameter Count / Size** | **9,429,340 Params** (18.29 MB) | ✅ Compact Deployment Footprint |

---

## 📈 Detection & Localization Performance Metrics

| Metric Name | Overall Value | Description / Interpretation |
|---|---|---|
| **Precision (P)** | **0.6719** (67.2%) | Proportion of detected debris that are true positive targets. |
| **Recall (R)** | **0.3621** (36.2%) | Proportion of all actual seabed debris captured by the model. |
| **F1-Score** | **0.4706** (47.1%) | Harmonic mean of Precision and Recall. |
| **mAP@50** | **0.3459** (34.6%) | Mean Average Precision at standard IoU threshold $0.50$. |
| **mAP@50-95** | **0.2124** (21.2%) | Stringent COCO primary benchmark averaged from IoU $0.50$ to $0.95$. |
| **Mean IoU (Matched)** | **0.7856** (78.6%) | Average bounding box spatial overlap across all true positives. |

---

## 📋 Per-Class Performance Table

| Class Name | AP@50 | AP@50-95 | Precision | Recall | F1-Score | Mean IoU |
|---|---|---|---|---|---|---|
| **Debris Target** | 61.9% | 44.1% | 70.6% | 64.7% | 67.5% | 80.6% |
| **Small Acoustic Fragment** | 2.5% | 0.8% | 21.7% | 2.6% | 4.7% | 63.3% |
| **Structural Cluster** | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| **Subsea Linear Structure** | 74.0% | 40.2% | 76.5% | 77.5% | 77.0% | 78.5% |


---

## ⚡ Deployment & Latency Benchmarks (Batch Size = 1)

* **Mean Inference Latency:** `11.89 ms`
* **Latency Standard Deviation:** `±1.88 ms`
* **95th Percentile Latency (P95):** `15.82 ms`
* **Min / Max Latency:** `9.34 ms / 19.02 ms`
* **Frames Per Second (FPS):** `84.13 FPS`
* **Hardware Device:** `NVIDIA GeForce RTX 4050 Laptop GPU` (6.0 GB VRAM)
* **Compute Framework:** `PyTorch 2.6.0+cu124 (CUDA 12.4)`

---

## 🔍 Training Diagnostics & Overfitting Analysis

* **Diagnostic Status:** `Analyzed successfully`
* **Epochs Trained:** `25`
* **Generalization Check:** 🟢 **Healthy Generalization** (Validation loss aligns with training loss)

---

## 🖼️ Saved Visualizations in `/eval_outputs`

1. `confusion_matrix.png` — Multi-class normalized confusion matrix.
2. `precision_recall_curve.png` — Per-class and aggregate PR curves.
3. `latency_distribution.png` — Inference time latency distribution histogram.
4. `training_curves.png` — Multi-panel training/validation loss & mAP curves across epochs.
