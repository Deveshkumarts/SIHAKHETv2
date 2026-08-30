# 🌊 YOLO11 Sea & Marine Debris Detection System

A production-grade, modular computer vision pipeline built with **YOLO11** and **PyTorch** designed to detect marine debris (plastics, fishing gear, cans, glass, and general litter) in complex ocean-surface and underwater environments.

The architecture is specifically engineered to handle harsh optical challenges:
* **Severe Light Attenuation & Color Cast**: Wavelength-dependent blue/green water absorption.
* **Water Turbidity & Backscatter**: Suspended silt, bubbles, and organic matter scattering light.
* **Small & Deformed Objects**: Distant micro-plastics, torn bags, tangled fishing nets, and partially buried debris on the seabed.
* **Background Confusion**: Distinguishing debris from coral reefs, rocks, algae, fish, and sun glint reflections.

---

## 📁 Project Architecture

```text
sea_debris_yolo/
│
├── dataset/                         # YOLO-formatted dataset
│   ├── images/                      # Raw image frames
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/                      # Normalized bounding box labels (.txt)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml                    # Dataset class mapping & split paths
│
├── configs/
│   └── train_config.yaml            # Hyperparameters, marine augmentations & device settings
│
├── scripts/
│   ├── check_dataset.py             # Dataset health audit, corruption check & geometry validation
│   ├── visualize_dataset.py         # Visual ground-truth bbox & label inspector
│   ├── train.py                     # YOLO11 fine-tuning engine & experiment metadata recorder
│   ├── evaluate.py                  # Model evaluation (P, R, mAP50, mAP50-95, per-class metrics)
│   ├── analyze_errors.py            # Diagnostic tool (False Positives, False Negatives, Low-Conf)
│   ├── predict.py                   # Image batch & single inference engine
│   ├── predict_video.py             # Memory-safe video streaming & live webcam processor
│   └── export_model.py              # Export weights to ONNX, TensorRT, TorchScript
│
├── utils/
│   ├── __init__.py
│   ├── dataset_utils.py             # Bounding box geometry, hash deduplication & label parser
│   ├── device_utils.py              # Auto CUDA/GPU detection, CPU fallback & VRAM safety
│   └── visualization.py             # Marine-contrast bounding boxes & HUD graphics
│
├── sample_data/
│   └── generate_sample_data.py      # Synthetic underwater starter dataset generator
│
├── outputs/                         # Structured run outputs & artifacts
│   ├── training/                    # Model checkpoints & training curves
│   ├── evaluation/                  # Precision-Recall charts & JSON evaluation reports
│   ├── predictions/                 # Annotated images & video streams
│   ├── error_analysis/              # Categorized FP/FN diagnostic visual galleries
│   └── exported/                    # Production-ready ONNX / TensorRT models
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ 1. Installation & Environment Setup

### Prerequisites
* Python **3.10+** (Tested on Python 3.12)
* NVIDIA GPU with CUDA 11.8+ / 12.0+ (Optional, automatic CPU fallback included)

### Step 1: Create Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🗂️ 2. Dataset Setup & Configuration

### Directory Format
Store your images and corresponding YOLO `.txt` label files in standard YOLO structure:
```text
dataset/
├── images/
│   ├── train/  -> sea_001.jpg, sea_002.png ...
│   ├── val/    -> sea_100.jpg ...
│   └── test/   -> sea_200.jpg ...
└── labels/
    ├── train/  -> sea_001.txt, sea_002.txt ...
    ├── val/    -> sea_100.txt ...
    └── test/   -> sea_200.txt ...
```

### Label Annotation Format
Each `.txt` file contains one row per object:
```text
<class_id> <center_x> <center_y> <width> <height>
```
*All coordinates are normalized to $[0.0, 1.0]$.*

### Class Configuration (`dataset/data.yaml`)
You can switch between single-class or multi-class debris detection simply by modifying `dataset/data.yaml` without changing any Python code:

**Multi-Class (Default):**
```yaml
path: .
train: images/train
val: images/val
test: images/test

nc: 6
names:
  0: plastic_bottle
  1: plastic_bag
  2: fishing_net
  3: can
  4: glass
  5: other_debris
```

**Single-Class Alternative:**
```yaml
nc: 1
names:
  0: debris
```

---

## 🧪 3. Synthetic Starter Dataset (Quick Smoke Test)

If you do not have your dataset ready yet, generate a starter simulated underwater dataset:
```bash
python sample_data/generate_sample_data.py
```

---

## 🔍 4. Dataset Validation & Quality Audit

Before training, run `check_dataset.py` to catch corrupted images, missing labels, out-of-bounds coordinates, zero-area boxes, and duplicate files:
```bash
python scripts/check_dataset.py --data dataset/data.yaml
```

**What it checks:**
* Corrupt or unreadable image headers.
* Missing `.txt` label files for images.
* Class IDs exceeding defined classes in `data.yaml`.
* Coordinates outside $[0, 1]$ or negative widths/heights.
* Duplicate image detection via MD5 hashing.
* Background (negative) image ratio.
* Per-class instance distribution.

---

## 🖼️ 5. Dataset Annotation Inspector

Inspect ground-truth bounding boxes visually to detect mislabeled or shifted annotations:
```bash
python scripts/visualize_dataset.py --split train --num 10 --save-dir outputs/visualizations
```

---

## 🚀 6. Model Training & Fine-Tuning

Fine-tune a pretrained YOLO11 model (`yolo11s.pt`, `yolo11n.pt`, or `yolo11m.pt`) with marine-tailored augmentations:

### Basic Training (Using YAML Configuration)
```bash
python scripts/train.py --config configs/train_config.yaml
```

### Custom Parameter Overrides
```bash
python scripts/train.py \
    --model yolo11s.pt \
    --imgsz 640 \
    --batch 16 \
    --epochs 100 \
    --device auto \
    --name yolo11s_marine_baseline
```

### Marine Augmentation Strategy (`configs/train_config.yaml`)
* **HSV-Hue & Saturation (`hsv_h: 0.015`, `hsv_s: 0.5`)**: Simulates varying water tints (green coastal vs. deep blue open ocean) and turbidity color washing.
* **HSV-Value (`hsv_v: 0.4`)**: Simulates light attenuation across varying water depths.
* **Mosaic (`mosaic: 1.0`)**: Combines 4 images into one, forcing the model to learn small debris detection.
* **Random Erasing (`erasing: 0.2`)**: Simulates occlusion by seaweed, fish, bubbles, and silt clouds.

---

## 📊 7. Model Evaluation & Benchmarking

Evaluate your trained checkpoint on the validation or test split:
```bash
python scripts/evaluate.py \
    --model outputs/training/baseline_yolo11s/weights/best.pt \
    --data dataset/data.yaml \
    --split val \
    --conf 0.25 \
    --iou 0.6
```

**Metrics Reported:**
* Overall **Precision (P)**, **Recall (R)**, **mAP@50**, and **mAP@50-95**.
* Per-class precision/recall/mAP table.
* Inference latency (ms per frame) and real-time FPS throughput.
* Saves `evaluation_summary.json` and `metrics_summary_chart.png` in `outputs/evaluation/`.

---

## 🔬 8. Error Analysis & Diagnostics

Inspect detector failure modes to diagnose why the model makes mistakes:
```bash
python scripts/analyze_errors.py \
    --model outputs/training/baseline_yolo11s/weights/best.pt \
    --data dataset/data.yaml \
    --split val \
    --conf 0.35 \
    --iou 0.5
```

**Generates Categorized Visual Galleries in `outputs/error_analysis/`:**
1. **False Positives (`false_positives/`)**: Background rocks, seaweeds, or reflections misclassified as debris.
2. **False Negatives (`false_negatives/`)**: Real debris objects missed by the detector.
3. **Low Confidence (`low_confidence/`)**: Debris detected with low certainty ($0.15 \le \text{conf} \le 0.40$).

---

## 🔮 9. Inference Pipelines

### Image Inference
Run detection on a single image or an entire folder:
```bash
python scripts/predict.py \
    --model outputs/training/baseline_yolo11s/weights/best.pt \
    --source dataset/images/val \
    --conf 0.35 \
    --save-dir outputs/predictions/
```

### Video Stream Inference (Memory-Safe Streaming)
Processes video files frame-by-frame without loading the entire video into RAM:
```bash
python scripts/predict_video.py \
    --model outputs/training/baseline_yolo11s/weights/best.pt \
    --source underwater_footage.mp4 \
    --output outputs/predictions/annotated_debris.mp4 \
    --conf 0.40
```

### Live Real-Time Camera / Webcam Stream
```bash
python scripts/predict_video.py \
    --model outputs/training/baseline_yolo11s/weights/best.pt \
    --source 0 \
    --show
```

---

## 🚢 10. Model Export for Deployment

Export your trained PyTorch `.pt` model to production-optimized formats:

### Export to ONNX (with FP16 Half-Precision & Dynamic Axes)
```bash
python scripts/export_model.py \
    --model outputs/training/baseline_yolo11s/weights/best.pt \
    --format onnx \
    --half \
    --dynamic
```

### Export to NVIDIA TensorRT Engine (for Jetson / Edge Devices)
```bash
python scripts/export_model.py \
    --model outputs/training/baseline_yolo11s/weights/best.pt \
    --format engine \
    --half
```

---

## 📐 11. Small Object Detection & Resolution Tradeoffs

Marine debris frequently occupies $< 2\%$ of the image area. Compare multi-resolution experiments:

| Resolution | Small Object Detection | GPU VRAM (Batch 16) | Inference Latency (RTX 4050) | Recommended Scenario |
| :--- | :--- | :--- | :--- | :--- |
| **640px** | Standard | ~2.8 GB | ~4.5 ms (~220 FPS) | Real-time edge ROVs / Jetson Nano |
| **832px** | Enhanced | ~4.2 GB | ~7.8 ms (~128 FPS) | High-accuracy surface patrol drones |
| **1024px** | Maximum | ~5.6 GB | ~12.2 ms (~82 FPS) | Offline deep seabed survey analysis |

---

## 🛠️ 12. Troubleshooting Playbook

### 1. CUDA Out-Of-Memory (OOM)
* **Symptom**: `RuntimeError: CUDA out of memory`.
* **Fix**: Reduce batch size in `configs/train_config.yaml` from `16` to `8` or `4`, or enable Automatic Mixed Precision (`amp: true`).

### 2. Low Recall (Missed Debris)
* **Symptom**: Model has high precision but misses small floating plastics.
* **Fix**:
  1. Lower the confidence threshold: `--conf 0.25` or `--conf 0.20`.
  2. Increase training resolution to `--imgsz 832` or `--imgsz 1024`.
  3. Increase `mosaic` (1.0) and `scale` (0.5) in `train_config.yaml`.

### 3. False Positives on Rocks & Coral
* **Symptom**: Natural seabed textures are misclassified as debris.
* **Fix**: Add **Background Images** (images containing seabed, rocks, and algae with **empty `.txt` label files**). Standard practice is 10%–15% background images in the training set.

### 4. Video Inference Window Closes Immediately
* **Symptom**: `Could not open video source`.
* **Fix**: Verify video file path or verify webcam index (`--source 0` vs `--source 1`).
