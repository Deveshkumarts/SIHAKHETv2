from ultralytics import YOLO
import torch
from pathlib import Path

DATA_YAML = r"C:\Users\CMRMuthuthiyagarajan\Downloads\SIH26\SIH_Dataset_27class\data.yaml"
WEIGHTS    = r"C:\Users\CMRMuthuthiyagarajan\Downloads\SIH26\yolo11s.pt"
PROJECT    = r"C:\Users\CMRMuthuthiyagarajan\Downloads\SIH26\runs\detect\sih27class"
NAME       = "yolo11s_sih_27class"

print("="*60)
print("  SIH 2026 - YOLOv11s Training - 27 Classes")
print(f"  GPU : {torch.cuda.get_device_name(0)}")
print(f"  Data: {DATA_YAML}")
print("="*60)

model = YOLO(WEIGHTS)
results = model.train(
    data=DATA_YAML,
    epochs=80,
    imgsz=640,
    batch=16,
    device=0,
    project=PROJECT,
    name=NAME,
    exist_ok=True,
    patience=20,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    degrees=15.0,
    translate=0.1,
    scale=0.5,
    verbose=True,
    save=True,
    save_period=10,
    val=True,
    plots=True,
    workers=0,
)
print("\n" + "="*60)
print("  TRAINING COMPLETE!")
print(f"  Best mAP50     : {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
print(f"  Best mAP50-95  : {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
print("="*60)
