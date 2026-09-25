"""Stage 7a - YOLO11-Seg service (known debris: box + class + confidence + instance mask + area + centroid)."""

from __future__ import annotations

from typing import Optional

from models.yolo11_seg.inference import YOLOSegDetector


def load_yolo_seg(cfg) -> YOLOSegDetector:
    y = cfg.yolo_seg
    weights = cfg.path("yolo_seg.weights")
    if cfg.get("export.backend", "pytorch") == "onnx":                      # ultralytics runs .onnx through ONNX Runtime
        onnx = cfg.path("export.onnx_dir") / "yolo11n_seg.onnx"
        weights = onnx if onnx.exists() else weights
    return YOLOSegDetector(weights, imgsz=y.imgsz, conf=y.conf, iou=y.iou, device=y.device,
                           half=y.half, min_mask_area_px=y.min_mask_area_px)
