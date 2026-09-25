"""
YOLO11-Seg inference: known-debris detection + classification + instance segmentation in ONE model.

Replaces the old YOLOv11 -> SegFormer-B0 -> ResNet18 chain. Each detection carries:
bbox, class, confidence, instance mask (polygon + raster), mask area, centroid.
Hardware independent: 'cpu' | 'cuda:0' | 'auto'; the same code path serves PyTorch weights (.pt) and
exported ONNX (.onnx) through ultralytics, so the Jetson deployment differs only in the weights file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from backend.pipeline.base import file_version


def mask_stats(mask: np.ndarray) -> Dict[str, Any]:
    """Area (px) and centroid (x, y) of a binary mask."""
    m = (mask > 0).astype(np.uint8)
    area = int(m.sum())
    if area == 0:
        return {"area": 0, "centroid": None}
    mo = cv2.moments(m, binaryImage=True)
    return {"area": area, "centroid": (float(mo["m10"] / mo["m00"]), float(mo["m01"] / mo["m00"]))}


class YOLOSegDetector:
    def __init__(self, weights: str | Path, imgsz: int = 640, conf: float = 0.25, iou: float = 0.45,
                 device: str = "auto", half: bool = False, min_mask_area_px: int = 6):
        self.weights = Path(weights)
        self.imgsz, self.conf, self.iou, self.half = imgsz, conf, iou, half
        self.min_mask_area_px = min_mask_area_px
        self.device = self._resolve(device)
        self.version = file_version(self.weights, "yolo11-seg")
        self.model = None
        self.class_names: Dict[int, str] = {}
        if self.weights.exists() and self.weights.stat().st_size > 0:
            from ultralytics import YOLO
            self.model = YOLO(str(self.weights))
            self.class_names = dict(self.model.names)

    @staticmethod
    def _resolve(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @property
    def available(self) -> bool:
        return self.model is not None

    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        h, w = image_bgr.shape[:2]
        res = self.model.predict(source=image_bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                 device=self.device, half=self.half and self.device != "cpu", verbose=False)[0]
        out: List[Dict[str, Any]] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        polys = res.masks.xy if res.masks is not None else [None] * len(res.boxes)
        for i, box in enumerate(res.boxes):
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].cpu().numpy()]
            cid = int(box.cls[0])
            poly = polys[i]
            mask = np.zeros((h, w), np.uint8)
            if poly is not None and len(poly) >= 3:
                cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
            stats = mask_stats(mask)
            if stats["centroid"] is None:                      # no usable mask -> box centre, area 0
                stats["centroid"] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            if 0 < stats["area"] < self.min_mask_area_px:
                continue
            out.append({
                "bbox": [x1, y1, x2, y2],
                "class_id": cid,
                "class_name": self.class_names.get(cid, f"cls_{cid}"),
                "conf": float(box.conf[0]),
                "mask": mask.astype(bool),
                "mask_polygon": [[float(px), float(py)] for px, py in (poly if poly is not None else [])],
                "mask_area_px": stats["area"],
                "centroid": stats["centroid"],
                "source": "YOLO11-Seg",
            })
        return out
