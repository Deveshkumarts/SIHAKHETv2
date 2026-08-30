"""
Reusable YOLO11 Detection Interface.
Wraps the fine-tuned YOLO11 model (832px, conf 0.15) and returns structured detections.
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.device_utils import select_device

DEFAULT_BEST_MODEL_PATH = "outputs/experiments/exp_resolution_832/weights/best.pt"
DEFAULT_DATA_YAML = "Combined_Dataset/data.yaml"


class YOLO11Detector:
    def __init__(
        self,
        model_path: str = DEFAULT_BEST_MODEL_PATH,
        conf_thresh: float = 0.15,
        iou_thresh: float = 0.5,
        imgsz: int = 832,
        device_str: str = "0"
    ):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            # Fallback to general training output if exp_resolution_832 path is missing
            fallback = Path("outputs/training/yolo11s_combined_dataset/weights/best.pt").resolve()
            if fallback.exists():
                self.model_path = fallback
            else:
                raise FileNotFoundError(f"YOLO11 model weights not found at: {self.model_path}")

        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.device = select_device(device_str)

        from ultralytics import YOLO
        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names

    def detect(self, image_input: Union[str, Path, np.ndarray]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Run YOLO11 detection on an image path or OpenCV BGR numpy array.
        
        Returns:
            (image_bgr, list_of_detections)
            Where each detection is:
            {
                "class_id": int,
                "class_name": str,
                "confidence": float,
                "bbox": [x1, y1, x2, y2]
            }
        """
        if isinstance(image_input, (str, Path)):
            image_bgr = cv2.imread(str(image_input))
            if image_bgr is None:
                raise ValueError(f"Failed to load image from path: {image_input}")
        else:
            image_bgr = image_input

        results = self.model.predict(
            source=image_bgr,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0])
            cid = int(box.cls[0])
            cname = self.class_names.get(cid, str(cid))

            detections.append({
                "class_id": cid,
                "class_name": cname,
                "confidence": round(conf, 4),
                "bbox": [round(coord, 2) for coord in xyxy]
            })

        return image_bgr, detections
