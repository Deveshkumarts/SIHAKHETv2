"""
Active Learning & Human-in-the-Loop (HITL) Feedback Engine.
Queues high-uncertainty detections and unclassified anomalies for expert review,
stores ground-truth corrections, and exports training samples for YOLO/ResNet fine-tuning.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import cv2
import numpy as np


class ActiveLearningManager:
    """
    Manages the Human-in-the-Loop Active Learning feedback pipeline.
    """

    def __init__(self, data_dir: Union[str, Path] = "data/active_learning"):
        self.data_dir = Path(data_dir)
        self.crops_dir = self.data_dir / "crops"
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.queue_file = self.data_dir / "review_queue.json"
        self.archive_file = self.data_dir / "reviewed_samples.json"
        self._init_files()

    def _init_files(self):
        if not self.queue_file.exists():
            with open(self.queue_file, "w", encoding="utf-8") as f:
                json.dump([], f)
        if not self.archive_file.exists():
            with open(self.archive_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _read_json(self, path: Path) -> List[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_json(self, path: Path, data: List[Dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def enqueue_for_review(
        self,
        detection: Dict[str, Any],
        roi_crop: Optional[np.ndarray] = None,
        reason: str = "High Uncertainty"
    ) -> str:
        """
        Enqueues an uncertain detection or anomaly for human expert review.
        """
        queue = self._read_json(self.queue_file)
        sample_id = f"sample_{int(time.time() * 1000)}"

        crop_rel_path = ""
        if roi_crop is not None and roi_crop.size > 0:
            crop_filename = f"{sample_id}.jpg"
            crop_full_path = self.crops_dir / crop_filename
            cv2.imwrite(str(crop_full_path), roi_crop)
            crop_rel_path = f"data/active_learning/crops/{crop_filename}"

        record = {
            "id": sample_id,
            "class_name": detection.get("class_name", "Target"),
            "confidence": detection.get("conf", 0.0),
            "uncertainty_flag": detection.get("uncertainty_flag", "HIGH"),
            "uncertainty_variance": detection.get("uncertainty_variance", 0.0),
            "latitude": detection.get("latitude", 0.0),
            "longitude": detection.get("longitude", 0.0),
            "error_ellipse_a": detection.get("error_ellipse_a", 0.0),
            "crop_path": crop_rel_path,
            "flag_reason": reason,
            "timestamp": time.time(),
            "status": "PENDING"
        }

        queue.append(record)
        self._write_json(self.queue_file, queue)
        return sample_id

    def get_pending_queue(self) -> List[Dict[str, Any]]:
        queue = self._read_json(self.queue_file)
        return [q for q in queue if q.get("status") == "PENDING"]

    def submit_review(
        self,
        sample_id: str,
        action: str,  # 'CONFIRM' | 'RELABEL' | 'REJECT'
        corrected_class: Optional[str] = None,
        operator_notes: str = ""
    ):
        """
        Processes human operator triage decision and moves sample to reviewed archive.
        """
        queue = self._read_json(self.queue_file)
        archive = self._read_json(self.archive_file)

        updated_queue = []
        for item in queue:
            if item["id"] == sample_id:
                item["status"] = "REVIEWED"
                item["action"] = action
                item["final_class"] = corrected_class if action == "RELABEL" else item["class_name"]
                item["operator_notes"] = operator_notes
                item["reviewed_time"] = time.time()
                archive.append(item)
            else:
                updated_queue.append(item)

        self._write_json(self.queue_file, updated_queue)
        self._write_json(self.archive_file, archive)

    def get_archive_stats(self) -> Dict[str, int]:
        archive = self._read_json(self.archive_file)
        return {
            "total_reviewed": len(archive),
            "confirmed": sum(1 for a in archive if a.get("action") == "CONFIRM"),
            "relabeled": sum(1 for a in archive if a.get("action") == "RELABEL"),
            "rejected": sum(1 for a in archive if a.get("action") == "REJECT"),
        }
