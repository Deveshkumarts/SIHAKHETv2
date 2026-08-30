"""
SegFormer Inference Module for Sonar ROI Segmentation.
Loads best.pt and predicts binary segmentation mask on a single ROI image.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segformer.model import SegFormerB0
from utils.device_utils import select_device


class SegFormerInference:
    def __init__(
        self,
        weights_path: str = "outputs/segformer/weights/best.pt",
        img_size: int = 224,
        device_str: str = "0",
        threshold: float = 0.5
    ):
        self.img_size = img_size
        self.threshold = threshold
        device_str = select_device(device_str)
        self.device = torch.device(f"cuda:{device_str}" if device_str.isdigit() else device_str)

        self.model = SegFormerB0(in_channels=3, num_classes=1)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, roi_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Predict binary segmentation mask for a single ROI crop.

        Args:
            roi_bgr: BGR numpy array (any size)

        Returns:
            (binary_mask_uint8, foreground_probability_score)
            - binary_mask: (H, W) uint8 mask at input resolution, 0 or 255
            - fg_score: mean foreground probability from raw sigmoid output
        """
        h_orig, w_orig = roi_bgr.shape[:2]

        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        img_tensor = torch.from_numpy(roi_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(img_tensor)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W)

        # Foreground confidence score: mean probability within predicted mask region
        fg_score = float(prob_map.mean())

        # Resize prob map back to original ROI resolution using nearest neighbor
        prob_resized = cv2.resize(prob_map, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        binary_mask = (prob_resized >= self.threshold).astype(np.uint8) * 255

        return binary_mask, fg_score
