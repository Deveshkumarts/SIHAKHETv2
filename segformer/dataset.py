"""
PyTorch Dataset Loader for Sonar ROI Segmentation.
Applies aligned spatial & intensity augmentations to image and binary mask simultaneously.
"""

import os
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any, List


class SonarSegDataset(Dataset):
    """
    Dataset loader for ROI images and binary segmentation masks.
    """
    def __init__(
        self,
        rois_dir: str,
        masks_dir: str,
        img_size: Tuple[int, int] = (224, 224),
        is_train: bool = True
    ):
        self.rois_dir = Path(rois_dir).resolve()
        self.masks_dir = Path(masks_dir).resolve()
        self.img_size = img_size
        self.is_train = is_train

        self.roi_files = sorted([p for p in self.rois_dir.glob("*.png")])
        if len(self.roi_files) == 0:
            raise FileNotFoundError(f"No ROI PNG files found in directory: {self.rois_dir}")

    def __len__(self) -> int:
        return len(self.roi_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        roi_path = self.roi_files[idx]
        mask_path = self.masks_dir / roi_path.name

        img_bgr = cv2.imread(str(roi_path))
        if img_bgr is None:
            raise ValueError(f"Failed to load ROI image: {roi_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if mask_path.exists():
            mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask_gray = np.zeros(img_rgb.shape[:2], dtype=np.uint8)

        # Resize to fixed input size
        img_resized = cv2.resize(img_rgb, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_gray, self.img_size, interpolation=cv2.INTER_NEAREST)
        binary_mask = (mask_resized > 127).astype(np.float32)

        # Apply aligned data augmentation during training
        if self.is_train:
            # 1. Random Horizontal Flip
            if np.random.rand() > 0.5:
                img_resized = cv2.flip(img_resized, 1)
                binary_mask = cv2.flip(binary_mask, 1)

            # 2. Random Vertical Flip
            if np.random.rand() > 0.5:
                img_resized = cv2.flip(img_resized, 0)
                binary_mask = cv2.flip(binary_mask, 0)

            # 3. Random Brightness / Contrast
            if np.random.rand() > 0.5:
                alpha = 0.8 + np.random.rand() * 0.4  # [0.8, 1.2]
                beta = (np.random.rand() - 0.5) * 30  # [-15, 15]
                img_resized = np.clip(img_resized * alpha + beta, 0, 255).astype(np.uint8)

        # Normalize image to [0, 1] and convert to PyTorch tensors (C, H, W)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).float()  # Shape: (1, H, W)

        return img_tensor, mask_tensor, roi_path.name
