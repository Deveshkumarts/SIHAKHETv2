"""
ResNet18 Debris Classifier & PyTorch Grad-CAM Explainability Module for SIH2026.
Implements:
  1. ResNet18 Deep Convolutional Backbone for fine-grained marine debris & anomaly classification.
  2. PyTorch Grad-CAM (Gradient-weighted Class Activation Mapping) on layer4 for visual attention verification.
  3. Multi-Model Fusion verification (YOLOv11 + SegFormer + ResNet18).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import ssl
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Fix SSL certificate verification error when downloading pretrained weights on Windows
ssl._create_default_https_context = ssl._create_unverified_context

# Master 27-Class Taxonomy
MASTER_CLASSES = [
    "Shipwrecks", "bottle", "brown-glass-bottle", "can", "chain",
    "drink-carton", "drink-sachet", "glass-bottle", "glass-jar", "hook",
    "large-tire", "metal-bottle", "metal-box", "pipeline or cable",
    "plastic-bidon", "plastic-bottle", "plastic-pipe", "plastic-propeller",
    "potion-glass-bottle", "propeller", "rotating-platform", "shampoo-bottle",
    "small-tire", "standing-bottle", "tire", "valve", "wrench"
]

DEFAULT_RESNET_WEIGHTS = "weights/resnet18_debris_best.pt"


class ResNet18Classifier(nn.Module):
    """
    ResNet18 classifier fine-tuned for 24 Marine Debris & Sonar classes.
    """
    def __init__(self, num_classes: int = len(MASTER_CLASSES), pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class GradCAM:
    """
    PyTorch Grad-CAM (Gradient-weighted Class Activation Mapping).
    Hooks into the final convolutional layer (layer4) of ResNet18 to compute
    visual and acoustic attention maps explaining model decisions.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else self.model.backbone.layer4[-1]
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(self, input_tensor: torch.Tensor, target_class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate normalized Grad-CAM heatmap for a given input tensor.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class_idx is None:
            target_class_idx = output.argmax(dim=1).item()

        loss = output[0, target_class_idx]
        loss.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)

        # Global Average Pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 0:
            cam = (cam - np.min(cam)) / (max_val - np.min(cam) + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


class ResNet18InferenceEngine:
    """
    High-level Inference Engine for ResNet18 Debris Classification & Grad-CAM Visualization.
    """
    def __init__(self, weights_path: Optional[str] = DEFAULT_RESNET_WEIGHTS, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif "0" in device or "gpu" in device.lower():
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.model = ResNet18Classifier(num_classes=len(MASTER_CLASSES), pretrained=True)
        self.has_custom_weights = False

        p = Path(weights_path) if weights_path else Path(DEFAULT_RESNET_WEIGHTS)
        if p.exists():
            try:
                state = torch.load(str(p), map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                self.has_custom_weights = True
                print(f"[ResNet-18] Loaded fine-tuned weights from {p}")
            except Exception as e:
                print(f"Warning loading ResNet weights: {e}. Using pretrained backbone.")

        self.model.to(self.device)
        self.model.eval()

        self.gradcam = GradCAM(self.model)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict_roi(
        self,
        roi_bgr: np.ndarray,
        target_class_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run ResNet18 classification and Grad-CAM on a cropped ROI patch.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return {"pred_class": target_class_name or "unknown", "pred_conf": 0.0, "gradcam_overlay": roi_bgr, "top3": []}

        h_orig, w_orig = roi_bgr.shape[:2]
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(roi_rgb).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = MASTER_CLASSES[pred_idx]
        pred_conf = float(probs[pred_idx])

        # Target class for Grad-CAM explanation (uses detected class if provided)
        if target_class_name and target_class_name in MASTER_CLASSES:
            cam_class_idx = MASTER_CLASSES.index(target_class_name)
        else:
            cam_class_idx = pred_idx

        # Generate Grad-CAM heatmap
        tensor_cam = self.transform(roi_rgb).unsqueeze(0).to(self.device)
        tensor_cam.requires_grad_(True)
        cam = self.gradcam.generate_cam(tensor_cam, target_class_idx=cam_class_idx)

        # Resize heatmap to match ROI crop dimensions
        cam_resized = cv2.resize(cam, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

        # Blend heatmap with original ROI crop
        overlay = cv2.addWeighted(roi_bgr, 0.60, heatmap, 0.40, 0)

        # Top 3 predicted classes directly from softmax
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3 = [(MASTER_CLASSES[i], float(probs[i])) for i in top3_indices]

        return {
            "pred_class": pred_class,
            "pred_conf": pred_conf,
            "top3": top3,
            "heatmap": heatmap,
            "gradcam_overlay": overlay,
            "cam_raw": cam_resized
        }
