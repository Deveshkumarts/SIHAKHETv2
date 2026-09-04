"""
Convolutional Autoencoder (CAE) for Acoustic Anomaly Detection.
Unsupervised learning branch for identifying novel, uncataloged subsea objects,
unusual seabed debris, or geometric anomalies via reconstruction loss.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SonarConvAutoencoder(nn.Module):
    """
    Lightweight, high-capacity Convolutional Autoencoder tailored for acoustic textures.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (32, 64, 64) for 128x128
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (64, 32, 32)
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (128, 16, 16)
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (256, 8, 8)

        # Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (128, 16, 16)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (64, 32, 32)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )  # -> (32, 64, 64)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )  # -> (in_channels, 128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        latent = self.enc4(e3)

        d4 = self.dec4(latent)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        reconstruction = self.dec1(d2)
        return reconstruction


class SonarAnomalyDetector:
    """
    Inference wrapper for Convolutional Autoencoder Anomaly Detection.
    Computes structural reconstruction residual maps and detects unclassified anomalies.
    """

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = "weights/autoencoder_best.pt",
        img_size: int = 128,
        device: str = "auto",
        anomaly_threshold_std: float = 2.5
    ):
        self.img_size = img_size
        self.threshold_std = anomaly_threshold_std

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = SonarConvAutoencoder(in_channels=3, latent_dim=128).to(self.device)
        self.has_weights = False

        if weights_path:
            p = Path(weights_path)
            if p.exists() and p.stat().st_size > 1000:
                try:
                    state = torch.load(str(p), map_location=self.device, weights_only=True)
                    self.model.load_state_dict(state)
                    self.has_weights = True
                except Exception:
                    pass

        self.model.eval()

    def compute_reconstruction(
        self,
        image_bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Reconstructs the image and produces a normalized error map.

        Returns:
            (reconstructed_bgr, error_map_uint8, mean_reconstruction_loss)
        """
        h_orig, w_orig = image_bgr.shape[:2]
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        tensor_in = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor_in = tensor_in.unsqueeze(0).to(self.device)

        with torch.no_grad():
            tensor_out = self.model(tensor_in)
            # Squared error per pixel
            diff = (tensor_in - tensor_out) ** 2
            error_map_small = diff.squeeze().mean(dim=0).cpu().numpy()  # (H, W)

            # Reconstructed image array
            recon_small = (tensor_out.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Upscale error map and reconstruction to original dimensions
        error_map = cv2.resize(error_map_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        recon_bgr = cv2.cvtColor(cv2.resize(recon_small, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR)

        # Normalize error map to [0, 255] uint8
        mean_err = float(np.mean(error_map))
        p99 = float(np.percentile(error_map, 99))
        error_norm = np.clip((error_map / max(1e-5, p99)) * 255.0, 0, 255).astype(np.uint8)

        return recon_bgr, error_norm, mean_err

    def detect_anomalies(
        self,
        image_bgr: np.ndarray,
        min_anomaly_area: int = 20,
        max_anomaly_area: int = 15000,
        sensitivity: float = 0.85
    ) -> List[Dict[str, any]]:
        """
        Finds regions in the image with high reconstruction loss indicating novel/unknown targets.

        Returns:
            List of detected anomaly regions:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "anomaly_score": float, # [0.0 - 1.0]
                    "area": int,
                    "type": "Unknown Anomaly",
                    "source": "ConvAutoencoder"
                }
            ]
        """
        recon_bgr, error_map, mean_err = self.compute_reconstruction(image_bgr)
        h, w = image_bgr.shape[:2]

        # Dynamic threshold based on sensitivity
        thresh_val = int(255.0 * (1.0 - (0.45 * sensitivity)))
        _, binary_err = cv2.threshold(error_map, thresh_val, 255, cv2.THRESH_BINARY)

        # Clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_clean = cv2.morphologyEx(binary_err, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clean, connectivity=8)

        anomalies = []
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            if min_anomaly_area <= area <= max_anomaly_area:
                patch = error_map[y:y+bh, x:x+bw]
                peak_err = float(np.max(patch)) / 255.0 if patch.size > 0 else 0.0
                mean_patch = float(np.mean(patch)) / 255.0 if patch.size > 0 else 0.0
                score = round(0.6 * peak_err + 0.4 * mean_patch, 3)

                anomalies.append({
                    "bbox": [x, y, x + bw, y + bh],
                    "anomaly_score": score,
                    "area": area,
                    "type": "Unknown Anomaly",
                    "source": "ConvAutoencoder"
                })

        return anomalies, error_map
