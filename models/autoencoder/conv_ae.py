"""
Convolutional Autoencoder for unknown-anomaly detection on side-scan sonar ROIs.

Trained ONLY on normal seabed (sand / rock / sediment / ripple texture). Anything the network
cannot reconstruct well -- i.e. structure it has never seen as background -- gets a high
reconstruction error and is flagged as a candidate unknown anomaly.

A narrow fully-connected bottleneck (latent_dim) is essential: the old fully-convolutional
16k-dimensional latent could reconstruct almost anything, anomalies included.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _down(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(ci, co, 4, 2, 1), nn.BatchNorm2d(co), nn.LeakyReLU(0.2, inplace=True))


def _up(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(nn.ConvTranspose2d(ci, co, 4, 2, 1), nn.BatchNorm2d(co), nn.LeakyReLU(0.2, inplace=True))


class PatchConvAE(nn.Module):
    def __init__(self, patch: int = 64, latent: int = 128, in_ch: int = 1):
        super().__init__()
        assert patch % 16 == 0, "patch size must be a multiple of 16"
        self.patch, self.latent, self.in_ch = patch, latent, in_ch
        self.s = patch // 16
        self.encoder = nn.Sequential(_down(in_ch, 16), _down(16, 32), _down(32, 64), _down(64, 128))
        self.to_latent = nn.Linear(128 * self.s * self.s, latent)
        self.from_latent = nn.Linear(latent, 128 * self.s * self.s)
        self.decoder = nn.Sequential(_up(128, 64), _up(64, 32), _up(32, 16),
                                     nn.ConvTranspose2d(16, in_ch, 4, 2, 1), nn.Sigmoid())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(x).flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.from_latent(z).view(-1, 128, self.s, self.s))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
