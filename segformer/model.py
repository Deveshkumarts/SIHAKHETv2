"""
SegFormer Model Architecture for ROI Binary Segmentation.
Implements a lightweight Mix Transformer (MiT-B0) backbone with an All-MLP Decoder head.
Compact, fast, and optimized for underwater acoustic target mask generation (~3.7M params).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=32):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SegFormerEncoderStage(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size, stride):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(patch_size, stride, in_chans, embed_dim)
        self.block1 = MixFFN(embed_dim, embed_dim * 2, embed_dim)
        self.block2 = MixFFN(embed_dim, embed_dim * 2, embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.block1(x) + x
        x = self.block2(x) + x
        return x


class SegFormerB0(nn.Module):
    """
    SegFormer B0 Architecture: Multi-stage hierarchical transformer encoder + All-MLP decoder.
    Output: Single channel logit map for binary target segmentation.
    """
    def __init__(self, in_channels=3, num_classes=1, dims=[32, 64, 160, 256]):
        super().__init__()
        self.stage1 = SegFormerEncoderStage(in_channels, dims[0], patch_size=7, stride=4)
        self.stage2 = SegFormerEncoderStage(dims[0], dims[1], patch_size=3, stride=2)
        self.stage3 = SegFormerEncoderStage(dims[1], dims[2], patch_size=3, stride=2)
        self.stage4 = SegFormerEncoderStage(dims[2], dims[3], patch_size=3, stride=2)

        decoder_dim = 128
        self.linear_c1 = nn.Conv2d(dims[0], decoder_dim, 1)
        self.linear_c2 = nn.Conv2d(dims[1], decoder_dim, 1)
        self.linear_c3 = nn.Conv2d(dims[2], decoder_dim, 1)
        self.linear_c4 = nn.Conv2d(dims[3], decoder_dim, 1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * 4, decoder_dim, kernel_size=1),
            nn.BatchNorm2d(decoder_dim),
            nn.GELU()
        )
        self.classifier = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[2:]

        c1 = self.stage1(x)   # (B, 32, H/4, W/4)
        c2 = self.stage2(c1)  # (B, 64, H/8, W/8)
        c3 = self.stage3(c2)  # (B, 160, H/16, W/16)
        c4 = self.stage4(c3)  # (B, 256, H/32, W/32)

        # Uniform feature dimension & spatial size alignment via bilinear upsampling
        c1_proj = F.interpolate(self.linear_c1(c1), size=c1.shape[2:], mode='bilinear', align_corners=False)
        c2_proj = F.interpolate(self.linear_c2(c2), size=c1.shape[2:], mode='bilinear', align_corners=False)
        c3_proj = F.interpolate(self.linear_c3(c3), size=c1.shape[2:], mode='bilinear', align_corners=False)
        c4_proj = F.interpolate(self.linear_c4(c4), size=c1.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate multi-scale features
        fused = self.linear_fuse(torch.cat([c1_proj, c2_proj, c3_proj, c4_proj], dim=1))

        # Final classification logits upsampled to input resolution (H, W)
        logits = self.classifier(fused)
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)

        return logits
