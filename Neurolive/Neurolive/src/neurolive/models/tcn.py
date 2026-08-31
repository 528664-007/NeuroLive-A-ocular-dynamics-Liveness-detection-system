"""
Phase 1 baseline — Temporal Convolutional Network for ocular-movement
(saccade) segmentation, replicating the paper's TCN branch.

Standard dilated-causal-conv TCN (Bai et al., 2018 architecture pattern),
operating on the per-frame voxel-grid sequence's spatial-pooled features.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp = pad  # causal: trim the right-side padding after each conv
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def _causal(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : -self.chomp] if self.chomp > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self._causal(self.conv1(x))))
        out = self.dropout(self.relu(self._causal(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class OcularTCN(nn.Module):
    """Input: (B, num_bins, H, W) voxel grid. Output: (B, num_bins, num_classes)
    per-frame segmentation logits (background / saccade / blink, matching
    the paper's 3-way ocular-event segmentation target).
    """

    def __init__(self, num_bins: int, num_classes: int = 3, channels=(32, 64, 64), kernel_size: int = 5):
        super().__init__()
        # spatial feature extractor applied per temporal frame, shared weights
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        feat_dim = 32 * 4 * 4

        layers = []
        in_ch = feat_dim
        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2 ** i))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(channels[-1], num_classes, 1)

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        b, t, h, w = voxel.shape
        frames = voxel.reshape(b * t, 1, h, w)
        feats = self.spatial_encoder(frames).reshape(b, t, -1).transpose(1, 2)  # (B, feat_dim, T)
        out = self.tcn(feats)          # (B, C, T)
        logits = self.head(out)        # (B, num_classes, T)
        return logits.transpose(1, 2)  # (B, T, num_classes)
