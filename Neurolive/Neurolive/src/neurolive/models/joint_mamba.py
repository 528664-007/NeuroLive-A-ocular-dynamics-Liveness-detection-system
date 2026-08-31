"""
Phase 2 — unified joint model: one shared temporal backbone, two heads
(segmentation + liveness classification), trained with a multi-task loss.
This is what replaces the paper's separate TCN + SCNN (Phase 1).

Backbone is `mamba-ssm` if it's importable (needs a working CUDA build —
flaky on some Windows/WSL + 6GB-VRAM setups per README), otherwise falls
back automatically to a depthwise Conv1D + GRU stack with the *identical*
(B, T, D) -> (B, T, D) interface, so the rest of the model and the training
script don't need to know which one is active. `JointLivenessModel.backbone_name`
tells you which one you actually got — log it, don't assume.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False


class _ConvGRUFallbackBackbone(nn.Module):
    """Same (B, T, D) -> (B, T, D) contract as a Mamba block, used when
    mamba-ssm isn't importable. Not a novel architecture — just a
    dependency-free stand-in so Phase 2 stays runnable everywhere.
    """

    def __init__(self, d_model: int, num_layers: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.gru(x)
        return out


class JointLivenessModel(nn.Module):
    """Input: (B, T, H, W) voxel grid (T = num_bins).
    Outputs: seg_logits (B, T, num_seg_classes), liveness_logits (B, num_liveness_classes).
    """

    def __init__(
        self,
        num_seg_classes: int = 3,
        num_liveness_classes: int = 2,
        d_model: int = 128,
        num_backbone_layers: int = 4,
        force_fallback: bool = False,
    ):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Linear(64 * 4 * 4, d_model)

        use_mamba = _MAMBA_AVAILABLE and not force_fallback
        if use_mamba:
            self.backbone = Mamba(d_model=d_model)
            self.backbone_name = "mamba_ssm"
        else:
            self.backbone = _ConvGRUFallbackBackbone(d_model, num_backbone_layers)
            self.backbone_name = "conv_gru_fallback"

        self.seg_head = nn.Linear(d_model, num_seg_classes)
        # liveness head pools over time then classifies
        self.liveness_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, num_liveness_classes)
        )

    def forward(self, voxel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, h, w = voxel.shape
        frames = voxel.reshape(b * t, 1, h, w)
        feats = self.spatial_encoder(frames).reshape(b, t, -1)
        feats = self.proj(feats)              # (B, T, D)
        hidden = self.backbone(feats)         # (B, T, D)

        seg_logits = self.seg_head(hidden)                       # (B, T, num_seg_classes)
        pooled = hidden.mean(dim=1)                               # (B, D)
        liveness_logits = self.liveness_head(pooled)              # (B, num_liveness_classes)
        return seg_logits, liveness_logits
