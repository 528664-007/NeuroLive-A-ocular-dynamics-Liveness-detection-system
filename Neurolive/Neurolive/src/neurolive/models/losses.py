"""Multi-task loss for the Phase 2 joint model: segmentation CE + liveness CE,
weighted sum. Weights are a training hyperparameter, not fixed by the paper
(the paper doesn't jointly train these — that's the whole point of Phase 2),
so default to equal weighting and expose it as a CLI flag in train_joint.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskLivenessLoss(nn.Module):
    def __init__(self, seg_weight: float = 1.0, liveness_weight: float = 1.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.liveness_weight = liveness_weight
        self.seg_ce = nn.CrossEntropyLoss()
        self.liveness_ce = nn.CrossEntropyLoss()

    def forward(
        self,
        seg_logits: torch.Tensor,     # (B, T, num_seg_classes)
        liveness_logits: torch.Tensor,  # (B, num_liveness_classes)
        seg_labels: torch.Tensor,     # (B, T)
        liveness_labels: torch.Tensor,  # (B,)
    ) -> dict:
        seg_loss = self.seg_ce(seg_logits.reshape(-1, seg_logits.shape[-1]), seg_labels.reshape(-1))
        liveness_loss = self.liveness_ce(liveness_logits, liveness_labels)
        total = self.seg_weight * seg_loss + self.liveness_weight * liveness_loss
        return {"total": total, "seg_loss": seg_loss, "liveness_loss": liveness_loss}
