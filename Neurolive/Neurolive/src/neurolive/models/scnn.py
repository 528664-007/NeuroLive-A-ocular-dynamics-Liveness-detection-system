"""
Phase 1 baseline — Spiking CNN for genuine/attack liveness classification,
replicating the paper's SCNN branch (their best-performing baseline:
95.37% top-1 accuracy / 4.65% ACER on their dataset — see STATUS.md, that
number is theirs, not reproduced yet here).

Uses snntorch's surrogate-gradient leaky-integrate-and-fire (LIF) neurons,
trained with backprop-through-time over the voxel-grid time steps — this is
the standard way spiking nets are trained when you don't have neuromorphic
hardware in the loop, and matches common SCNN liveness-detection setups.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import snntorch as snn
import snntorch.utils
from snntorch import surrogate


class SpikingConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, beta: float = 0.9):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.lif = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid(), init_hidden=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif(self.pool(self.conv(x)))


class LivenessSCNN(nn.Module):
    """Input: (B, num_bins, H, W) voxel grid, treated as a T-step spike
    train (one frame per time step). Output: (B, 2) genuine-vs-attack logits,
    summed over time steps (rate coding).
    """

    def __init__(self, num_classes: int = 2, channels=(1, 16, 32)):
        super().__init__()
        self.block1 = SpikingConvBlock(channels[0], channels[1])
        self.block2 = SpikingConvBlock(channels[1], channels[2])
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(channels[2] * 4 * 4, num_classes)
        self.out_lif = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid(), init_hidden=True)

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        b, t, h, w = voxel.shape
        # reset hidden spiking state for this batch
        snn.utils.reset(self)

        spike_sum = torch.zeros(b, self.fc.out_features, device=voxel.device)
        for step in range(t):
            frame = voxel[:, step : step + 1]  # (B, 1, H, W)
            x = self.block1(frame)
            x = self.block2(x)
            x = self.pool(x).flatten(1)
            cur = self.fc(x)
            spk = self.out_lif(cur)
            spike_sum = spike_sum + spk
        return spike_sum  # (B, num_classes) — rate-coded logits
