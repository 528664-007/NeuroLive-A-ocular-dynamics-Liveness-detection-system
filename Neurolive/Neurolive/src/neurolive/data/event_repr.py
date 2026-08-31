"""
Event-stream -> tensor representations.

An event camera outputs a sparse stream of (x, y, t, p) tuples: pixel
coordinate, microsecond timestamp, and polarity (+1 / -1 brightness change).
Everything downstream (TCN, SCNN, Mamba backbone) consumes one of the two
dense representations built here, not raw events.

References for the representations themselves (standard in event-vision
literature, not paper-specific):
  - Voxel grid: Zhu et al., "Unsupervised Event-based Learning of Optical
    Flow, Depth, and Egomotion", CVPR 2019.
  - Time surface: Lagorce et al., "HOTS: A Hierarchy Of event-based
    Time-Surfaces", PAMI 2017.
"""
from __future__ import annotations

import numpy as np
import torch


def events_to_voxel_grid(
    events: np.ndarray,
    num_bins: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Bilinearly-interpolated voxel grid, standard event->tensor encoding.

    Args:
        events: (N, 4) array of [x, y, t, p], t in microseconds ascending,
            p in {-1, +1}.
        num_bins: number of temporal bins to accumulate into.
        height, width: sensor resolution.

    Returns:
        FloatTensor of shape (num_bins, height, width).
    """
    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    if events.shape[0] == 0:
        return torch.from_numpy(voxel)

    x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    t_min, t_max = t.min(), t.max()
    dt = max(t_max - t_min, 1.0)
    # normalize timestamps to [0, num_bins - 1]
    t_norm = (t - t_min) / dt * (num_bins - 1)

    left = np.floor(t_norm).astype(np.int64)
    right = np.clip(left + 1, 0, num_bins - 1)
    right_w = t_norm - left
    left_w = 1.0 - right_w

    x = x.astype(np.int64)
    y = y.astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, p, left, right, left_w, right_w = (
        x[valid], y[valid], p[valid], left[valid], right[valid],
        left_w[valid], right_w[valid],
    )

    np.add.at(voxel, (left, y, x), p * left_w)
    np.add.at(voxel, (right, y, x), p * right_w)
    return torch.from_numpy(voxel)


def events_to_time_surface(
    events: np.ndarray,
    height: int,
    width: int,
    tau_us: float = 5000.0,
) -> torch.Tensor:
    """Exponentially-decayed time surface at the last event's timestamp.

    Each pixel holds exp(-(t_last - t_pixel) / tau), i.e. "how recently did
    this pixel fire". Two channels: positive-polarity surface, negative.
    """
    surf = np.zeros((2, height, width), dtype=np.float32)
    if events.shape[0] == 0:
        return torch.from_numpy(surf)

    x, y, t, p = events[:, 0].astype(np.int64), events[:, 1].astype(np.int64), events[:, 2], events[:, 3]
    t_ref = t.max()
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[valid], y[valid], t[valid], p[valid]

    for ch, pol in enumerate((1, -1)):
        mask = p == pol
        if not mask.any():
            continue
        # last-writer-wins per pixel: events are time-ordered, so a simple
        # scatter assignment naturally keeps the most recent timestamp.
        last_t = np.full((height, width), -np.inf, dtype=np.float64)
        last_t[y[mask], x[mask]] = t[mask]
        decay = np.exp(-(t_ref - last_t) / tau_us)
        decay[np.isneginf(last_t)] = 0.0
        surf[ch] = decay.astype(np.float32)

    return torch.from_numpy(surf)


def activity_profile(events: np.ndarray, num_bins: int) -> np.ndarray:
    """1-D event-count-per-bin signal — the 'activity profile' used for the
    peak-detection blink segmentation baseline and for the demo's decision
    explanation plot (paper's Fig. 4/5 style).
    """
    profile = np.zeros(num_bins, dtype=np.float32)
    if events.shape[0] == 0:
        return profile
    t = events[:, 2]
    t_min, t_max = t.min(), t.max()
    dt = max(t_max - t_min, 1.0)
    bin_idx = np.clip(((t - t_min) / dt * (num_bins - 1)).astype(np.int64), 0, num_bins - 1)
    np.add.at(profile, bin_idx, 1.0)
    return profile
