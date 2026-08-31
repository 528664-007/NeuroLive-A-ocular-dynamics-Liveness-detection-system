"""
Phase 4 — event-native face/eye ROI localization, removing the paper's RGB
dependency (their Section II explicitly uses RGB-assisted detection).

v0 approach: eyes are the highest-frequency local event generators on a face
during a blink/saccade challenge (this is literally the premise the paper's
own liveness signal relies on), so a short accumulation window's event-
density map should peak at the two eye regions. This is a real, reasonable
heuristic — but it is UNVALIDATED: it has only been unit-tested on synthetic
event clouds with injected density peaks (tests/test_smoke.py), never
against a real face's event stream. Treat accuracy as unknown until you can
run it against real recordings. A learned detector (small CNN on the density
map) is the natural upgrade path once you have labeled ROI data — this file
intentionally stays heuristic-only until then, rather than training a
detector against data that doesn't exist.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class ROI:
    x: int
    y: int
    width: int
    height: int
    confidence: float  # relative peak density, NOT a calibrated probability


def localize_eye_rois(
    events: np.ndarray,
    frame_height: int,
    frame_width: int,
    roi_size: int = 48,
    smoothing_sigma: float = 3.0,
    expected_eyes: int = 2,
) -> list[ROI]:
    """Returns up to `expected_eyes` candidate ROIs, ranked by local event
    density. No RGB input, no face detector — purely event-driven.
    """
    if events.shape[0] == 0:
        return []

    density = np.zeros((frame_height, frame_width), dtype=np.float32)
    x = np.clip(events[:, 0].astype(np.int64), 0, frame_width - 1)
    y = np.clip(events[:, 1].astype(np.int64), 0, frame_height - 1)
    np.add.at(density, (y, x), 1.0)
    density = ndimage.gaussian_filter(density, sigma=smoothing_sigma)

    rois: list[ROI] = []
    working = density.copy()
    half = roi_size // 2
    for _ in range(expected_eyes):
        peak_idx = np.unravel_index(np.argmax(working), working.shape)
        peak_val = working[peak_idx]
        if peak_val <= 0:
            break
        py, px = peak_idx
        rois.append(ROI(
            x=int(np.clip(px - half, 0, frame_width - roi_size)),
            y=int(np.clip(py - half, 0, frame_height - roi_size)),
            width=roi_size, height=roi_size,
            confidence=float(peak_val),
        ))
        # suppress the region around this peak so the next argmax finds a
        # different one (basic non-max suppression)
        y0, y1 = max(0, py - roi_size), min(frame_height, py + roi_size)
        x0, x1 = max(0, px - roi_size), min(frame_width, px + roi_size)
        working[y0:y1, x0:x1] = 0

    return rois
