"""
Webcam-to-simulated-events bridge.

This project's models expect real event-camera input (asynchronous
per-pixel brightness-change events, see data/event_repr.py). A laptop
webcam is a conventional frame camera — fundamentally different hardware,
not just a lower-quality substitute. There is no way to make webcam input
"actually" event data; the best honest option is to *simulate* events from
consecutive frames via thresholded differencing, which is what this module
does.

This is a real, standard technique (the same idea behind ESIM and similar
video-to-events simulators), but it is an approximation:
  - Temporal resolution is capped at the webcam's frame rate (typically
    30-60 fps), not the microsecond resolution real event cameras provide.
  - Noise and threshold-choice artifacts don't match a real sensor's
    characteristics.
A model trained on real event data (like this project's checkpoint) is
being run outside its training distribution when fed simulated events.
Label every output derived from this module as simulated, in code,
logs, and UI — do not let a demo audience assume this is real event-camera
footage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrameEventSimulatorConfig:
    threshold: float = 12.0          # 0-255 intensity-change threshold to register an event
    max_events_per_frame_pair: int = 4000  # cap for performance / voxelization sanity
    downscale_to: tuple[int, int] | None = (128, 128)  # (width, height); None = native resolution


class FrameEventSimulator:
    """Stateful: call `step(frame)` once per captured frame, in order.
    Returns a (N, 4) array of simulated [x, y, t, p] events for the gap
    since the previous frame (empty array on the first call, since there is
    no previous frame to diff against).

    `t` is wall-clock microseconds (time.perf_counter()-based), consistent
    with the units events_to_voxel_grid / events_to_time_surface expect.
    """

    def __init__(self, config: FrameEventSimulatorConfig | None = None):
        self.cfg = config or FrameEventSimulatorConfig()
        self._prev_gray: np.ndarray | None = None
        self._prev_t_us: float | None = None

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_t_us = None

    def step(self, frame_bgr: np.ndarray, t_us: float) -> np.ndarray:
        """frame_bgr: HxWx3 uint8 array from cv2.VideoCapture.read().
        t_us: capture timestamp in microseconds (monotonic clock).
        """
        import cv2

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.cfg.downscale_to is not None:
            gray = cv2.resize(gray, self.cfg.downscale_to, interpolation=cv2.INTER_AREA)
        gray = gray.astype(np.int16)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_t_us = t_us
            return np.zeros((0, 4), dtype=np.float32)

        diff = gray - self._prev_gray
        ys_pos, xs_pos = np.where(diff > self.cfg.threshold)
        ys_neg, xs_neg = np.where(diff < -self.cfg.threshold)

        n_pos, n_neg = len(xs_pos), len(xs_neg)
        total = n_pos + n_neg
        if total == 0:
            events = np.zeros((0, 4), dtype=np.float32)
        else:
            # spread events uniformly across the inter-frame interval so
            # the voxel grid's temporal binning has something to bin,
            # rather than collapsing every event onto one instant
            t0, t1 = self._prev_t_us, t_us
            xs = np.concatenate([xs_pos, xs_neg]).astype(np.float32)
            ys = np.concatenate([ys_pos, ys_neg]).astype(np.float32)
            ps = np.concatenate([np.ones(n_pos), -np.ones(n_neg)]).astype(np.float32)
            ts = np.sort(np.random.uniform(t0, t1, total)).astype(np.float32)
            events = np.stack([xs, ys, ts, ps], axis=1)

            if events.shape[0] > self.cfg.max_events_per_frame_pair:
                idx = np.random.choice(events.shape[0], self.cfg.max_events_per_frame_pair, replace=False)
                events = events[idx]
                events = events[np.argsort(events[:, 2])]

        self._prev_gray = gray
        self._prev_t_us = t_us
        return events
