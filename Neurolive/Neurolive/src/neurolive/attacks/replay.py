"""Replay-attack event simulation for smoke-testing the attack-evaluation
pipeline before real replay recordings exist.

This does NOT reproduce the paper's actual replay-attack artifacts (display
resampling, refresh-rate flicker, etc.) — it's a structural stand-in so
Phase 3's evaluation code has something to run against. Real replay-attack
evaluation needs the real recordings (see README's dataset note).
"""
from __future__ import annotations

import numpy as np


def simulate_replay_degradation(events: np.ndarray, display_hz: float = 60.0) -> np.ndarray:
    """Coarsely mimics one real replay-attack artifact: temporal resampling
    to a display's refresh rate collapses fine-grained event timing into
    discrete frame boundaries. Snaps event timestamps to the nearest
    1/display_hz interval, which is directionally what a screen replay does
    to microsecond-precision saccade timing — but is not a substitute for a
    real recorded replay attack.
    """
    if events.shape[0] == 0:
        return events
    frame_period_us = 1_000_000.0 / display_hz
    out = events.copy()
    out[:, 2] = np.round(out[:, 2] / frame_period_us) * frame_period_us
    return out
