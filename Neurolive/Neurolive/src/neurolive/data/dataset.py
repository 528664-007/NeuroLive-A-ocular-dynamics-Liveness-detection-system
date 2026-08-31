"""
Dataset classes.

`RGBELivenessDataset` expects real recordings on disk (see README for the
expected layout — it mirrors the original RGBE-Gaze release's per-subject
per-session structure, extended with a `label` and `attack_type` per clip).
It is NOT runnable until real data exists at `data_root`; it raises a clear
error rather than silently returning nothing.

`SyntheticEventLivenessDataset` fabricates structurally valid event clips
with the same __getitem__ contract, purely so the model/training code can be
exercised end-to-end without real data. Every sample is stamped
`is_synthetic=True` and nothing that touches this class should ever be
reported as a real result.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from neurolive.data.event_repr import events_to_voxel_grid


@dataclass
class LivenessSample:
    voxel: torch.Tensor          # (num_bins, H, W)
    seg_labels: torch.Tensor      # (num_bins,) per-frame ocular-event class (0=none,1=saccade,2=blink)
    liveness_label: torch.Tensor  # scalar: 1=genuine, 0=attack
    attack_type: str              # "genuine" | "replay" | "print" | "ai_injection"
    is_synthetic: bool


class RGBELivenessDataset(Dataset):
    """Expected layout under `data_root`:

        data_root/
          index.jsonl          # one JSON object per clip:
                                # {"events_path": "...", "label": 0|1,
                                #  "attack_type": "genuine|replay|print|ai_injection",
                                #  "seg_labels_path": "..."}
          <events files referenced above, .npy of shape (N, 4) = [x,y,t,p]>
          <seg_labels files, .npy of shape (num_bins,) int64>

    This mirrors (but does not bundle) the structure you'd build from the
    RGBE-Gaze release plus attack recordings. Raises FileNotFoundError with
    a clear message if data_root/index.jsonl doesn't exist, rather than
    returning an empty dataset silently.
    """

    def __init__(self, data_root: str, num_bins: int = 32, height: int = 480, width: int = 640):
        self.data_root = Path(data_root)
        index_path = self.data_root / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index.jsonl at {index_path}. This dataset class expects "
                "real recordings — see README.md 'Getting the data'. If you "
                "want to smoke-test the pipeline without real data, use "
                "SyntheticEventLivenessDataset instead."
            )
        self.num_bins, self.height, self.width = num_bins, height, width
        self.entries = [json.loads(l) for l in index_path.read_text().splitlines() if l.strip()]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> LivenessSample:
        entry = self.entries[idx]
        events = np.load(self.data_root / entry["events_path"])
        voxel = events_to_voxel_grid(events, self.num_bins, self.height, self.width)
        seg_labels = torch.from_numpy(np.load(self.data_root / entry["seg_labels_path"])).long()
        return LivenessSample(
            voxel=voxel,
            seg_labels=seg_labels,
            liveness_label=torch.tensor(entry["label"], dtype=torch.long),
            attack_type=entry["attack_type"],
            is_synthetic=False,
        )


class SyntheticEventLivenessDataset(Dataset):
    """Fabricated data for smoke-testing only. See module docstring."""

    ATTACK_TYPES = ("genuine", "replay", "print", "ai_injection")

    def __init__(
        self,
        size: int = 64,
        num_bins: int = 32,
        height: int = 64,
        width: int = 64,
        seed: Optional[int] = 0,
    ):
        self.size, self.num_bins, self.height, self.width = size, num_bins, height, width
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> LivenessSample:
        rng = np.random.default_rng((self.rng.integers(0, 2**31 - 1), idx))
        n_events = rng.integers(200, 2000)
        xs = rng.integers(0, self.width, n_events)
        ys = rng.integers(0, self.height, n_events)
        ts = np.sort(rng.uniform(0, 1_000_000, n_events))
        ps = rng.choice([-1, 1], n_events)
        events = np.stack([xs, ys, ts, ps], axis=1).astype(np.float32)

        voxel = events_to_voxel_grid(events, self.num_bins, self.height, self.width)
        seg_labels = torch.from_numpy(rng.integers(0, 3, self.num_bins)).long()
        attack_type = self.ATTACK_TYPES[idx % len(self.ATTACK_TYPES)]
        label = 1 if attack_type == "genuine" else 0

        return LivenessSample(
            voxel=voxel,
            seg_labels=seg_labels,
            liveness_label=torch.tensor(label, dtype=torch.long),
            attack_type=attack_type,
            is_synthetic=True,
        )


def collate_liveness(batch: list[LivenessSample]) -> dict:
    return {
        "voxel": torch.stack([b.voxel for b in batch]),
        "seg_labels": torch.stack([b.seg_labels for b in batch]),
        "liveness_label": torch.stack([b.liveness_label for b in batch]),
        "attack_type": [b.attack_type for b in batch],
        "is_synthetic": [b.is_synthetic for b in batch],
    }
