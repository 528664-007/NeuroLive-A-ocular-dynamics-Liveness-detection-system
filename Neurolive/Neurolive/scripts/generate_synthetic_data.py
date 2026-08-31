"""Writes a small synthetic dataset to disk in the exact layout
RGBELivenessDataset expects (index.jsonl + .npy files), so you can test the
*real* data-loading path (not just SyntheticEventLivenessDataset in memory)
before you have actual recordings.

Usage: python scripts/generate_synthetic_data.py --out data/synthetic_smoke --n 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ATTACK_TYPES = ["genuine", "replay", "print", "ai_injection"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="data/synthetic_smoke")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--num-bins", type=int, default=16)
    args = p.parse_args()

    out_dir = Path(args.out)
    events_dir, labels_dir = out_dir / "events", out_dir / "seg_labels"
    events_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    entries = []
    for i in range(args.n):
        n_events = rng.integers(200, 1500)
        events = np.stack([
            rng.integers(0, 64, n_events),
            rng.integers(0, 64, n_events),
            np.sort(rng.uniform(0, 1e6, n_events)),
            rng.choice([-1, 1], n_events),
        ], axis=1).astype(np.float32)
        seg_labels = rng.integers(0, 3, args.num_bins).astype(np.int64)

        events_path = f"events/clip_{i:04d}.npy"
        labels_path = f"seg_labels/clip_{i:04d}.npy"
        np.save(out_dir / events_path, events)
        np.save(out_dir / labels_path, seg_labels)

        attack_type = ATTACK_TYPES[i % len(ATTACK_TYPES)]
        entries.append({
            "events_path": events_path,
            "seg_labels_path": labels_path,
            "label": 1 if attack_type == "genuine" else 0,
            "attack_type": attack_type,
        })

    (out_dir / "index.jsonl").write_text("\n".join(json.dumps(e) for e in entries))
    print(f"Wrote {args.n} synthetic clips to {out_dir}/ (SYNTHETIC — not real data, see README)")


if __name__ == "__main__":
    main()
