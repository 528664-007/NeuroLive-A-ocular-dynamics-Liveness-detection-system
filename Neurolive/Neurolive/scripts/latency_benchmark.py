"""
Phase 5 — latency benchmark harness.

Measures real wall-clock latency for the parts of the pipeline that exist
on this hardware: event-stream -> voxel grid -> model forward pass ->
decision. This is genuine, reproducible timing data — not fabricated — but
it is NOT the same thing the paper's own Section on future work asks for
("real-time/live latency measurement"), because that implies camera-to-
decision latency on live hardware, and there's no event camera here to
generate that first leg. Read the printed report's header before citing a
number from this script anywhere.

Usage:
    python scripts/latency_benchmark.py --model joint --device cpu --n-runs 100
    python scripts/latency_benchmark.py --model joint --checkpoint runs/phase2_joint/joint_model.pt --device cuda
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from neurolive.data.event_repr import events_to_voxel_grid
from neurolive.models.joint_mamba import JointLivenessModel
from neurolive.models.scnn import LivenessSCNN
from neurolive.models.tcn import OcularTCN


def _fake_event_batch(n_events: int, height: int, width: int) -> np.ndarray:
    """Structurally realistic event volume for one challenge window — NOT a
    real captured stream. Used only so the voxelization step gets timed on
    a realistic event count, since that step's cost scales with N events,
    not with whether the events are real.
    """
    rng = np.random.default_rng()
    x = rng.integers(0, width, n_events)
    y = rng.integers(0, height, n_events)
    t = np.sort(rng.uniform(0, 300_000, n_events))  # ~300ms challenge window
    p = rng.choice([-1, 1], n_events)
    return np.stack([x, y, t, p], axis=1).astype(np.float32)


def build_model(name: str, num_bins: int, checkpoint: str | None, device: str):
    if name == "joint":
        model = JointLivenessModel(force_fallback=(device == "cpu"))
    elif name == "tcn_scnn":
        # timed together since Phase 1 needs both for a decision
        tcn, scnn = OcularTCN(num_bins=num_bins), LivenessSCNN()
        if checkpoint:
            raise ValueError("--checkpoint isn't supported for tcn_scnn (two files needed) — load manually if needed.")
        return (tcn.to(device).eval(), scnn.to(device).eval())
    else:
        raise ValueError(f"unknown --model {name}")

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def benchmark(model, model_name: str, args) -> dict:
    latencies_ms = []
    for i in range(args.n_runs):
        events = _fake_event_batch(args.n_events, args.height, args.width)

        t0 = time.perf_counter()
        voxel = events_to_voxel_grid(events, args.num_bins, args.height, args.width).unsqueeze(0).to(args.device)
        if model_name == "joint":
            model(voxel)
        else:
            tcn, scnn = model
            tcn(voxel)
            scnn(voxel)
        if args.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000)

    return {
        "n_runs": args.n_runs,
        "n_events_per_run": args.n_events,
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": sorted(latencies_ms)[int(0.95 * len(latencies_ms)) - 1],
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["joint", "tcn_scnn"], default="joint")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-runs", type=int, default=50)
    p.add_argument("--n-events", type=int, default=1500, help="events per challenge window, tune to your expected saccade/blink event rate")
    p.add_argument("--num-bins", type=int, default=32)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--out", type=str, default="runs/latency_benchmark.json")
    args = p.parse_args()

    model = build_model(args.model, args.num_bins, args.checkpoint, args.device)
    results = benchmark(model, args.model, args)
    results["model"] = args.model
    results["device"] = args.device
    results["checkpoint_loaded"] = bool(args.checkpoint)
    results["SCOPE_NOTE"] = (
        "This times event->voxel->model->decision on synthetic event batches on "
        "this machine. It is NOT camera-to-decision latency on live hardware — "
        "no event camera exists on this setup. See this script's module docstring."
    )

    print(json.dumps(results, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
