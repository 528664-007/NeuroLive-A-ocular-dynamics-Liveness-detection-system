"""End-to-end smoke tests on synthetic data. These prove the pipeline runs;
they say nothing about real-world accuracy — see STATUS.md.
"""
import json
from pathlib import Path

import numpy as np
import torch

from neurolive.data.dataset import SyntheticEventLivenessDataset, collate_liveness
from neurolive.data.event_repr import activity_profile, events_to_time_surface, events_to_voxel_grid
from neurolive.localization.event_native_roi import localize_eye_rois
from neurolive.models.joint_mamba import JointLivenessModel
from neurolive.models.losses import MultiTaskLivenessLoss
from neurolive.models.scnn import LivenessSCNN
from neurolive.models.tcn import OcularTCN


def _synthetic_events(n=500, h=64, w=64):
    rng = np.random.default_rng(0)
    x = rng.integers(0, w, n)
    y = rng.integers(0, h, n)
    t = np.sort(rng.uniform(0, 1e6, n))
    p = rng.choice([-1, 1], n)
    return np.stack([x, y, t, p], axis=1).astype(np.float32)


def test_voxel_grid_shape():
    events = _synthetic_events()
    voxel = events_to_voxel_grid(events, num_bins=16, height=64, width=64)
    assert voxel.shape == (16, 64, 64)
    assert torch.isfinite(voxel).all()


def test_voxel_grid_empty_events():
    voxel = events_to_voxel_grid(np.zeros((0, 4), dtype=np.float32), num_bins=8, height=32, width=32)
    assert voxel.shape == (8, 32, 32)
    assert (voxel == 0).all()


def test_time_surface_shape():
    events = _synthetic_events()
    surf = events_to_time_surface(events, height=64, width=64)
    assert surf.shape == (2, 64, 64)
    assert (surf >= 0).all() and (surf <= 1).all()


def test_activity_profile_sums_to_event_count():
    events = _synthetic_events(n=300)
    profile = activity_profile(events, num_bins=20)
    assert abs(profile.sum() - 300) < 1e-3


def test_event_native_roi_finds_density_peaks():
    # inject a dense cluster at a known location, check the ROI lands near it
    rng = np.random.default_rng(1)
    background = np.stack([
        rng.integers(0, 64, 50), rng.integers(0, 64, 50),
        np.sort(rng.uniform(0, 1e6, 50)), rng.choice([-1, 1], 50),
    ], axis=1).astype(np.float32)
    cluster_x, cluster_y = 15, 20
    cluster = np.stack([
        rng.integers(cluster_x - 2, cluster_x + 2, 200),
        rng.integers(cluster_y - 2, cluster_y + 2, 200),
        np.sort(rng.uniform(0, 1e6, 200)), rng.choice([-1, 1], 200),
    ], axis=1).astype(np.float32)
    events = np.concatenate([background, cluster])

    rois = localize_eye_rois(events, frame_height=64, frame_width=64, roi_size=16, expected_eyes=1)
    assert len(rois) == 1
    roi = rois[0]
    assert abs((roi.x + roi.width / 2) - cluster_x) < 16
    assert abs((roi.y + roi.height / 2) - cluster_y) < 16


def test_synthetic_dataset_and_collate():
    ds = SyntheticEventLivenessDataset(size=4, num_bins=8, height=32, width=32)
    batch = collate_liveness([ds[i] for i in range(4)])
    assert batch["voxel"].shape == (4, 8, 32, 32)
    assert all(batch["is_synthetic"])


def test_tcn_forward_shape():
    model = OcularTCN(num_bins=8, num_classes=3)
    voxel = torch.randn(2, 8, 32, 32)
    out = model(voxel)
    assert out.shape == (2, 8, 3)


def test_scnn_forward_shape():
    model = LivenessSCNN(num_classes=2)
    voxel = torch.rand(2, 6, 16, 16)  # small spatial size to keep the spiking loop fast
    out = model(voxel)
    assert out.shape == (2, 2)


def test_generate_report_handles_missing_runs(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/generate_report.py", "--runs-dir", str(tmp_path / "nonexistent"), "--out-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0
    assert "not yet run" in result.stdout
    assert (tmp_path / "out" / "results_tables.md").exists()


def test_latency_benchmark_runs(tmp_path):
    import subprocess
    import sys

    out_file = tmp_path / "latency.json"
    result = subprocess.run(
        [sys.executable, "scripts/latency_benchmark.py", "--n-runs", "3", "--n-events", "100",
         "--height", "16", "--width", "16", "--num-bins", "4", "--out", str(out_file)],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["n_runs"] == 3
    assert "SCOPE_NOTE" in data


def test_joint_model_forward_and_backward():
    model = JointLivenessModel(d_model=32, force_fallback=True)  # force fallback: no GPU/mamba in CI
    assert model.backbone_name == "conv_gru_fallback"
    voxel = torch.randn(2, 8, 16, 16)
    seg_logits, live_logits = model(voxel)
    assert seg_logits.shape == (2, 8, 3)
    assert live_logits.shape == (2, 2)

    loss_fn = MultiTaskLivenessLoss()
    seg_labels = torch.randint(0, 3, (2, 8))
    live_labels = torch.randint(0, 2, (2,))
    losses = loss_fn(seg_logits, live_logits, seg_labels, live_labels)
    losses["total"].backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and all(g >= 0 for g in grad_norms)
