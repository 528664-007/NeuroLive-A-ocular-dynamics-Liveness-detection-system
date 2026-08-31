"""Run a trained model over a dataloader and write runs/<name>/metrics.json.

This is intentionally the single place that produces metrics.json files, so
STATUS.md's claim ("only runs/*/metrics.json belongs in your report") stays
true. Every metrics.json is stamped with `is_synthetic` — if any sample in
the eval set was synthetic, the whole file is marked so downstream tooling
(and you) can't accidentally cite it as a real result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neurolive.eval.metrics import acer, per_attack_type_breakdown, segmentation_f1, top1_accuracy


@torch.no_grad()
def evaluate_joint_model(model, dataloader, device: str, out_path: str) -> dict:
    model.eval().to(device)

    all_seg_preds, all_seg_labels = [], []
    all_live_preds, all_live_labels, all_attack_types = [], [], []
    any_synthetic = False

    for batch in dataloader:
        voxel = batch["voxel"].to(device)
        seg_logits, liveness_logits = model(voxel)

        all_seg_preds.append(seg_logits.argmax(-1).cpu().numpy().ravel())
        all_seg_labels.append(batch["seg_labels"].numpy().ravel())
        all_live_preds.append(liveness_logits.argmax(-1).cpu().numpy())
        all_live_labels.append(batch["liveness_label"].numpy())
        all_attack_types.extend(batch["attack_type"])
        any_synthetic = any_synthetic or any(batch["is_synthetic"])

    seg_preds = np.concatenate(all_seg_preds)
    seg_labels = np.concatenate(all_seg_labels)
    live_preds = np.concatenate(all_live_preds)
    live_labels = np.concatenate(all_live_labels)

    results = {
        "is_synthetic": bool(any_synthetic),
        "n_samples": int(len(live_labels)),
        "liveness_accuracy": top1_accuracy(live_preds, live_labels),
        "liveness_acer": acer(live_preds, live_labels),
        "segmentation_f1_macro": segmentation_f1(seg_preds, seg_labels),
        "per_attack_type": per_attack_type_breakdown(live_preds, live_labels, all_attack_types),
    }
    if any_synthetic:
        results["WARNING"] = "Evaluation set includes synthetic data — these numbers are a pipeline smoke test, not a real result."

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    return results
