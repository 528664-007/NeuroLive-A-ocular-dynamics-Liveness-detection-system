"""Metrics matching what the paper reports, so runs/*/metrics.json is
directly comparable to their numbers without unit conversion.

ACER (Average Classification Error Rate) = (APCER + BPCER) / 2, the
standard presentation-attack-detection metric (ISO/IEC 30107-3):
  APCER = attack presentations classified as genuine / total attack presentations
  BPCER = genuine presentations classified as attack / total genuine presentations
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def top1_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def acer(preds: np.ndarray, labels: np.ndarray, genuine_label: int = 1) -> dict:
    """preds, labels: 1D arrays, genuine_label marks the 'live' class."""
    attack_mask = labels != genuine_label
    genuine_mask = labels == genuine_label

    apcer = float((preds[attack_mask] == genuine_label).mean()) if attack_mask.any() else float("nan")
    bpcer = float((preds[genuine_mask] != genuine_label).mean()) if genuine_mask.any() else float("nan")
    acer_val = (apcer + bpcer) / 2 if not (np.isnan(apcer) or np.isnan(bpcer)) else float("nan")
    return {"apcer": apcer, "bpcer": bpcer, "acer": acer_val}


def segmentation_f1(preds: np.ndarray, labels: np.ndarray, average: str = "macro") -> float:
    """preds, labels: (N,) flattened per-frame class predictions."""
    return float(f1_score(labels, preds, average=average, zero_division=0))


def per_attack_type_breakdown(preds: np.ndarray, labels: np.ndarray, attack_types: list[str], genuine_label: int = 1) -> dict:
    """Phase 3 need: accuracy broken down per attack_type (replay/print/ai_injection)."""
    breakdown = {}
    types = sorted(set(attack_types))
    for t in types:
        mask = np.array([a == t for a in attack_types])
        if not mask.any():
            continue
        breakdown[t] = {
            "n": int(mask.sum()),
            "accuracy": top1_accuracy(preds[mask], labels[mask]),
        }
    return breakdown
