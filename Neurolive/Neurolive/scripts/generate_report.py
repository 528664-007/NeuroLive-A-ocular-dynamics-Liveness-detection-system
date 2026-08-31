"""
Phase 6 — writeup support: turns runs/*/metrics.json into the comparison
table, ablation table, and per-attack-type breakdown you need for the
report/defense. Deliberately does NOT invent placeholder numbers when a run
is missing — it tells you what's missing instead, so you can't accidentally
paste a table with silently-fabricated cells into your report.

Usage:
    python scripts/generate_report.py
    python scripts/generate_report.py --out-dir report_assets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PAPER_BASELINE = {
    "source": "Mastropasqua et al., IEEE FG 2026 (arXiv:2604.26285) — their numbers, not this repo's",
    "scnn_liveness_accuracy": 0.9537,
    "scnn_liveness_acer": 0.0465,
    "tcn_segmentation_f1_saccade": 0.8965,
    "blink_peak_detection_f1": 0.9535,
}


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def build_comparison_table(phase1: dict | None, phase2: dict | None) -> str:
    lines = [
        "| Metric | Paper baseline (📄 quoted) | Phase 1 (this repo) | Phase 2 joint (this repo) |",
        "|---|---|---|---|",
    ]

    def cell(run: dict | None, key: str, is_synthetic_check=True) -> str:
        if run is None:
            return "not yet run"
        if is_synthetic_check and run.get("is_synthetic"):
            return "SYNTHETIC — not a real result"
        val = run.get(key)
        return f"{val:.4f}" if isinstance(val, float) else str(val)

    p1_acc = cell(phase1, "scnn_liveness_accuracy")
    p2_acc = cell(phase2, "liveness_accuracy")
    lines.append(f"| Liveness accuracy | {PAPER_BASELINE['scnn_liveness_accuracy']:.4f} | {p1_acc} | {p2_acc} |")

    p1_acer = "not yet run" if phase1 is None else (
        "SYNTHETIC — not a real result" if phase1.get("is_synthetic")
        else f"{phase1.get('scnn_liveness_acer', {}).get('acer', 'n/a')}"
    )
    p2_acer = "not yet run" if phase2 is None else (
        "SYNTHETIC — not a real result" if phase2.get("is_synthetic")
        else f"{phase2.get('liveness_acer', {}).get('acer', 'n/a')}"
    )
    lines.append(f"| ACER | {PAPER_BASELINE['scnn_liveness_acer']:.4f} | {p1_acer} | {p2_acer} |")

    p1_f1 = cell(phase1, "tcn_segmentation_f1_macro")
    p2_f1 = cell(phase2, "segmentation_f1_macro")
    lines.append(f"| Segmentation F1 (macro) | {PAPER_BASELINE['tcn_segmentation_f1_saccade']:.4f} (saccade only) | {p1_f1} | {p2_f1} |")

    if phase2 and "backbone" in phase2:
        lines.append("")
        lines.append(f"Phase 2 backbone used: `{phase2['backbone']}`")

    return "\n".join(lines)


def build_attack_breakdown_table(phase2: dict | None) -> str:
    if phase2 is None or "per_attack_type" not in phase2:
        return "No per-attack-type breakdown available yet — run Phase 2/3 evaluation first."
    if phase2.get("is_synthetic"):
        header = "**SYNTHETIC DATA — pipeline smoke test only, not a real per-attack-type result.**\n\n"
    else:
        header = ""
    lines = [header + "| Attack type | n | Accuracy |", "|---|---|---|"]
    for atype, stats in phase2["per_attack_type"].items():
        lines.append(f"| {atype} | {stats['n']} | {stats['accuracy']:.4f} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--out-dir", type=str, default="report_assets")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    phase1 = _load(runs_dir / "phase1_baseline" / "metrics.json")
    phase2 = _load(runs_dir / "phase2_joint" / "metrics.json")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison_table(phase1, phase2)
    breakdown = build_attack_breakdown_table(phase2)

    report = (
        "# NeuroLive — results tables (auto-generated, do not hand-edit)\n\n"
        "## Baseline vs. joint model comparison\n\n" + comparison + "\n\n"
        "## Per-attack-type breakdown (Phase 2/3)\n\n" + breakdown + "\n\n"
        "---\n"
        "Regenerate with `python scripts/generate_report.py` after any new "
        "training run. Cells reading 'not yet run' or 'SYNTHETIC' are exactly "
        "that — do not fill them in by hand.\n"
    )
    (out_dir / "results_tables.md").write_text(report, encoding="utf-8")
    print(report.encode("ascii", errors="replace").decode("ascii"))
    print(f"\nWritten to {out_dir}/results_tables.md")

    if phase2 and phase2.get("activity_profile_example"):
        _plot_activity_profile(phase2["activity_profile_example"], out_dir / "activity_profile_example.png")


def _plot_activity_profile(profile: list, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(profile)
    ax.set_xlabel("time bin")
    ax.set_ylabel("event count")
    ax.set_title("Activity profile (decision explanation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
