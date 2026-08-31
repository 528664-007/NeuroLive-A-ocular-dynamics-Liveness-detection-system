# NeuroLive — Event-Camera Liveness Detection

Final-year project extending **Mastropasqua et al., "Event-based Liveness Detection
using Temporal Ocular Dynamics: An Exploratory Approach"** (UBA/CONICET,
IEEE FG 2026, arXiv:2604.26285).

Baseline reported by the paper (SCNN on their RGBE-Gaze replay-attack extension):
**95.37% top-1 accuracy / 4.65% ACER**; TCN saccade segmentation **89.65% F1**;
peak-detection blink segmentation **95.35% F1**. These are *their* numbers,
reproduced here for reference only — see STATUS.md for what has and hasn't
been verified in *this* repo.

## Read STATUS.md before you read anything else

This repo was scaffolded in a sandbox with **no GPU and no access to the
actual dataset** (it's hosted on OneDrive, outside the sandbox's network
allowlist). Every number you see anywhere in this repo that isn't explicitly
marked `[SMOKE TEST, SYNTHETIC DATA]` is either (a) a number quoted from the
paper and cited as such, or (b) not yet measured. Nothing was invented to
make this look more finished than it is. STATUS.md has the honest phase-by-
phase breakdown.

## Your hardware (HP Victus 16-s0095AX)

Ryzen 7 7840HS, RTX 3050 Laptop GPU (**6 GB VRAM**), 16 GB RAM. This is
enough to train the Phase 1 TCN + SCNN baseline and the Phase 2 joint model
at the batch/sequence sizes in `configs/` without modification. Two things
to know going in:

- **No event camera.** This machine cannot capture live event streams. The
  dataset (recorded RGBE-Gaze + replay attacks) is fine for Phases 1–3.
  Phases 4–5 (event-native ROI, live real-time demo) are built to spec and
  unit-tested on recorded/synthetic streams, but "real-time on live hardware"
  claims can't be validated without a Prophesee (or similar) sensor.
- **`mamba-ssm`'s CUDA kernels can be finicky to compile** on Windows/WSL
  with a 6 GB card. `models/joint_mamba.py` auto-falls-back to a
  depthwise-separable Conv1D+GRU temporal backbone with the *same* I/O
  interface if `mamba_ssm` fails to import, so Phase 2 still runs either way
  — just note in your writeup which backbone you actually trained if you hit
  the fallback.

## Getting the data

The base RGBE-Gaze dataset (66 subjects, RGB+event+gaze, no attacks) is
public:
- Repo: https://github.com/GuangrongZhao/RGBE-Gaze
- Data (OneDrive): linked from that repo

**The replay-attack extension used by the Mastropasqua et al. paper does not
appear to have a public release as of this writing.** I could not find a
dataset link, DOI, or repo for it. You likely need to either (a) email the
authors to ask, or (b) collect your own small replay-attack extension
following their Section III protocol (display a genuine recording on a
screen, re-record with the event camera) — which itself requires event
camera access. Flagging this now rather than mid-Phase-3, per your own
ground rules.

Until you have real attack data, `scripts/generate_synthetic_data.py`
produces structurally-correct-but-fake event streams so you can develop and
smoke-test the full pipeline. Every artifact derived from it is labeled
`synthetic` in filenames and logs.

## Setup

**For a detailed, hardware-specific walkthrough (Windows/WSL2, RTX 3050,
CUDA install, troubleshooting table), see `SETUP_GUIDE.md`.** Quick version:

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .          # makes `neurolive` importable for the -m commands below
pytest tests/ -v          # smoke tests on synthetic data — should all pass

# then try the full pipeline on synthetic data end to end:
python -m neurolive.train.train_baseline --smoke-test --epochs 2
python -m neurolive.train.train_joint --smoke-test --epochs 2
```

If `mamba-ssm` fails to build (see hardware notes below), remove that line
from `requirements.txt` and rerun `pip install -r requirements.txt` — Phase
2 will use its Conv1D+GRU fallback automatically.

## Repo layout

```
src/neurolive/
  data/        event representations, dataset classes, synthetic generator
  models/      tcn.py, scnn.py (Phase 1 baseline), joint_mamba.py (Phase 2)
  eval/        accuracy / ACER / F1 metrics + evaluation harness
  train/       train_baseline.py (Phase 1), train_joint.py (Phase 2)
  attacks/     Phase 3 — replay / print / AI-injection attack generation
  localization/ Phase 4 — event-native face/eye ROI (no RGB dependency)
  utils/
demo/
  backend/     FastAPI challenge-response + decision-explanation API
  frontend/    React demo UI (challenge prompt, live signal viz)
scripts/       synthetic data generator, latency benchmark harness
tests/         pytest smoke tests (synthetic data only)
```

## Running Phase 1 (once you have real data)

```bash
python -m neurolive.train.train_baseline \
  --data-root /path/to/rgbe_gaze_liveness \
  --epochs 50 --batch-size 16 --device cuda
```

Writes checkpoints + a metrics JSON to `runs/phase1_baseline/`. Compare that
JSON against the paper's numbers in STATUS.md's table — do not hand-edit the
JSON to match.

## Latency benchmarking (Phase 5)

```bash
python scripts/latency_benchmark.py --model joint --device cuda --n-runs 100
```

Times the real event->voxel->model->decision path on this machine — genuine
wall-clock numbers, not fabricated. It is **not** camera-to-decision latency
on live hardware, since there's no event camera here to generate that first
leg; the script's own output says so in `SCOPE_NOTE`.

## Generating report tables (Phase 6)

```bash
python scripts/generate_report.py
```

Reads `runs/phase1_baseline/metrics.json` and `runs/phase2_joint/metrics.json`
and writes `report_assets/results_tables.md` (comparison table + per-attack-
type breakdown). If a run doesn't exist yet, the corresponding cells say
"not yet run" — never a filled-in number. If a run used synthetic data, its
cells say so instead of showing the number. Re-run this after every real
training run instead of copying numbers into your report by hand.

## AI-generated injection attacks (Phase 3) — scope note

The plan calls for using an existing open-source face-reenactment/face-swap
tool to *generate test attack samples only* — i.e. to synthesize the kind of
forged input a liveness system should reject, exactly as anti-spoofing
literature (CelebA-Spoof, OULU-NPU-style extensions, DeepFake presentation-
attack datasets) already does. `src/neurolive/attacks/ai_injection.py` is a
thin wrapper interface around such a tool with no bundled model weights or
generation code of its own — you plug in a tool you install separately. This
keeps attack-sample generation clearly separated from, and outside, your
actual contribution (the detector).
