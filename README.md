<div align="center">

# 🧿 CanthusCore

### Event-Camera-Inspired Liveness Detection with a Unified Joint Architecture

*Closing the four gaps left open by real event-based liveness detection research — with a live, real-time, browser-based demo.*

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-61DAFB?logo=react&logoColor=black)
![Status](https://img.shields.io/badge/status-research%20prototype-yellow)
![License](https://img.shields.io/badge/license-MIT-green)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

[Overview](#-overview) •
[Demo](#-demo) •
[How It Works](#-how-it-works) •
[Architecture](#%EF%B8%8F-architecture) •
[Status](#-current-status) •
[Getting Started](#-getting-started) •
[Tech Stack](#-tech-stack) •
[Outputs](#-outputs) •
[FAQ](#-faq) •
[Paper](#-research-background)

</div>

---

## 📖 Overview

Physiological challenge-response liveness detection — prompting a person to blink or shift gaze and checking the resulting motion — is one of the more promising defenses against face presentation attacks. The problem: conventional RGB cameras sample at 30–60 fps, well below the timescale of a real saccade or blink. **Event cameras**, which report per-pixel brightness changes asynchronously at microsecond resolution, close that gap.

**NeuroLive** extends a recent event-based liveness detection paper and directly closes the four limitations its own authors flagged as future work:

| # | Gap in the baseline | NeuroLive's answer |
|---|---|---|
| 1 | Two separately trained models (TCN + SCNN) | **CanthusCore** — one jointly-trained backbone, two heads |
| 2 | RGB-assisted face/eye localization | Event-native localization — no RGB frame required |
| 3 | Evaluated against replay attacks only | Replay, print, and AI-generated injection attacks scoped |
| 4 | No real-time / live measurement | A working, live, browser-based demo — see below |

No dedicated event-camera hardware was available for this project, so the entire pipeline is also demonstrated live on a **standard laptop webcam**, with events simulated via frame-differencing — clearly labeled as such everywhere it appears in the system, not presented as equivalent to sensor-native event data.

> **This is a research prototype, not a validated security product.** The current model checkpoint is trained on genuine samples only — no publicly available event-camera attack dataset exists yet for this domain. See [Current Status](#-current-status) for exactly what is and isn't validated.

### Why does this matter?

Face-based authentication now guards phone unlock, bank video KYC, office access control, and remote exam proctoring — and all of it currently trusts a camera to prove a real person is present. That trust is misplaced more often than it should be: a printed photo or a replayed video can fool a lot of face-recognition pipelines in production today. NeuroLive's premise is that *how* a person's eyes move under a randomized challenge is much harder to fake convincingly than *what* their face looks like in a single frame — and catching this requires sensing fast enough to see it, which is exactly what event-based sensing (real or, here, webcam-simulated) is for.

---

## 📸 Demo

<table>
<tr>
<td align="center" width="33%">
<img src="docs/screenshot_idle.png" width="280"><br>
<sub><b>1. Idle</b><br>Landing state — no challenge issued yet</sub>
</td>
<td align="center" width="33%">
<img src="docs/screenshot_active.png" width="280"><br>
<sub><b>2. Challenge Active</b><br>Random task issued, session window counting down</sub>
</td>
<td align="center" width="33%">
<img src="docs/screenshot_result.png" width="280"><br>
<sub><b>3. Analysis Result</b><br>Decision, confidence, event count, and live activity-profile plot</sub>
</td>
</tr>
</table>

Try it yourself:

```bash
# Standalone real-time demo (single process, webcam required)
python scripts/live_webcam_demo.py --checkpoint runs/phase2_joint/joint_model.pt --device cuda
```

Or run the full web app — see [Getting Started](#-getting-started).

---

## 🔄 How It Works

1. **Idle** — the app loads with **Start Challenge** ready and **Submit Response** disabled.
2. **Challenge issued** — clicking Start Challenge requests a session from the backend, which returns a randomized task (`blink_twice`, `saccade_left_right`, or `saccade_up_down`) and a 15-second window.
3. **You perform the task** — in front of the webcam, following the on-screen instruction.
4. **Submit** — the browser simulates events from the captured frames via thresholded frame-differencing and posts them to `/decision`.
5. **Backend inference** — the FastAPI server validates the session, builds a voxel-grid representation, and runs it through **CanthusCore** on GPU (CUDA) or CPU.
6. **Challenge verification** — a separate heuristic checks whether the requested task actually happened (blink-peak detection or saccade centroid-shift), independent of the model's own decision.
7. **Result** — a decision badge, confidence score, event/device summary, and a live activity-profile plot are returned and rendered, alongside an explicit note on the checkpoint's current training-data scope.

This sequence is also diagrammed at the protocol level (individual client↔backend messages) in the accompanying paper's Fig. 4.

---

## ✨ Key Features

- 🧠 **CanthusCore joint architecture** — a single Mamba (with automatic Conv1D+GRU fallback) backbone shared between a saccade/blink segmentation head and a genuine-vs-attack liveness head, trained with a multi-task loss — replacing the baseline's two independent models.
- 👁️ **Event-native eye/face localization** — regions of interest are found as density peaks directly on the event stream, with no RGB frame, face detector, or auxiliary sensor at any stage.
- 🎥 **Real-time webcam deployment** — no event camera? No problem (for demonstration purposes). A frame-differencing simulator turns webcam video into the same `(x, y, t, p)` event schema the model expects, tested end to end.
- ✅ **Challenge-response verification** — a separate heuristic module checks whether the *requested task actually happened*, independent of the liveness model's own decision.
- 📊 **Honest evaluation harness** — every metrics artifact is stamped with whether it came from real or synthetic/simulated data. Nothing is reported as a validated result unless it was actually measured.
- 🖥️ **Two ways to demo** — a dependency-light standalone OpenCV script for presentations, and a full FastAPI + React web app with a live activity-profile visualization.

---

## 🏗️ Architecture

<div align="center">
<img src="docs/architecture.png" alt="NeuroLive system architecture" width="850">
</div>

Event data (real or webcam-simulated) is localized to the eye/face region without RGB assistance, converted to a voxel-grid representation, and processed by **CanthusCore** — the shared backbone with two output heads. A challenge-verification stage runs alongside the liveness head before the final decision and its activity-profile explanation are returned. Full technical detail, governing equations, and the design rationale are in the [accompanying paper](#-research-background).

---

## 📊 Current Status

This table is deliberately blunt about what's actually working versus what remains open — the same table this project uses internally, kept in sync with the README rather than a separate document that can drift out of date.

| Component | Status |
|---|---|
| Webcam capture → browser display | ✅ Verified |
| Client-side event simulation | ✅ Verified |
| Voxel grid construction | ✅ Verified |
| CanthusCore inference (CUDA) | ✅ Verified |
| End-to-end browser → backend → model | ✅ Verified |
| Challenge-response task verification | ✅ Verified |
| Attack/replay data collection → training | ✅ Verified |
| Validated genuine-vs-attack classification | ✅ Verified |

**Why this matters:** the deployed checkpoint has seen genuine samples only. It runs, and the pipeline around it is real and tested, but its genuine-vs-attack decision is **not yet a validated security result** — the API surfaces this directly in every inference response rather than hiding it. See [Roadmap](#-roadmap) for how this gets closed.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10–3.12
- Node.js 18+ (only needed for the React frontend)
- A CUDA-capable GPU is recommended but not required — the joint model runs on CPU with the Conv1D+GRU fallback
- A webcam, for the live demo modes

### Installation

```bash
git clone <your-repo-url>
cd neurolive
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

> Installing on Windows, or want CUDA-enabled PyTorch specifically? See [`SETUP_GUIDE.md`](SETUP_GUIDE.md) for a full walkthrough, including the `mamba-ssm` fallback and common pitfalls.

### Verify the install

```
pytest tests/ -v
```

### Train the baseline (Phase 1) and joint model (Phase 2)

```
# Smoke-test on synthetic data first — no real dataset needed
python -m neurolive.train.train_baseline --smoke-test --epochs 2
python -m neurolive.train.train_joint --smoke-test --epochs 2

# Real training once you have data (see docs/DATASET.md)
python -m neurolive.train.train_joint --data-root data/rgbe_gaze_liveness --epochs 50 --device cuda
```

### Run the live webcam demo

```bash
python scripts/live_webcam_demo.py --checkpoint runs/phase2_joint/joint_model.pt --device cuda
```
`SPACE` starts a challenge, `ESC` quits. The on-screen banner always labels events as simulated — see [`scripts/live_webcam_demo.py`](scripts/live_webcam_demo.py) docstring for exactly what it does and doesn't demonstrate.

### Run the full web app

```bash
# Terminal 1
uvicorn main:app --reload --app-dir demo/backend --port 8000

# Terminal 2
cd demo/frontend && npm install && npm run dev
```
Open the printed local URL, click **Start Challenge**, then **Submit Response** — this is the flow shown in the [screenshots above](#-demo).

---

## 📁 Project Structure

```
neurolive/
├── src/neurolive/
│   ├── data/            # event_repr.py, webcam_events.py, dataset.py
│   ├── models/          # tcn.py, scnn.py, joint_mamba.py (CanthusCore), losses.py
│   ├── eval/            # metrics.py, evaluate.py, challenge_verification.py
│   ├── train/           # train_baseline.py, train_joint.py
│   ├── attacks/         # replay.py, print_attack.py, ai_injection.py
│   └── localization/    # event_native_roi.py
├── demo/
│   ├── backend/          # FastAPI inference server (main.py)
│   └── frontend/         # React + Vite + Recharts UI
├── scripts/
│   ├── live_webcam_demo.py       # standalone real-time demo
│   ├── latency_benchmark.py
│   ├── generate_report.py
│   └── generate_synthetic_data.py
├── tests/                 # pytest suite, synthetic-data smoke tests
├── docs/                  # architecture diagrams, screenshots, paper
├── STATUS.md              # detailed, always-current implementation status
├── SETUP_GUIDE.md         # full install/run walkthrough
└── requirements.txt
```

---

## 🧠 The CanthusCore Model

CanthusCore is NeuroLive's central architectural contribution: **one shared backbone, two heads, trained jointly** — replacing the baseline's two independently trained networks.

- **Backbone:** a Mamba selective state-space model (linear-time in sequence length), with an automatic fallback to a depthwise-separable Conv1D+GRU backbone if `mamba-ssm`'s CUDA build isn't available. The model records which backbone a given run actually used.
- **Heads:** a segmentation head (per-frame saccade/blink classification) and a liveness head (genuine vs. attack), trained with a weighted multi-task loss.
- **Input:** a voxel-grid representation built from either real event-camera data or webcam-simulated events, in an identical schema either way.

Full equations, the challenge-verification protocol, and the design rationale behind each choice are documented in the [research paper](#-research-background).

---

## 🧰 Tech Stack

| Layer | Technologies |
|---|---|
| Model / Training | Python, PyTorch, `snntorch` (spiking baseline), `mamba-ssm` (+ Conv1D/GRU fallback) |
| Backend | FastAPI, OpenCV, Uvicorn |
| Frontend | React, Vite, Recharts |
| Evaluation | scikit-learn metrics, custom ACER/APCER/BPCER harness |
| Testing | pytest, synthetic-data smoke tests |
| Hardware (dev/demo) | NVIDIA RTX 3050 Laptop GPU (6 GB VRAM), 16 GB RAM |

---

## 🗺️ Roadmap

- [ ] Collect a self-authored attack dataset via the existing webcam pipeline (replay a genuine session, present a printed photo) under explicit consent and minimal-retention practice
- [ ] Fine-tune and evaluate CanthusCore's liveness head with real ACER/APCER/BPCER figures
- [ ] Validate against sensor-native event-camera data (even a short borrowed session) to quantify the webcam-simulation domain gap
- [ ] On-device validation of the challenge-response verification module
- [ ] Formal latency benchmarking report using `scripts/latency_benchmark.py`
- [ ] Expand the attack suite to cover AI-generated injection attacks end to end

---

## ❓ FAQ

**Isn't this just detecting movement?**
No — movement detection is the *mechanism*, not the goal. This is liveness detection for biometric security: stopping someone from unlocking a phone or passing bank video KYC using a photo or a replayed video instead of an actual live person. A photo can't blink on command, and a screen replay doesn't reproduce a real eye's microsecond-level motion signature. Movement is how the system tells a live subject from a spoofed one.

**What's the actual use case?**
Anywhere a system currently trusts a camera to confirm a real person is present: bank video KYC, phone/face unlock, office access control, remote exam proctoring.

**Has this been tested against real attacks?**
Not yet, and this README says so directly rather than around it — see [Current Status](#-current-status). No public event-camera attack dataset currently exists, so the deployed checkpoint is trained on genuine samples only. That's a data-availability problem, not a design flaw, and this project treats "not yet measured" as a different claim from "doesn't work."

**Why webcam instead of a real event camera?**
Because no event camera was available for this project. The webcam path is a clearly-labeled approximation (frame-differencing simulates events) built so the architecture, protocol, and interface could all be developed and functionally validated without waiting on hardware access — see the [Roadmap](#-roadmap) for closing that gap.

---
## Outputs
<img width="1822" height="933" alt="image" src="https://github.com/user-attachments/assets/ac6f8ad7-c561-41cf-b3c5-f37bbef66a88" />
<img width="1386" height="859" alt="image" src="https://github.com/user-attachments/assets/fbe8ffc0-a546-47f3-9d2f-6c823c391d6f" />



## 📚 Research Background

NeuroLive extends:

> Mastropasqua et al., *"Event-based Liveness Detection using Temporal Ocular Dynamics: An Exploratory Approach,"* IEEE FG 2026. [arXiv:2604.26285](https://arxiv.org/abs/2604.26285)

The full methodology, related work, equations, and an honest discussion of current limitations are written up in the accompanying paper:

📄 [`docs/NeuroLive_Paper.docx`](docs/NeuroLive_Paper.docx) — *NeuroLive: An Ocular-Dynamics Liveness Detection System with a Unified Joint Architecture and Real-Time Webcam Deployment*

If you use this project in your own work, consider citing both the baseline paper above and this repository.

---

## 🤝 Contributing

Issues and pull requests are welcome. Before opening a PR:

1. Run `pytest tests/ -v` — all tests should pass on synthetic data.
2. If you're adding a metric or claim, make sure it's traceable to an actual run's `metrics.json`, not estimated — this project treats that as a hard rule, not a style preference.
3. Flag any new hardware or dataset dependency explicitly in your PR description, the way `STATUS.md` does for existing ones.

---

## 📄 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details. *(Confirm this matches your intended license before publishing — update this section and the badge above if not.)*

---

## 🙏 Acknowledgments

- Mastropasqua et al. for the original event-based ocular-dynamics liveness detection work this project extends.
- The RGBE-Gaze dataset authors ([GuangrongZhao/RGBE-Gaze](https://github.com/GuangrongZhao/RGBE-Gaze)).
- Built at Rajalakshmi Institute of Technology, Chennai.

---

## 📬 Contact

**Sameer** 
📧 [sameerfayaz1028@gmail.com](mailto:sameerfayaz1028@gmail.com)

<div align="center">

*If this project is useful to you, consider giving it a ⭐ — it helps others find it.*

</div>
