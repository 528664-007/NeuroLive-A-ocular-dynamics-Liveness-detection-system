# Status — read this before citing any number from this repo

Legend: ✅ implemented + smoke-tested (synthetic data) · 🧩 scaffolded, not yet
runnable end-to-end · ⛔ blocked on external dependency (data or hardware) ·
📄 number is quoted from the source paper, not produced by this repo.

| Phase | What | Code status | Real numbers? |
|---|---|---|---|
| 1 | TCN + SCNN baseline replication | ✅ trains/evals on synthetic data | ⛔ needs real RGBE-Gaze + attack extension |
| 2 | Mamba joint model (or Conv1D+GRU fallback) | ✅ forward/backward pass verified | ⛔ same data blocker; also needs Phase 1 numbers to ablate against |
| 3 | Print + AI-injection attack suite | 🧩 interfaces + replay attack only | ⛔ needs external face-reenactment tool (not bundled) + Phase 2 model |
| 4 | Event-native ROI (no RGB) | 🧩 heuristic v0 (event-density clustering), unit-tested on synthetic streams only | ⛔ unvalidated against real event data — accuracy unknown |
| 5 | Real-time demo + latency bench | ✅ FastAPI/React demo tested; `latency_benchmark.py` tested and produces real wall-clock numbers | ⛔ those numbers are inference-only (voxel->model), not camera-to-decision — no event camera on your hardware |
| 6 | Writeup tables/plots | ✅ `generate_report.py` tested in both empty and populated states | ⛔ populates automatically once 1–2 produce real (non-synthetic) metrics.json files — currently nothing real to plot |

**Baseline numbers cited anywhere in this repo (95.37% accuracy, 4.65% ACER,
89.65% F1, 95.35% F1) are 📄 from the Mastropasqua et al. paper**, reproduced
for comparison context only. They are not this repo's output. Do not let
these end up in your report's "our results" table — only your own
`runs/*/metrics.json` belongs there, and only once it exists.

## What would unblock each ⛔

- **Phases 1–3**: the real attack-extension dataset. Either the authors
  release it, or you collect a small one yourself (needs event camera
  access — check if your CV lab or IIT-M has a Prophesee/DVS sensor before
  assuming you need to buy one).
- **Phases 4–5**: any event camera for validation. Even a short borrowed
  session is enough to sanity-check Phase 4's ROI heuristic and get one real
  latency number for Phase 5.
- **Phase 6**: falls out automatically once 1–2 have real metrics.json files.

## Next concrete step

Run `pytest tests/ -v` locally to confirm the smoke tests pass on your
machine too, then start on getting real data (email the paper's authors is
probably the fastest path for the attack extension). Everything downstream
depends on that more than on any more code being written.
