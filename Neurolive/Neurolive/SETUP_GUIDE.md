# NeuroLive — Step-by-Step Setup & Run Guide

Written for your machine: HP Victus 16-s0095AX (Windows 11, Ryzen 7 7840HS,
RTX 3050 6GB, 16GB RAM). Follow this in order — each step assumes the
previous one worked. Where a step can silently "sort of" work, I've said
what to check before moving on.

---

## 0. Decide: native Windows or WSL2?

Two of this project's dependencies (`mamba-ssm`'s CUDA kernels, and general
ML tooling) behave meaningfully better on Linux. You have two real options:

- **Native Windows** — simpler to start, but Phase 2's Mamba backbone will
  almost certainly fail to build and fall back to the Conv1D+GRU backbone
  (the code handles this automatically — nothing breaks, you just won't be
  running the actual Mamba architecture named in your project plan).
- **WSL2 (Ubuntu)** — more setup up front, but `mamba-ssm` has a real chance
  of building, and it's what "production-quality, reproducible" ML projects
  actually run on in practice.

**Recommendation: use WSL2** if you want the real Mamba backbone for your
ablation table. If you're fine citing the fallback backbone (it's a
legitimate architecture choice, just not the one named in your plan), native
Windows is less setup. Steps below cover both — skip whichever you don't
need.

### If you're going WSL2:
```powershell
wsl --install -d Ubuntu-22.04
```
Restart when prompted, set up your Ubuntu username/password, then do
everything below **inside that Ubuntu terminal**, not PowerShell. Your GPU
is visible inside WSL2 automatically on Windows 11 — no extra driver step
needed, just make sure your NVIDIA driver on the Windows side is current
(GeForce Experience or nvidia.com, not the WSL side).

---

## 1. Install prerequisites

**Native Windows:**
1. Python 3.10–3.12 from python.org (check "Add to PATH" during install).
2. Git from git-scm.com (or use GitHub Desktop if you prefer a GUI).
3. Node.js 18+ from nodejs.org (needed for the demo frontend only — skip
   if you're not getting to Phase 5 yet).
4. NVIDIA driver: open GeForce Experience → check for driver updates, or
   download directly from nvidia.com for your RTX 3050.

**WSL2 (Ubuntu):**
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip build-essential
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
```

Verify Python:
```
python --version    # Windows
python3 --version   # WSL2
```
Should print 3.10, 3.11, or 3.12.

---

## 2. Unzip the project and open a terminal there

Extract `NeuroLive.zip` somewhere simple — avoid deeply nested folders or
paths with spaces (e.g. `C:\dev\neurolive`, not
`C:\Users\you\Documents\My Projects\...`). Then:

```powershell
cd C:\dev\neurolive        # Windows
```
```bash
cd ~/neurolive             # WSL2 — if you extracted on the Windows side,
                            # your C: drive is at /mnt/c/... from WSL2
```

---

## 3. Create and activate a virtual environment

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks the activation script with an execution-policy error,
run this once (as your normal user, not admin) and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

```bash
# WSL2 / Linux
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` at the start of your prompt from here on. Every
command below assumes it's active.

---

## 4. Install PyTorch with CUDA support (do this before requirements.txt)

Generic `pip install torch` gives you a CPU-only build, which defeats the
point of having a 3050. Go to **pytorch.org/get-started/locally** and use
the exact command it generates for your setup (Stable, your OS, Pip, your
CUDA version) — the specific CUDA build number changes over time, so
copy-paste from there rather than a version I hardcode here. It'll look
like:

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
(`cu121` etc. — use whatever the site gives you.)

**Verify the GPU is actually visible before going further:**
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU found')"
```
This must print `True` and `NVIDIA GeForce RTX 3050 Laptop GPU` (or similar)
before you continue. If it prints `False`:
- Windows native: reinstall the NVIDIA driver, reboot, retry.
- WSL2: confirm `nvidia-smi` works inside WSL2 first — if it doesn't, your
  Windows-side driver needs updating (WSL2 GPU passthrough rides on it).

---

## 5. Install the rest of the dependencies

```bash
pip install -r requirements.txt
```

If this fails specifically on `mamba-ssm` (native Windows: it likely will;
WSL2: it might, depending on your CUDA toolkit version), that's expected
and handled — see the next step. Everything else should install cleanly.

**If mamba-ssm fails:**
```bash
pip install -r requirements.txt --no-deps  # then retry without it:
```
Open `requirements.txt`, comment out or delete the `mamba-ssm` line, save,
and rerun `pip install -r requirements.txt`. Phase 2's model will use its
Conv1D+GRU fallback automatically — you don't need to change any code. You
can confirm which backbone you're actually running any time by checking
`model.backbone_name` (the training script prints it).

If you want to actually try to get `mamba-ssm` working (WSL2 only, don't
bother on native Windows):
```bash
sudo apt install -y nvidia-cuda-toolkit   # if not already present
pip install mamba-ssm --no-build-isolation
```
This compiles CUDA kernels and can take 10–20 minutes. If it fails, don't
sink more time into it — the fallback is a legitimate architecture choice
for your ablation, just note in your writeup which backbone you actually
trained.

---

## 6. Install the project itself and run the smoke tests

```bash
pip install -e .
pytest tests/ -v
```

**All 11 tests must pass before you touch real data.** If any fail, the
error will point at a specific import or shape mismatch — fix that before
moving on, don't work around it by skipping the test.

---

## 7. Confirm the full pipeline runs on synthetic data

```bash
python -m neurolive.train.train_baseline --smoke-test --epochs 2 --device cuda
python -m neurolive.train.train_joint --smoke-test --epochs 2 --device cuda
```

Both should print epoch losses, then a metrics.json summary with
`"is_synthetic": true` and a `"WARNING"` field. That warning is correct and
expected — this step only proves the pipeline runs on your GPU, it says
nothing about real accuracy yet.

Check `runs/phase1_baseline/metrics.json` and `runs/phase2_joint/metrics.json`
exist. If they do, your training loop, GPU, and checkpointing all work
end-to-end. This is the point where "the code works" becomes true on your
machine specifically, not just in the sandbox I built it in.

---

## 8. Get real data

1. Base dataset (genuine recordings only, no attacks):
   `https://github.com/GuangrongZhao/RGBE-Gaze` — follow their README to
   download from the linked OneDrive.
2. **The replay-attack extension is not publicly released** as far as I
   could find. Email the paper's authors (Mastropasqua et al., IEEE FG
   2026) and ask — cite the arXiv ID (2604.26285) in your email so they
   know exactly which recordings you mean.
3. **Important gap to know about now, not later:** the RGBE-Gaze download
   arrives in *their* raw layout (`convert2eventspace/`, `prophesee/`,
   `gazepoint/` folders per subject/session), not the
   `index.jsonl` + per-clip `.npy` layout `RGBELivenessDataset` expects.
   Converting between them — parsing their Prophesee event format,
   segmenting into challenge-response clips, generating segmentation labels
   for saccades/blinks — is real data-engineering work this repo doesn't
   do for you yet, because it depends on the exact raw format you actually
   receive. Once you have the download in hand, that conversion script is
   the next concrete coding task (and a good one to bring to a Claude Code
   session with the actual files in front of it, since I can't write a
   correct parser against a format I can't inspect).

Until then, `scripts/generate_synthetic_data.py` gets you a
correctly-shaped fake dataset on disk so you can exercise the *real* file-
loading code path (not just the in-memory synthetic dataset the smoke tests
use):
```bash
python scripts/generate_synthetic_data.py --out data/synthetic_smoke --n 64
python -m neurolive.train.train_baseline --data-root data/synthetic_smoke --epochs 5 --device cuda
```

---

## 9. Once real data + index.jsonl exist: real training

```bash
python -m neurolive.train.train_baseline \
  --data-root data/rgbe_gaze_liveness \
  --epochs 50 --batch-size 16 --num-bins 32 --device cuda

python -m neurolive.train.train_joint \
  --data-root data/rgbe_gaze_liveness \
  --epochs 50 --batch-size 16 --num-bins 32 --device cuda
```

6GB VRAM note: if you hit `CUDA out of memory`, drop `--batch-size` first
(try 8, then 4) before changing anything else — these models are small
enough that batch size is almost always the fix.

Real metrics land in `runs/phase1_baseline/metrics.json` and
`runs/phase2_joint/metrics.json`, without the synthetic warning this time.
**These are the only numbers that belong in your report.**

---

## 10. Latency benchmark (Phase 5)

```bash
python scripts/latency_benchmark.py --model joint --device cuda --n-runs 200 \
  --checkpoint runs/phase2_joint/joint_model.pt
```
Read the `SCOPE_NOTE` in its output before citing the number anywhere — it's
real inference latency on your GPU, not camera-to-decision latency (no event
camera exists to measure that first leg).

---

## 11. Run the demo (Phase 5)

Two terminals, both with `.venv` activated:

**Terminal 1 — backend:**
```bash
uvicorn main:app --reload --app-dir demo/backend --port 8000
```
Visit `http://localhost:8000/docs` — you should see the auto-generated
FastAPI docs with `/challenge`, `/decision`, `/health`.

**Terminal 2 — frontend:**
```bash
cd demo/frontend
npm install
npm run dev
```
It'll print a local URL (typically `http://localhost:5173`). Open it — you
should see the "Start challenge" / "Submit response" UI. `/decision` will
say `"inconclusive"` until you wire a trained checkpoint into
`demo/backend/main.py`'s decide() function (currently a stub — that's a
small, honest follow-up task once you have a real Phase 2 checkpoint from
step 9).

---

## 12. Generate report tables (Phase 6)

```bash
python scripts/generate_report.py
```
Writes `report_assets/results_tables.md`. Re-run this after every real
training run instead of copying numbers into your report by hand — it reads
straight from your metrics.json files, so there's no transcription step
where a number could get typo'd or fudged.

---

## Troubleshooting quick-reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is False | CPU-only torch install, or driver issue | Reinstall torch using the exact pytorch.org command; update NVIDIA driver |
| `mamba-ssm` build fails | Expected on native Windows; needs CUDA toolchain on WSL2 | Remove it from requirements.txt, use the automatic fallback backbone |
| `ModuleNotFoundError: neurolive` | Package not installed | `pip install -e .` from the repo root |
| `CUDA out of memory` | Batch size too large for 6GB VRAM | Lower `--batch-size` |
| `snn.utils` AttributeError | Wrong snntorch import | Already fixed in this repo's scnn.py — if you see this, check you're on the latest copy of the file |
| Frontend can't reach backend | Backend not running, or wrong port | Confirm `uvicorn` terminal is still running and shows port 8000; `API_BASE` in `App.jsx` must match |
| `FileNotFoundError: index.jsonl` | Using RGBELivenessDataset without real/synthetic data prepared | Run `scripts/generate_synthetic_data.py` first, or point `--data-root` at data with a real index.jsonl |

If you hit something not on this list, the actual error message + which
step you were on is exactly what to bring to a Claude Code session running
on this machine — it can see your real terminal output and your actual
files, which I can't from here.
