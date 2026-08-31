"""
Standalone real-time webcam demo for NeuroLive.

Why standalone rather than the FastAPI+React app for a live presentation:
one Python process, no dev server / backend server / browser to keep alive
simultaneously on stage. Fewer things that can fail mid-defense.

WHAT THIS ACTUALLY DOES (read before presenting):
  1. Captures frames from your laptop webcam.
  2. Simulates events from consecutive frames via thresholded differencing
     (neurolive.data.webcam_events) — NOT real event-camera data. This is
     clearly labeled in the on-screen UI; keep it labeled that way if you
     show this live.
  3. Localizes eye ROIs on the *simulated event stream itself* (Phase 4's
     event-native heuristic) — so even in this webcam fallback, ROI
     localization never looks at the raw RGB frame, which preserves the
     architecture's core claim even though the events feeding it are
     approximated.
  4. Runs the Phase 2 joint model (checkpoint you trained) on the
     resulting voxel grid.
  5. If your checkpoint was trained on genuine-only data (check your own
     STATUS.md / runs/*/metrics.json), the genuine-vs-attack decision is
     NOT a validated result — say so if asked. This script still shows the
     decision, because showing the pipeline run end-to-end is legitimate;
     just don't claim it as a validated liveness detector.

USAGE:
    python scripts/live_webcam_demo.py --checkpoint runs/phase2_joint/joint_model.pt --device cuda

CONTROLS:
    SPACE — start a challenge (accumulate events for --challenge-duration seconds, then infer)
    ESC   — quit

NOTE ON THE MODEL CLASS:
    This loads neurolive.models.joint_mamba.JointLivenessModel, matching
    "Processed by Phase 2 Model" in your current demo UI. If your actual
    checkpoint uses a different class, config, or a wrapped state_dict
    (e.g. saved inside a dict with other keys), edit `load_model()` below
    to match exactly how you saved it — I don't have that exact saving
    code in front of me, so this is a best-effort match, not a guarantee.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch

from neurolive.data.event_repr import activity_profile, events_to_voxel_grid
from neurolive.data.webcam_events import FrameEventSimulator, FrameEventSimulatorConfig
from neurolive.localization.event_native_roi import localize_eye_rois
from neurolive.models.joint_mamba import JointLivenessModel

# ---------- UI colors (BGR, OpenCV convention) ----------
BG = (36, 26, 20)
ACCENT = (216, 180, 0)      # cyan-ish
WHITE = (240, 240, 240)
GRAY = (140, 140, 140)
GREEN = (110, 200, 60)
RED = (70, 70, 220)
AMBER = (0, 170, 255)

STATE_IDLE, STATE_CAPTURING, STATE_INFERRING, STATE_RESULT = "idle", "capturing", "inferring", "result"


def load_model(checkpoint_path: str, device: str) -> JointLivenessModel:
    model = JointLivenessModel(force_fallback=(device == "cpu"))
    state = torch.load(checkpoint_path, map_location=device)
    # handle both a raw state_dict and a {"model_state_dict": ...}-style wrapper
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval().to(device)
    print(f"[NeuroLive] Loaded checkpoint: {checkpoint_path} | backbone: {model.backbone_name} | device: {device}")
    return model


def draw_banner(frame, text, color, y):
    cv2.rectangle(frame, (0, y), (frame.shape[1], y + 26), BG, -1)
    cv2.putText(frame, text, (10, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_activity_profile(frame, profile, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 50, 45), -1)
    if profile is None or len(profile) < 2 or profile.max() <= 0:
        return
    norm = profile / (profile.max() + 1e-6)
    pts = []
    for i, v in enumerate(norm):
        px = x + int(i / (len(norm) - 1) * w)
        py = y + h - int(v * (h - 6)) - 3
        pts.append((px, py))
    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, ACCENT, 2, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--challenge-duration", type=float, default=1.5, help="seconds of events to accumulate per challenge")
    p.add_argument("--threshold", type=float, default=12.0, help="frame-diff intensity threshold for simulated events")
    p.add_argument("--num-bins", type=int, default=32)
    p.add_argument("--voxel-size", type=int, default=64, help="voxel grid height/width fed to the model")
    p.add_argument("--result-hold-seconds", type=float, default=4.0)
    args = p.parse_args()

    model = load_model(args.checkpoint, args.device)
    sim = FrameEventSimulator(FrameEventSimulatorConfig(threshold=args.threshold, downscale_to=(args.voxel_size, args.voxel_size)))

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera_index}. Try a different --camera-index.")

    state = STATE_IDLE
    captured_events = []
    challenge_start_t = None
    result_shown_at = None
    last_decision, last_confidence, last_profile = None, None, None

    print("[NeuroLive] SPACE = start challenge, ESC = quit")
    t0 = time.perf_counter()

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            print("[NeuroLive] Camera read failed, stopping.")
            break
        t_us = (time.perf_counter() - t0) * 1_000_000
        events = sim.step(frame_bgr, t_us)

        if state == STATE_CAPTURING:
            if events.shape[0] > 0:
                captured_events.append(events)
            if time.perf_counter() - challenge_start_t >= args.challenge_duration:
                state = STATE_INFERRING

        display = cv2.resize(frame_bgr, (640, 480))
        scale_x, scale_y = 640 / frame_bgr.shape[1], 480 / frame_bgr.shape[0]

        if state == STATE_INFERRING:
            all_events = np.concatenate(captured_events, axis=0) if captured_events else np.zeros((0, 4), dtype=np.float32)
            rois = localize_eye_rois(all_events, frame_height=args.voxel_size, frame_width=args.voxel_size, roi_size=max(8, args.voxel_size // 4))
            voxel = events_to_voxel_grid(all_events, args.num_bins, args.voxel_size, args.voxel_size).unsqueeze(0).to(args.device)
            with torch.no_grad():
                seg_logits, liveness_logits = model(voxel)
                probs = torch.softmax(liveness_logits, dim=-1)[0]
                pred = int(probs.argmax())
                confidence = float(probs[pred])
            last_decision = "GENUINE" if pred == 1 else "ATTACK-LIKE"
            last_confidence = confidence
            last_profile = activity_profile(all_events, num_bins=40)
            print(f"[NeuroLive] n_events={all_events.shape[0]} rois={len(rois)} decision={last_decision} confidence={confidence:.3f}")
            captured_events = []
            state = STATE_RESULT
            result_shown_at = time.perf_counter()

        if state == STATE_RESULT and time.perf_counter() - result_shown_at > args.result_hold_seconds:
            state = STATE_IDLE

        # ---- overlay ----
        draw_banner(display, "NeuroLive — Live Webcam Demo  |  SIMULATED EVENTS, not real event-camera data", AMBER, 0)

        if state == STATE_IDLE:
            cv2.putText(display, "SPACE: start challenge   ESC: quit", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        elif state == STATE_CAPTURING:
            remaining = max(0.0, args.challenge_duration - (time.perf_counter() - challenge_start_t))
            cv2.putText(display, f"Capturing challenge... {remaining:.1f}s", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT, 2, cv2.LINE_AA)
        elif state == STATE_RESULT:
            color = GREEN if last_decision == "GENUINE" else RED
            cv2.rectangle(display, (10, 40), (400, 200), (30, 25, 22), -1)
            cv2.putText(display, last_decision, (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            cv2.putText(display, f"confidence: {last_confidence:.1%}", (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
            cv2.putText(display, "Activity profile:", (25, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1, cv2.LINE_AA)
            draw_activity_profile(display, last_profile, 25, 158, 350, 35)

        cv2.imshow("NeuroLive Live Demo", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32 and state == STATE_IDLE:  # SPACE
            state = STATE_CAPTURING
            captured_events = []
            sim.reset()
            challenge_start_t = time.perf_counter()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
