"""
NeuroLive - FastAPI backend with browser webcam support.

Pipeline:

Browser webcam
    ↓
JPEG frames
    ↓
FastAPI
    ↓
Frame differencing
    ↓
Pseudo events (x, y, t, polarity)
    ↓
Voxel grid (5, 64, 96)
    ↓
Phase 2 JointLivenessModel
    ↓
Liveness prediction

IMPORTANT:
The current Phase 2 checkpoint was trained using genuine RGBE-Gaze
samples only.

Therefore the model output is NOT a validated genuine-vs-replay
benchmark.

Attack/replay samples are required for valid APCER, BPCER and ACER.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from neurolive.data.event_repr import (
    activity_profile,
    events_to_voxel_grid,
)
from neurolive.models.joint_mamba import JointLivenessModel


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="NeuroLive Demo API",
    description="Browser webcam + simulated-event liveness inference",
    version="4.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Configuration
# ============================================================

# These MUST match the Phase 2 training configuration.
NUM_BINS = 5
HEIGHT = 64
WIDTH = 96

# Frame-difference threshold.
#
# Lower value:
#   More pseudo-events
#
# Higher value:
#   Fewer pseudo-events
#
EVENT_THRESHOLD = 15

# Increased from 30 seconds.
# This prevents the frontend from losing the session while
# the user is performing the challenge.
SESSION_TIMEOUT_SECONDS = 90

# Completed sessions are kept for this long.
#
# This is important because the frontend can accidentally
# send /decision twice.
COMPLETED_SESSION_RETENTION_SECONDS = 30

# Maximum pseudo-events generated from one webcam frame.
MAX_EVENTS_PER_FRAME = 3000

# Maximum events stored for one challenge.
MAX_TOTAL_EVENTS = 100000

# Minimum number of events required for inference.
MIN_EVENTS_FOR_DECISION = 100

# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "phase2_joint"
    / "joint_model.pt"
)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Model
# ============================================================

model = None
model_load_error = None


def load_model():
    """
    Load the trained Phase 2 checkpoint once when FastAPI starts.
    """

    global model
    global model_load_error

    try:

        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT}"
            )

        # Your trained Phase 2 model was trained with
        # the fallback Conv1D + GRU backbone.
        model = JointLivenessModel(
            d_model=128,
            force_fallback=True,
        ).to(DEVICE)

        state = torch.load(
            CHECKPOINT,
            map_location=DEVICE,
        )

        # Support different checkpoint formats.
        if isinstance(state, dict):

            if "state_dict" in state:
                state = state["state_dict"]

            elif "model_state_dict" in state:
                state = state["model_state_dict"]

        model.load_state_dict(
            state,
            strict=True,
        )

        model.eval()

        print("=" * 70)
        print("NeuroLive model loaded successfully")
        print("Checkpoint :", CHECKPOINT)
        print("Device     :", DEVICE)

        if DEVICE.type == "cuda":
            print(
                "GPU        :",
                torch.cuda.get_device_name(0),
            )

        print(
            "Input      :",
            f"({NUM_BINS}, {HEIGHT}, {WIDTH})",
        )

        print(
            "Backbone   :",
            model.backbone_name,
        )

        print("=" * 70)

    except Exception as exc:

        model = None
        model_load_error = str(exc)

        print("=" * 70)
        print("WARNING: NeuroLive model loading failed")
        print("ERROR:", model_load_error)
        print("=" * 70)


# Load model during application startup.
load_model()


# ============================================================
# Session storage
# ============================================================

_sessions: dict[str, dict] = {}


# ============================================================
# Challenges
# ============================================================

CHALLENGES = [
    "blink_twice",
    "saccade_left_right",
    "saccade_up_down",
]


# ============================================================
# Pydantic models
# ============================================================

class ChallengeResponse(BaseModel):
    session_id: str
    challenge: str
    expires_in_s: int


class FrameResponse(BaseModel):
    session_id: str
    events_added: int
    total_events: int


class DecisionRequest(BaseModel):
    session_id: str

    # These are calculated by the browser challenge detector.
    challenge_passed: bool = False

    # 0.0 - 1.0
    challenge_score: float = 0.0

    challenge_message: str = ""


class DecisionResult(BaseModel):
    session_id: str
    challenge: str

    challenge_passed: bool
    challenge_score: float

    decision: str
    liveness_confidence: float

    activity_profile: list[float]

    num_events: int
    device: str

    note: str


# ============================================================
# Session cleanup
# ============================================================

def cleanup_sessions():
    """
    Remove very old sessions.

    Completed sessions are retained for a short period so that
    duplicate /decision requests can receive the same result
    instead of returning 'Unknown or expired session'.
    """

    now = time.time()

    expired_ids = []

    for session_id, session in list(_sessions.items()):

        created_at = session.get(
            "created_at",
            now,
        )

        completed_at = session.get(
            "completed_at"
        )

        # Completed session.
        if completed_at is not None:

            if (
                now - completed_at
                > COMPLETED_SESSION_RETENTION_SECONDS
            ):
                expired_ids.append(session_id)

        # Active session.
        else:

            if (
                now - created_at
                > SESSION_TIMEOUT_SECONDS
            ):
                expired_ids.append(session_id)

    for session_id in expired_ids:
        _sessions.pop(
            session_id,
            None,
        )


# ============================================================
# Root
# ============================================================

@app.get("/")
def root():

    cleanup_sessions()

    return {
        "application": "NeuroLive",
        "status": "running",

        "model_loaded": model is not None,

        "device": str(DEVICE),

        "checkpoint": str(CHECKPOINT),

        "checkpoint_exists": CHECKPOINT.exists(),

        "backbone": (
            model.backbone_name
            if model is not None
            else None
        ),

        "input_shape": [
            NUM_BINS,
            HEIGHT,
            WIDTH,
        ],

        "session_timeout_seconds":
            SESSION_TIMEOUT_SECONDS,

        "event_threshold":
            EVENT_THRESHOLD,
    }


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():

    cleanup_sessions()

    return {

        "status": "ok",

        "model_loaded":
            model is not None,

        "device":
            str(DEVICE),

        "checkpoint":
            str(CHECKPOINT),

        "checkpoint_exists":
            CHECKPOINT.exists(),

        "num_bins":
            NUM_BINS,

        "height":
            HEIGHT,

        "width":
            WIDTH,

        "event_threshold":
            EVENT_THRESHOLD,

        "min_events":
            MIN_EVENTS_FOR_DECISION,

        "backbone": (
            model.backbone_name
            if model is not None
            else None
        ),

        "active_sessions":
            len(_sessions),
    }


# ============================================================
# Start challenge
# ============================================================

@app.post(
    "/challenge",
    response_model=ChallengeResponse,
)
def new_challenge():

    cleanup_sessions()

    session_id = str(
        uuid.uuid4()
    )

    challenge = str(
        np.random.choice(
            CHALLENGES
        )
    )

    now = time.time()

    _sessions[session_id] = {

        "challenge":
            challenge,

        "created_at":
            now,

        "completed_at":
            None,

        # Session-relative timestamp.
        #
        # This prevents float32 precision problems caused by
        # using the Unix epoch directly as a microsecond value.
        "start_time_us":
            time.time_ns() / 1000.0,

        # Previous grayscale frame.
        "previous_gray":
            None,

        # Accumulated pseudo-events.
        "events":
            [],

        # Store result for duplicate /decision requests.
        "result":
            None,
    }

    print(
        f"[CHALLENGE] "
        f"session={session_id} "
        f"challenge={challenge}"
    )

    return ChallengeResponse(

        session_id=session_id,

        challenge=challenge,

        expires_in_s=
            SESSION_TIMEOUT_SECONDS,
    )


# ============================================================
# Frame -> pseudo events
# ============================================================

def frame_to_events(
    frame: np.ndarray,
    previous_gray: np.ndarray | None,
    timestamp_us: float,
):
    """
    Convert two webcam frames into simulated events.

    Event format:

        [x, y, timestamp_us, polarity]

    polarity:

        +1 = intensity increased
        -1 = intensity decreased

    IMPORTANT:

    These are simulated events generated from RGB webcam
    frames.

    They are NOT measurements from a real event camera.
    """

    # Resize to the model spatial resolution.
    frame = cv2.resize(
        frame,
        (WIDTH, HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    # Convert to grayscale.
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    # First frame cannot generate differences.
    if previous_gray is None:

        return (
            gray,
            np.zeros(
                (0, 4),
                dtype=np.float32,
            ),
        )

    # Frame difference.
    diff = (
        gray.astype(np.int16)
        - previous_gray.astype(np.int16)
    )

    # Threshold.
    mask = (
        np.abs(diff)
        >= EVENT_THRESHOLD
    )

    ys, xs = np.where(mask)

    # No motion.
    if len(xs) == 0:

        return (
            gray,
            np.zeros(
                (0, 4),
                dtype=np.float32,
            ),
        )

    # --------------------------------------------------------
    # Limit events per frame
    # --------------------------------------------------------

    if len(xs) > MAX_EVENTS_PER_FRAME:

        magnitudes = np.abs(
            diff[ys, xs]
        )

        keep = np.argpartition(
            magnitudes,
            -MAX_EVENTS_PER_FRAME,
        )[
            -MAX_EVENTS_PER_FRAME:
        ]

        xs = xs[keep]
        ys = ys[keep]

    # --------------------------------------------------------
    # Polarity
    # --------------------------------------------------------

    polarity = np.where(
        diff[ys, xs] > 0,
        1.0,
        -1.0,
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamps = np.full(
        len(xs),
        timestamp_us,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Event array
    # --------------------------------------------------------

    events = np.column_stack(
        [
            xs.astype(np.float32),
            ys.astype(np.float32),
            timestamps,
            polarity,
        ]
    )

    return (
        gray,
        events,
    )


# ============================================================
# Browser webcam frame endpoint
# ============================================================

@app.post(
    "/camera/frame",
    response_model=FrameResponse,
)
async def camera_frame(
    session_id: str,
    frame: UploadFile = File(...),
):

    cleanup_sessions()

    # --------------------------------------------------------
    # Validate session
    # --------------------------------------------------------

    if session_id not in _sessions:

        raise HTTPException(
            status_code=404,
            detail=(
                "Unknown session. "
                "Start a new challenge."
            ),
        )

    session = _sessions[
        session_id
    ]

    # --------------------------------------------------------
    # Do not accept frames after decision
    # --------------------------------------------------------

    if session.get(
        "completed_at"
    ) is not None:

        raise HTTPException(
            status_code=409,
            detail=(
                "Challenge already completed. "
                "Start a new challenge."
            ),
        )

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - session["created_at"]
    )

    if (
        elapsed
        > SESSION_TIMEOUT_SECONDS
    ):

        _sessions.pop(
            session_id,
            None,
        )

        raise HTTPException(
            status_code=408,
            detail=(
                "Challenge expired. "
                "Start a new challenge."
            ),
        )

    # --------------------------------------------------------
    # Read uploaded JPEG
    # --------------------------------------------------------

    data = await frame.read()

    if not data:

        raise HTTPException(
            status_code=400,
            detail="Empty webcam frame.",
        )

    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    image_array = np.frombuffer(
        data,
        dtype=np.uint8,
    )

    frame_bgr = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if frame_bgr is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not decode webcam frame."
            ),
        )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_us = (
        time.time_ns() / 1000.0
        - session["start_time_us"]
    )

    # --------------------------------------------------------
    # Generate pseudo-events
    # --------------------------------------------------------

    previous_gray = (
        session["previous_gray"]
    )

    current_gray, events = (
        frame_to_events(
            frame_bgr,
            previous_gray,
            timestamp_us,
        )
    )

    session["previous_gray"] = (
        current_gray
    )

    # --------------------------------------------------------
    # Store events
    # --------------------------------------------------------

    current_total = sum(
        len(x)
        for x in session["events"]
    )

    remaining = (
        MAX_TOTAL_EVENTS
        - current_total
    )

    events_to_store = events[
        :max(0, remaining)
    ]

    if len(events_to_store) > 0:

        session["events"].append(
            events_to_store
        )

    total_events = sum(
        len(x)
        for x in session["events"]
    )

    return FrameResponse(

        session_id=
            session_id,

        events_added=
            len(events_to_store),

        total_events=
            total_events,
    )


# ============================================================
# Decision endpoint
# ============================================================

@app.post(
    "/decision",
    response_model=DecisionResult,
)
def decide(
    request: DecisionRequest,
):

    cleanup_sessions()

    # --------------------------------------------------------
    # Model check
    # --------------------------------------------------------

    if model is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Model could not be loaded: "
                + str(model_load_error)
            ),
        )

    # --------------------------------------------------------
    # Session check
    # --------------------------------------------------------

    if request.session_id not in _sessions:

        raise HTTPException(
            status_code=404,
            detail=(
                "Unknown or expired session. "
                "Please start a new challenge."
            ),
        )

    session = _sessions[
        request.session_id
    ]

    # --------------------------------------------------------
    # DUPLICATE SUBMISSION FIX
    # --------------------------------------------------------
    #
    # If frontend accidentally sends /decision twice,
    # return the exact previous result.
    #
    # This prevents:
    #
    # "Unknown or expired session"
    #
    # after the first successful request.

    if session.get(
        "result"
    ) is not None:

        print(
            "[DECISION] "
            "Duplicate request - "
            "returning cached result."
        )

        return session["result"]

    # --------------------------------------------------------
    # Check timeout
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - session["created_at"]
    )

    if (
        elapsed
        > SESSION_TIMEOUT_SECONDS
    ):

        _sessions.pop(
            request.session_id,
            None,
        )

        raise HTTPException(
            status_code=408,
            detail=(
                "Challenge session expired. "
                "Please start a new challenge."
            ),
        )

    # --------------------------------------------------------
    # Gather events
    # --------------------------------------------------------

    if session["events"]:

        events = np.concatenate(
            session["events"],
            axis=0,
        )

    else:

        events = np.zeros(
            (0, 4),
            dtype=np.float32,
        )

    # --------------------------------------------------------
    # Validate event count
    # --------------------------------------------------------

    if (
        len(events)
        < MIN_EVENTS_FOR_DECISION
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Only {len(events)} motion events "
                f"were captured. At least "
                f"{MIN_EVENTS_FOR_DECISION} events "
                f"are required. Move your face/eyes "
                f"and try again."
            ),
        )

    print(
        f"[DECISION] "
        f"session={request.session_id} "
        f"events={len(events)} "
        f"challenge_passed="
        f"{request.challenge_passed}"
    )

    # --------------------------------------------------------
    # Activity profile
    # --------------------------------------------------------

    profile = activity_profile(
        events,
        num_bins=NUM_BINS,
    )

    profile = [
        float(x)
        for x in profile
    ]

    # --------------------------------------------------------
    # Challenge verification
    # --------------------------------------------------------

    if not request.challenge_passed:

        note = (
            "Challenge-response verification "
            "failed. The model result was not "
            "promoted to a genuine decision. "
            "The current Phase 2 checkpoint "
            "was trained on genuine RGBE-Gaze "
            "samples only."
        )

        result = DecisionResult(

            session_id=
                request.session_id,

            challenge=
                session["challenge"],

            challenge_passed=False,

            challenge_score=
                float(
                    request.challenge_score
                ),

            decision=
                "inconclusive",

            liveness_confidence=0.0,

            activity_profile=
                profile,

            num_events=
                len(events),

            device=
                str(DEVICE),

            note=
                note,
        )

        # Cache result.
        session["result"] = result.model_dump()

        session["completed_at"] = (
            time.time()
        )

        return result

    # --------------------------------------------------------
    # Convert events -> voxel grid
    # --------------------------------------------------------

    voxel = events_to_voxel_grid(
        events,
        num_bins=NUM_BINS,
        height=HEIGHT,
        width=WIDTH,
    )

    # --------------------------------------------------------
    # Verify voxel shape
    # --------------------------------------------------------

    expected_shape = (
        NUM_BINS,
        HEIGHT,
        WIDTH,
    )

    if tuple(voxel.shape) != expected_shape:

        raise HTTPException(
            status_code=500,
            detail=(
                "Unexpected voxel shape: "
                f"{tuple(voxel.shape)}. "
                f"Expected {expected_shape}."
            ),
        )

    # --------------------------------------------------------
    # Batch dimension + device
    # --------------------------------------------------------

    voxel = (
        voxel
        .unsqueeze(0)
        .to(DEVICE)
        .float()
    )

    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    try:

        with torch.no_grad():

            _, liveness_logits = (
                model(voxel)
            )

            probabilities = (
                torch.softmax(
                    liveness_logits,
                    dim=-1,
                )[0]
            )

            predicted_class = int(
                probabilities.argmax().item()
            )

            confidence = float(
                probabilities[
                    predicted_class
                ].item()
            )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Model inference failed: "
                + str(exc)
            ),
        )

    # --------------------------------------------------------
    # Class convention
    # --------------------------------------------------------
    #
    # 0 = attack
    # 1 = genuine
    #

    if predicted_class == 1:

        decision = "genuine"

    else:

        decision = "attack"

    # --------------------------------------------------------
    # Important limitation
    # --------------------------------------------------------

    note = (
        "Camera frames were converted to "
        "simulated events using frame "
        "differencing and processed by the "
        "trained Phase 2 model. The "
        "challenge-response check also passed. "
        "IMPORTANT: the current checkpoint was "
        "trained on genuine RGBE-Gaze samples "
        "only. Therefore this is a live "
        "end-to-end inference demonstration, "
        "not a validated genuine-vs-replay "
        "liveness benchmark. Attack/replay "
        "samples are required for valid APCER, "
        "BPCER and ACER."
    )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    result = DecisionResult(

        session_id=
            request.session_id,

        challenge=
            session["challenge"],

        challenge_passed=True,

        challenge_score=
            float(
                request.challenge_score
            ),

        decision=
            decision,

        liveness_confidence=
            confidence,

        activity_profile=
            profile,

        num_events=
            len(events),

        device=
            str(DEVICE),

        note=
            note,
    )

    # --------------------------------------------------------
    # CACHE RESULT
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # DO NOT delete the session immediately.
    #
    # The frontend may accidentally send /decision
    # again. We return the cached result instead.

    session["result"] = (
        result.model_dump()
    )

    session["completed_at"] = (
        time.time()
    )

    print(
        f"[RESULT] "
        f"decision={decision} "
        f"confidence={confidence:.4f} "
        f"events={len(events)}"
    )

    return result


# ============================================================
# Session status endpoint
# ============================================================

@app.get(
    "/session/{session_id}"
)
def session_status(
    session_id: str,
):

    cleanup_sessions()

    if session_id not in _sessions:

        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    session = _sessions[
        session_id
    ]

    total_events = sum(
        len(x)
        for x in session["events"]
    )

    return {

        "session_id":
            session_id,

        "challenge":
            session["challenge"],

        "created_at":
            session["created_at"],

        "completed":
            session.get(
                "completed_at"
            ) is not None,

        "total_events":
            total_events,

        "has_result":
            session.get(
                "result"
            ) is not None,
    }


# ============================================================
# Manual session reset
# ============================================================

@app.delete(
    "/session/{session_id}"
)
def delete_session(
    session_id: str,
):

    if session_id not in _sessions:

        return {
            "status": "already_removed",
            "session_id": session_id,
        }

    _sessions.pop(
        session_id,
        None,
    )

    return {
        "status": "deleted",
        "session_id": session_id,
    }