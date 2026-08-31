import { useEffect, useRef, useState } from "react";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import Landing from "./Landing";
import LiveDemo from "./LiveDemo";


// ============================================================
// CONFIGURATION
// ============================================================

const API_BASE = "http://localhost:8000";

const WIDTH = 96;
const HEIGHT = 64;

// Lower threshold = more simulated events.
// 18 is reasonable for normal webcam movement.
const EVENT_THRESHOLD = 18;

// Send webcam frames every 100 ms.
const FRAME_UPLOAD_INTERVAL = 100;

// Minimum time before automatic submission.
const MIN_CAPTURE_SECONDS = 3.0;

// Safety maximum.
const MAX_CAPTURE_SECONDS = 12.0;

// Backend should have at least this many events.
const MIN_BACKEND_EVENTS = 100;

// MediaPipe model.
const FACE_MODEL = "/models/face_landmarker.task";

// Webcam frame upload resolution.
const UPLOAD_WIDTH = 320;
const UPLOAD_HEIGHT = 240;

// Decision request timeout.
const DECISION_TIMEOUT_MS = 30000;

// ============================================================
// CHALLENGE NAMES
// ============================================================

const CHALLENGE_NAMES = {
  blink_twice: "BLINK TWICE",
  saccade_left_right: "LOOK LEFT → RIGHT",
  saccade_up_down: "LOOK UP → DOWN",
};

// ============================================================
// UTILITY FUNCTIONS
// ============================================================

function distance(a, b) {
  if (!a || !b) {
    return 0;
  }

  const dx = a.x - b.x;
  const dy = a.y - b.y;

  return Math.sqrt(dx * dx + dy * dy);
}

function average(points) {
  if (!points || !points.length) {
    return {
      x: 0,
      y: 0,
    };
  }

  return {
    x:
      points.reduce(
        (sum, p) => sum + p.x,
        0
      ) / points.length,

    y:
      points.reduce(
        (sum, p) => sum + p.y,
        0
      ) / points.length,
  };
}

// ============================================================
// EYE ASPECT RATIO
// ============================================================

function eyeAspectRatio(lm, ids) {
  if (!lm || lm.length < 478) {
    return 0;
  }

  const p1 = lm[ids[0]];
  const p2 = lm[ids[1]];
  const p3 = lm[ids[2]];
  const p4 = lm[ids[3]];
  const p5 = lm[ids[4]];
  const p6 = lm[ids[5]];

  if (
    !p1 ||
    !p2 ||
    !p3 ||
    !p4 ||
    !p5 ||
    !p6
  ) {
    return 0;
  }

  const vertical1 = distance(p2, p6);
  const vertical2 = distance(p3, p5);

  const horizontal = distance(p1, p4);

  if (horizontal < 1e-6) {
    return 0;
  }

  return (
    (vertical1 + vertical2) /
    (2 * horizontal)
  );
}

// ============================================================
// GAZE ESTIMATION
// ============================================================

function getGaze(lm) {
  if (!lm || lm.length < 478) {
    return {
      horizontal: 0.5,
      vertical: 0.5,
    };
  }

  const leftIris = average([
    lm[474],
    lm[475],
    lm[476],
    lm[477],
  ]);

  const rightIris = average([
    lm[469],
    lm[470],
    lm[471],
    lm[472],
  ]);

  const leftEyeOuter = lm[263];
  const leftEyeInner = lm[362];

  const rightEyeOuter = lm[33];
  const rightEyeInner = lm[133];

  const leftWidth = Math.max(
    0.001,
    distance(
      leftEyeOuter,
      leftEyeInner
    )
  );

  const rightWidth = Math.max(
    0.001,
    distance(
      rightEyeOuter,
      rightEyeInner
    )
  );

  const leftRatio =
    (leftIris.x - leftEyeOuter.x) /
    leftWidth;

  const rightRatio =
    (rightIris.x - rightEyeOuter.x) /
    rightWidth;

  const horizontal =
    (leftRatio + rightRatio) / 2;

  const leftUpper = lm[386];
  const leftLower = lm[374];

  const rightUpper = lm[159];
  const rightLower = lm[145];

  const leftHeight = Math.max(
    0.001,
    distance(leftUpper, leftLower)
  );

  const rightHeight = Math.max(
    0.001,
    distance(rightUpper, rightLower)
  );

  const leftVertical =
    (leftIris.y - leftUpper.y) /
    leftHeight;

  const rightVertical =
    (rightIris.y - rightUpper.y) /
    rightHeight;

  const vertical =
    (leftVertical + rightVertical) / 2;

  return {
    horizontal,
    vertical,
  };
}

// ============================================================
// APP
// ============================================================

export default function App() {
  const [view, setView] = useState("landing");

  // ==========================================================
  // DOM
  // ==========================================================

  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // ==========================================================
  // CAMERA / MEDIAPIPE
  // ==========================================================

  const streamRef = useRef(null);
  const faceLandmarkerRef = useRef(null);

  const animationRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);

  // ==========================================================
  // LOCAL EVENT STREAM
  // ==========================================================

  const previousFrameRef = useRef(null);
  const eventsRef = useRef([]);

  // ==========================================================
  // CHALLENGE
  // ==========================================================

  const challengeRef = useRef(null);

  const challengeStartRef = useRef(0);

  const blinkClosedRef = useRef(false);

  const blinkCountRef = useRef(0);

  const gazeBaselineRef = useRef(null);

  const gazeStateRef = useRef("CENTER");

  // ==========================================================
  // BACKEND STREAM
  // ==========================================================

  const frameTimerRef = useRef(null);

  const uploadingFrameRef = useRef(false);

  const backendEventCountRef = useRef(0);

  // ==========================================================
  // SUBMISSION
  // ==========================================================

  const submittingRef = useRef(false);

  const autoSubmitTimerRef = useRef(null);

  // ==========================================================
  // REACT STATE
  // ==========================================================

  const [cameraReady, setCameraReady] =
    useState(false);

  const [modelReady, setModelReady] =
    useState(false);

  const [session, setSession] =
    useState(null);

  const [challenge, setChallenge] =
    useState(null);

  const [challengeStatus, setChallengeStatus] =
    useState("idle");

  const [challengeMessage, setChallengeMessage] =
    useState("Press Start Challenge");

  const [blinkCount, setBlinkCount] =
    useState(0);

  const [gazeState, setGazeState] =
    useState("CENTER");

  const [eventCount, setEventCount] =
    useState(0);

  const [backendEventCount, setBackendEventCount] =
    useState(0);

  const [decision, setDecision] =
    useState(null);

  const [error, setError] =
    useState(null);

  const [loading, setLoading] =
    useState(true);

  // ==========================================================
  // CAMERA INITIALIZATION
  // ==========================================================

  useEffect(() => {
    let mounted = true;

    async function startCamera() {
      try {
        if (
          !navigator.mediaDevices ||
          !navigator.mediaDevices.getUserMedia
        ) {
          throw new Error(
            "Browser does not support webcam access."
          );
        }

        const stream =
          await navigator.mediaDevices.getUserMedia(
            {
              video: {
                width: {
                  ideal: 640,
                },

                height: {
                  ideal: 480,
                },

                facingMode: "user",
              },

              audio: false,
            }
          );

        if (!mounted) {
          stream
            .getTracks()
            .forEach((track) =>
              track.stop()
            );

          return;
        }

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject =
            stream;

          await videoRef.current.play();

          setCameraReady(true);

          console.log(
            "[CanthusCore] Camera ready."
          );
        }
      } catch (err) {
        console.error(
          "[CanthusCore] Camera error:",
          err
        );

        setError(
          `Could not access webcam: ${
            err.message ||
            "Please allow camera permission."
          }`
        );
      }
    }

    startCamera();

    return () => {
      mounted = false;

      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) =>
            track.stop()
          );
      }
    };
  }, []);

  // ==========================================================
  // MEDIAPIPE INITIALIZATION
  // ==========================================================

  useEffect(() => {
    let mounted = true;

    async function initializeFaceLandmarker() {
      try {
        console.log(
          "[CanthusCore] Loading MediaPipe..."
        );

        const vision =
          await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
          );

        const landmarker =
          await FaceLandmarker.createFromOptions(
            vision,
            {
              baseOptions: {
                modelAssetPath:
                  FACE_MODEL,

                delegate: "GPU",
              },

              runningMode: "VIDEO",

              numFaces: 1,

              minFaceDetectionConfidence: 0.5,

              minFacePresenceConfidence: 0.5,

              minTrackingConfidence: 0.5,
            }
          );

        if (!mounted) {
          landmarker.close();
          return;
        }

        faceLandmarkerRef.current =
          landmarker;

        setModelReady(true);
        setLoading(false);

        console.log(
          "[CanthusCore] MediaPipe ready."
        );
      } catch (err) {
        console.error(
          "[CanthusCore] MediaPipe error:",
          err
        );

        setError(
          "Face landmark model could not be loaded. " +
          "Make sure public/models/face_landmarker.task exists."
        );

        setLoading(false);
      }
    }

    initializeFaceLandmarker();

    return () => {
      mounted = false;

      if (
        faceLandmarkerRef.current
      ) {
        faceLandmarkerRef.current.close();
      }
    };
  }, []);

  // ==========================================================
  // GENERATE SIMULATED EVENTS
  // ==========================================================

  function generateEvents(
    video,
    timestampMs
  ) {
    const canvas =
      canvasRef.current;

    if (!canvas) {
      return;
    }

    const ctx =
      canvas.getContext("2d", {
        willReadFrequently: true,
      });

    if (!ctx) {
      return;
    }

    canvas.width = WIDTH;
    canvas.height = HEIGHT;

    ctx.drawImage(
      video,
      0,
      0,
      WIDTH,
      HEIGHT
    );

    const imageData =
      ctx.getImageData(
        0,
        0,
        WIDTH,
        HEIGHT
      );

    const current =
      imageData.data;

    const previous =
      previousFrameRef.current;

    if (!previous) {
      previousFrameRef.current =
        new Uint8Array(current);

      return;
    }

    const maxEventsPerFrame = 400;

    let generated = 0;

    for (
      let y = 0;
      y < HEIGHT;
      y++
    ) {
      for (
        let x = 0;
        x < WIDTH;
        x++
      ) {
        const index =
          (y * WIDTH + x) * 4;

        const currentGray =
          0.299 * current[index] +
          0.587 *
            current[index + 1] +
          0.114 *
            current[index + 2];

        const previousGray =
          0.299 * previous[index] +
          0.587 *
            previous[index + 1] +
          0.114 *
            previous[index + 2];

        const difference =
          currentGray -
          previousGray;

        if (
          Math.abs(difference) >=
          EVENT_THRESHOLD
        ) {
          eventsRef.current.push({
            x,
            y,

            // microseconds
            t: timestampMs * 1000,

            p:
              difference > 0
                ? 1
                : -1,
          });

          generated++;

          if (
            generated >=
            maxEventsPerFrame
          ) {
            break;
          }
        }
      }

      if (
        generated >=
        maxEventsPerFrame
      ) {
        break;
      }
    }

    previousFrameRef.current =
      new Uint8Array(current);

    setEventCount(
      eventsRef.current.length
    );
  }

  // ==========================================================
  // UPLOAD WEBCAM FRAME TO FASTAPI
  // ==========================================================

  async function uploadCurrentFrame() {
    if (!session?.session_id) {
      return;
    }

    if (!videoRef.current) {
      return;
    }

    if (uploadingFrameRef.current) {
      return;
    }

    const video =
      videoRef.current;

    if (
      video.readyState <
      HTMLMediaElement.HAVE_CURRENT_DATA
    ) {
      return;
    }

    const canvas =
      document.createElement(
        "canvas"
      );

    canvas.width =
      UPLOAD_WIDTH;

    canvas.height =
      UPLOAD_HEIGHT;

    const ctx =
      canvas.getContext("2d");

    if (!ctx) {
      return;
    }

    ctx.drawImage(
      video,
      0,
      0,
      UPLOAD_WIDTH,
      UPLOAD_HEIGHT
    );

    uploadingFrameRef.current =
      true;

    try {
      const blob =
        await new Promise(
          (resolve) => {
            canvas.toBlob(
              resolve,
              "image/jpeg",
              0.65
            );
          }
        );

      if (!blob) {
        return;
      }

      const formData =
        new FormData();

      formData.append(
        "frame",
        blob,
        "webcam.jpg"
      );

      const url =
        `${API_BASE}/camera/frame` +
        `?session_id=${encodeURIComponent(
          session.session_id
        )}`;

      const response =
        await fetch(url, {
          method: "POST",
          body: formData,
        });

      if (!response.ok) {
        const text =
          await response.text();

        console.error(
          "[CanthusCore] Frame upload failed:",
          response.status,
          text
        );

        return;
      }

      const data =
        await response.json();

      // ------------------------------------------------------
      // Backend event count
      // ------------------------------------------------------

      let count = null;

      if (
        typeof data.total_events ===
        "number"
      ) {
        count =
          data.total_events;
      } else if (
        typeof data.event_count ===
        "number"
      ) {
        count =
          data.event_count;
      } else if (
        typeof data.events ===
        "number"
      ) {
        count =
          data.events;
      }

      if (count !== null) {
        backendEventCountRef.current =
          count;

        setBackendEventCount(
          count
        );
      }

      console.debug(
        "[CanthusCore] Frame uploaded:",
        {
          eventsAdded:
            data.events_added,
          totalEvents:
            data.total_events ??
            data.event_count ??
            data.events,
        }
      );
    } catch (err) {
      console.error(
        "[CanthusCore] Frame upload error:",
        err
      );
    } finally {
      uploadingFrameRef.current =
        false;
    }
  }

  // ==========================================================
  // START BACKEND STREAM
  // ==========================================================

  function startFrameStreaming() {
    stopFrameStreaming();

    // Immediate upload.
    uploadCurrentFrame();

    frameTimerRef.current =
      setInterval(() => {
        uploadCurrentFrame();
      }, FRAME_UPLOAD_INTERVAL);

    console.log(
      "[CanthusCore] Backend frame stream started."
    );
  }

  // ==========================================================
  // STOP BACKEND STREAM
  // ==========================================================

  function stopFrameStreaming() {
    if (frameTimerRef.current) {
      clearInterval(
        frameTimerRef.current
      );

      frameTimerRef.current =
        null;
    }

    // Do not modify uploadingFrameRef here.
    // A current request may still be completing.
  }

  // ==========================================================
  // CHALLENGE ANALYSIS
  // ==========================================================

  function processChallenge(
    landmarks
  ) {
    if (!challengeRef.current) {
      return;
    }

    if (!landmarks) {
      return;
    }

    const currentChallenge =
      challengeRef.current;

    // --------------------------------------------------------
    // BLINK
    // --------------------------------------------------------

    const leftEAR =
      eyeAspectRatio(
        landmarks,
        [
          33,
          160,
          158,
          133,
          153,
          144,
        ]
      );

    const rightEAR =
      eyeAspectRatio(
        landmarks,
        [
          362,
          385,
          387,
          263,
          373,
          380,
        ]
      );

    const ear =
      (leftEAR + rightEAR) /
      2;

    const blinkClosed =
      ear < 0.2;

    if (
      currentChallenge ===
      "blink_twice"
    ) {
      if (
        blinkClosed &&
        !blinkClosedRef.current
      ) {
        blinkClosedRef.current =
          true;
      }

      if (
        !blinkClosed &&
        blinkClosedRef.current
      ) {
        blinkClosedRef.current =
          false;

        blinkCountRef.current +=
          1;

        setBlinkCount(
          blinkCountRef.current
        );

        if (
          blinkCountRef.current >=
          2
        ) {
          setChallengeStatus(
            "passed"
          );

          setChallengeMessage(
            "✓ Two blinks detected"
          );
        } else {
          setChallengeMessage(
            `Blink ${blinkCountRef.current} / 2 detected`
          );
        }
      }
    }

    // --------------------------------------------------------
    // GAZE
    // --------------------------------------------------------

    const gaze =
      getGaze(landmarks);

    if (
      currentChallenge ===
        "saccade_left_right" ||
      currentChallenge ===
        "saccade_up_down"
    ) {
      if (
        gazeBaselineRef.current ===
        null
      ) {
        gazeBaselineRef.current =
          gaze;

        return;
      }

      const base =
        gazeBaselineRef.current;

      const dx =
        gaze.horizontal -
        base.horizontal;

      const dy =
        gaze.vertical -
        base.vertical;

      // ------------------------------------------------------
      // LEFT -> RIGHT
      // ------------------------------------------------------

      if (
        currentChallenge ===
        "saccade_left_right"
      ) {
        if (dx < -0.1) {
          gazeStateRef.current =
            "LEFT";

          setGazeState(
            "LEFT"
          );
        }

        if (
          gazeStateRef.current ===
            "LEFT" &&
          dx > 0.1
        ) {
          gazeStateRef.current =
            "RIGHT";

          setGazeState(
            "RIGHT"
          );

          setChallengeStatus(
            "passed"
          );

          setChallengeMessage(
            "✓ Left → Right detected"
          );
        }
      }

      // ------------------------------------------------------
      // UP -> DOWN
      // ------------------------------------------------------

      if (
        currentChallenge ===
        "saccade_up_down"
      ) {
        if (dy < -0.1) {
          gazeStateRef.current =
            "UP";

          setGazeState(
            "UP"
          );
        }

        if (
          gazeStateRef.current ===
            "UP" &&
          dy > 0.1
        ) {
          gazeStateRef.current =
            "DOWN";

          setGazeState(
            "DOWN"
          );

          setChallengeStatus(
            "passed"
          );

          setChallengeMessage(
            "✓ Up → Down detected"
          );
        }
      }
    }
  }

  // ==========================================================
  // PROCESS CAMERA
  // ==========================================================

  useEffect(() => {
    let running = true;

    function processVideo() {
      if (!running) {
        return;
      }

      const video =
        videoRef.current;

      const landmarker =
        faceLandmarkerRef.current;

      if (
        video &&
        landmarker &&
        video.readyState >= 2
      ) {
        const timestamp =
          performance.now();

        // ----------------------------------------------------
        // Generate simulated events
        // ----------------------------------------------------

        if (
          challengeRef.current &&
          challengeStatus !==
            "completed" &&
          challengeStatus !==
            "processing"
        ) {
          generateEvents(
            video,
            timestamp
          );
        }

        // ----------------------------------------------------
        // MediaPipe
        // ----------------------------------------------------

        if (
          video.currentTime !==
          lastVideoTimeRef.current
        ) {
          lastVideoTimeRef.current =
            video.currentTime;

          try {
            const result =
              landmarker.detectForVideo(
                video,
                timestamp
              );

            if (
              result.faceLandmarks &&
              result.faceLandmarks
                .length
            ) {
              processChallenge(
                result.faceLandmarks[0]
              );
            }
          } catch (err) {
            console.error(
              "[CanthusCore] Face detection error:",
              err
            );
          }
        }
      }

      animationRef.current =
        requestAnimationFrame(
          processVideo
        );
    }

    animationRef.current =
      requestAnimationFrame(
        processVideo
      );

    return () => {
      running = false;

      if (
        animationRef.current
      ) {
        cancelAnimationFrame(
          animationRef.current
        );
      }
    };
  }, [challengeStatus]);

  // ==========================================================
  // START CHALLENGE
  // ==========================================================

  async function startChallenge() {
    setError(null);
    setDecision(null);

    submittingRef.current =
      false;

    stopFrameStreaming();

    if (
      autoSubmitTimerRef.current
    ) {
      clearTimeout(
        autoSubmitTimerRef.current
      );

      autoSubmitTimerRef.current =
        null;
    }

    try {
      // ------------------------------------------------------
      // Request new session
      // ------------------------------------------------------

      const response =
        await fetch(
          `${API_BASE}/challenge`,
          {
            method: "POST",
          }
        );

      if (!response.ok) {
        throw new Error(
          `Backend returned ${response.status}`
        );
      }

      const data =
        await response.json();

      console.log(
        "[CanthusCore] Challenge started:",
        data
      );

      // ------------------------------------------------------
      // Session
      // ------------------------------------------------------

      setSession(data);

      setChallenge(
        data.challenge
      );

      challengeRef.current =
        data.challenge;

      challengeStartRef.current =
        performance.now();

      // ------------------------------------------------------
      // Reset local events
      // ------------------------------------------------------

      eventsRef.current = [];

      previousFrameRef.current =
        null;

      // ------------------------------------------------------
      // Reset backend count
      // ------------------------------------------------------

      backendEventCountRef.current =
        0;

      setBackendEventCount(0);

      setEventCount(0);

      // ------------------------------------------------------
      // Reset challenge
      // ------------------------------------------------------

      blinkCountRef.current =
        0;

      blinkClosedRef.current =
        false;

      gazeBaselineRef.current =
        null;

      gazeStateRef.current =
        "CENTER";

      setBlinkCount(0);

      setGazeState(
        "CENTER"
      );

      setChallengeStatus(
        "capturing"
      );

      setChallengeMessage(
        `Perform: ${
          CHALLENGE_NAMES[
            data.challenge
          ] || data.challenge
        }`
      );

      // ------------------------------------------------------
      // Start backend frame stream
      // ------------------------------------------------------

      setTimeout(() => {
        startFrameStreaming();
      }, 200);
    } catch (err) {
      console.error(
        "[CanthusCore] Start challenge error:",
        err
      );

      setError(
        `Could not start challenge: ${
          err.message
        }`
      );

      setChallengeStatus(
        "failed"
      );
    }
  }

  // ==========================================================
  // SUBMIT RESPONSE
  // ==========================================================

  async function submitResponse(
    automatic = false
  ) {
    if (!session?.session_id) {
      setError(
        "No active challenge session. Please start a new challenge."
      );

      return;
    }

    if (
      submittingRef.current
    ) {
      return;
    }

    submittingRef.current =
      true;

    setError(null);

    try {
      // ------------------------------------------------------
      // Stop regular frame timer
      // ------------------------------------------------------

      stopFrameStreaming();

      const elapsed =
        (performance.now() -
          challengeStartRef.current) /
        1000;

      // ------------------------------------------------------
      // Challenge result
      // ------------------------------------------------------

      let passed =
        challengeStatus ===
        "passed";

      if (
        elapsed >
        MAX_CAPTURE_SECONDS
      ) {
        passed = false;
      }

      // ------------------------------------------------------
      // Current event counts
      // ------------------------------------------------------

      const backendEvents =
        backendEventCountRef.current;

      const localEvents =
        eventsRef.current;

      console.log(
        "[CanthusCore] Submission:",
        {
          session:
            session.session_id,

          challenge,

          passed,

          elapsed,

          backendEvents,

          localEvents:
            localEvents.length,
        }
      );

      // ------------------------------------------------------
      // Not enough backend events
      // ------------------------------------------------------

      if (
        backendEvents <
        MIN_BACKEND_EVENTS
      ) {
        if (automatic) {
          setChallengeStatus(
            "capturing"
          );

          setChallengeMessage(
            `Capturing more motion... ${backendEvents} backend events`
          );

          submittingRef.current =
            false;

          startFrameStreaming();

          return;
        }

        setChallengeStatus(
          "capturing"
        );

        setError(
          `Not enough motion events reached the backend. ` +
          `Current events: ${backendEvents}. ` +
          `Minimum required: ${MIN_BACKEND_EVENTS}.`
        );

        submittingRef.current =
          false;

        startFrameStreaming();

        return;
      }

      // ------------------------------------------------------
      // Local event check
      // ------------------------------------------------------

      if (
        localEvents.length ===
        0
      ) {
        setChallengeStatus(
          "failed"
        );

        setError(
          "No local events were generated. Move your face slightly and try again."
        );

        submittingRef.current =
          false;

        return;
      }

      // ------------------------------------------------------
      // Processing
      // ------------------------------------------------------

      setChallengeStatus(
        "processing"
      );

      setChallengeMessage(
        "Processing event stream..."
      );

      // ------------------------------------------------------
      // Request body
      // ------------------------------------------------------

      const body = {
        session_id:
          session.session_id,

        challenge_passed:
          passed,

        challenge_score:
          passed ? 1.0 : 0.0,

        challenge_message:
          challengeMessage,

        // Compatibility/fallback event data.
        x: localEvents.map(
          (e) => e.x
        ),

        y: localEvents.map(
          (e) => e.y
        ),

        t: localEvents.map(
          (e) => e.t
        ),

        p: localEvents.map(
          (e) => e.p
        ),
      };

      console.log(
        "[CanthusCore] Sending /decision..."
      );

      // ------------------------------------------------------
      // AbortController timeout
      // ------------------------------------------------------

      const controller =
        new AbortController();

      const timeoutId =
        setTimeout(() => {
          controller.abort();
        }, DECISION_TIMEOUT_MS);

      let response;

      try {
        response =
          await fetch(
            `${API_BASE}/decision`,
            {
              method: "POST",

              headers: {
                "Content-Type":
                  "application/json",
              },

              body:
                JSON.stringify(body),

              signal:
                controller.signal,
            }
          );
      } finally {
        clearTimeout(
          timeoutId
        );
      }

      // ------------------------------------------------------
      // Parse response
      // ------------------------------------------------------

      const text =
        await response.text();

      let result;

      try {
        result =
          JSON.parse(text);
      } catch {
        result = {
          detail: text,
        };
      }

      console.log(
        "[CanthusCore] Backend response:",
        result
      );

      // ------------------------------------------------------
      // HTTP error
      // ------------------------------------------------------

      if (!response.ok) {
        throw new Error(
          result.detail ||
            `Backend returned ${response.status}`
        );
      }

      // ------------------------------------------------------
      // SUCCESS
      // ------------------------------------------------------

      setDecision(result);

      setChallengeStatus(
        result.challenge_passed ===
          false
          ? "failed"
          : "completed"
      );

      setChallengeMessage(
        result.challenge_passed ===
          false
          ? "✕ Challenge failed"
          : "✓ Challenge completed"
      );

      setError(null);
    } catch (err) {
      console.error(
        "[CanthusCore] Decision error:",
        err
      );

      if (
        err.name ===
        "AbortError"
      ) {
        setError(
          "The backend took more than 30 seconds to respond. " +
          "Check the FastAPI terminal for errors and try again."
        );
      } else {
        setError(
          `Decision failed: ${
            err.message ||
            "Unknown backend error"
          }`
        );
      }

      setChallengeStatus(
        "failed"
      );

      setChallengeMessage(
        "Decision failed. Please restart the challenge."
      );
    } finally {
      submittingRef.current =
        false;

      stopFrameStreaming();
    }
  }

  // ==========================================================
  // AUTOMATIC SUBMISSION
  // ==========================================================

  useEffect(() => {
    if (
      challengeStatus !==
      "passed"
    ) {
      return;
    }

    if (!session?.session_id) {
      return;
    }

    if (
      submittingRef.current
    ) {
      return;
    }

    if (
      autoSubmitTimerRef.current
    ) {
      clearTimeout(
        autoSubmitTimerRef.current
      );

      autoSubmitTimerRef.current =
        null;
    }

    const elapsed =
      (performance.now() -
        challengeStartRef.current) /
      1000;

    const remaining =
      Math.max(
        0,
        MIN_CAPTURE_SECONDS -
          elapsed
      );

    autoSubmitTimerRef.current =
      setTimeout(() => {
        if (
          !submittingRef.current
        ) {
          submitResponse(true);
        }
      }, remaining * 1000 + 500);

    return () => {
      if (
        autoSubmitTimerRef.current
      ) {
        clearTimeout(
          autoSubmitTimerRef.current
        );

        autoSubmitTimerRef.current =
          null;
      }
    };
  }, [
    challengeStatus,
    session,
  ]);

  // ==========================================================
  // RESET
  // ==========================================================

  function restart() {
    stopFrameStreaming();

    if (
      autoSubmitTimerRef.current
    ) {
      clearTimeout(
        autoSubmitTimerRef.current
      );

      autoSubmitTimerRef.current =
        null;
    }

    submittingRef.current =
      false;

    challengeRef.current =
      null;

    setSession(null);

    setChallenge(null);

    setDecision(null);

    setChallengeStatus(
      "idle"
    );

    setChallengeMessage(
      "Press Start Challenge"
    );

    eventsRef.current = [];

    previousFrameRef.current =
      null;

    blinkCountRef.current =
      0;

    blinkClosedRef.current =
      false;

    gazeBaselineRef.current =
      null;

    gazeStateRef.current =
      "CENTER";

    backendEventCountRef.current =
      0;

    setBlinkCount(0);

    setGazeState(
      "CENTER"
    );

    setEventCount(0);

    setBackendEventCount(0);

    setError(null);

    console.log(
      "[CanthusCore] Challenge reset."
    );
  }

  // ==========================================================
  // CLEANUP
  // ==========================================================

  useEffect(() => {
    return () => {
      stopFrameStreaming();

      if (
        autoSubmitTimerRef.current
      ) {
        clearTimeout(
          autoSubmitTimerRef.current
        );
      }

      if (
        animationRef.current
      ) {
        cancelAnimationFrame(
          animationRef.current
        );
      }

      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) =>
            track.stop()
          );
      }

      if (
        faceLandmarkerRef.current
      ) {
        faceLandmarkerRef.current.close();
      }
    };
  }, []);

  // ==========================================================
  // ACTIVITY CHART
  // ==========================================================

  const chartData =
    decision?.activity_profile?.map(
      (value, index) => ({
        bin: index + 1,
        activity: value,
      })
    ) || [];

  // ==========================================================
  // ATTACH CAMERA ON VIEW CHANGE
  // ==========================================================
  useEffect(() => {
    if (view === "live" && videoRef.current && streamRef.current) {
      if (videoRef.current.srcObject !== streamRef.current) {
        videoRef.current.srcObject = streamRef.current;
        videoRef.current.play().catch(e => console.error("Play error", e));
        setCameraReady(true);
      }
    }
  }, [view]);

  // ==========================================================
  // RENDER ROUTING
  // ==========================================================

  if (view === "landing") {
    return <Landing onEnter={() => setView("live")} />;
  }

  return (
    <LiveDemo
      videoRef={videoRef}
      canvasRef={canvasRef}
      cameraReady={cameraReady}
      modelReady={modelReady}
      session={session}
      challenge={challenge}
      challengeStatus={challengeStatus}
      challengeMessage={challengeMessage}
      blinkCount={blinkCount}
      gazeState={gazeState}
      eventCount={eventCount}
      backendEventCount={backendEventCount}
      decision={decision}
      error={error}
      startChallenge={startChallenge}
      submitResponse={submitResponse}
      restart={restart}
    />
  );
}
