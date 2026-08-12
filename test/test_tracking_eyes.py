"""
Eye-gaze desktop switching, using MediaPipe Face Mesh's iris landmarks
(refine_landmarks=True). No hand tracking runs here - this is a standalone
mode, separate from test_tracking.py / test_tracking_wilor.py.

Look left and hold your gaze there for 2 seconds -> previous desktop.
Look right and hold for 2 seconds -> next desktop.

First run (or with --calibrate) walks you through looking left and right so
it can learn your personal gaze range instead of guessing fixed thresholds.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import pyautogui

GAZE_CALIBRATION_PATH = Path(__file__).resolve().parent / "gaze_calibration.json"
CALIBRATION_HOLD_SECONDS = 3.0
HOLD_SECONDS = 2.0  # how long a held gaze must last before it triggers

# landmark indices from MediaPipe's iris-refined face mesh
LEFT_EYE_CORNERS = (362, 263)
RIGHT_EYE_CORNERS = (33, 133)
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # required for iris landmarks
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


def get_gaze_x(frame):
    """frame: already-mirrored BGR. Returns a 0..1 gaze ratio (mean of both
    eyes' iris position between their corners) or None if no face found."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0].landmark
    ratios = []
    for (c1, c2), iris in ((LEFT_EYE_CORNERS, LEFT_IRIS_CENTER), (RIGHT_EYE_CORNERS, RIGHT_IRIS_CENTER)):
        x1, x2 = lm[c1].x, lm[c2].x
        span_min, span_max = min(x1, x2), max(x1, x2)
        if span_max - span_min < 1e-6:
            continue
        ratios.append((lm[iris].x - span_min) / (span_max - span_min))

    return sum(ratios) / len(ratios) if ratios else None


def _collect_gaze(cap, label, seconds):
    samples = []
    start = time.time()
    while time.time() - start < seconds:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        gaze_x = get_gaze_x(frame)
        remaining = seconds - (time.time() - start)
        cv2.putText(frame, f"Hold: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"{remaining:0.1f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        if gaze_x is not None:
            samples.append(gaze_x)
            cv2.putText(frame, "face detected", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "no face detected", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("H.A.N.D.S - Eye Calibration", frame)
        cv2.waitKey(1)

    return samples


def run_gaze_calibration(cap):
    print("Calibration: LOOK LEFT and hold...")
    left_samples = _collect_gaze(cap, "LOOK LEFT", CALIBRATION_HOLD_SECONDS)
    print("Calibration: LOOK RIGHT and hold...")
    right_samples = _collect_gaze(cap, "LOOK RIGHT", CALIBRATION_HOLD_SECONDS)
    cv2.destroyWindow("H.A.N.D.S - Eye Calibration")

    if not left_samples or not right_samples:
        print("  WARNING: no face seen during calibration, using default thresholds")
        data = {"left_mean": 0.35, "right_mean": 0.65}
    else:
        data = {
            "left_mean": sum(left_samples) / len(left_samples),
            "right_mean": sum(right_samples) / len(right_samples),
        }
    print(f"  left={data['left_mean']:.3f}  right={data['right_mean']:.3f}")
    save_gaze_calibration(data)
    print(f"Calibration saved to {GAZE_CALIBRATION_PATH}")
    return data


def load_gaze_calibration():
    if GAZE_CALIBRATION_PATH.exists():
        with open(GAZE_CALIBRATION_PATH) as f:
            return json.load(f)
    return None


def save_gaze_calibration(data):
    with open(GAZE_CALIBRATION_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def get_or_run_gaze_calibration(cap, force=False):
    if not force:
        existing = load_gaze_calibration()
        if existing is not None:
            print(f"Loaded existing gaze calibration from {GAZE_CALIBRATION_PATH} (pass --calibrate to redo it)")
            return existing
    return run_gaze_calibration(cap)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera")
else:
    print("Camera opened successfully")

cal = get_or_run_gaze_calibration(cap, force='--calibrate' in sys.argv)
left_mean = cal['left_mean']
right_mean = cal['right_mean']
center = (left_mean + right_mean) / 2
deadzone = abs(right_mean - left_mean) * 0.15
left_threshold, right_threshold = center - deadzone, center + deadzone
if right_threshold <= left_threshold:  # degenerate calibration (e.g. no face seen), fall back to a small neutral zone
    left_threshold, right_threshold = center - 0.05, center + 0.05

hold_side = None  # 'left' or 'right' - which side the gaze is currently held toward
hold_start = None
triggered_this_hold = False  # fires once per hold, not every frame while held

pyautogui.PAUSE = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    gaze_x = get_gaze_x(frame)
    now = time.time()

    if gaze_x is not None:
        if gaze_x < left_threshold:
            side = 'left'
        elif gaze_x > right_threshold:
            side = 'right'
        else:
            side = 'center'
    else:
        side = None

    if side in ('left', 'right'):
        if side != hold_side:
            hold_side = side
            hold_start = now
            triggered_this_hold = False

        hold_duration = now - hold_start
        cv2.putText(
            frame, f"{side.upper()} {hold_duration:0.1f}/{HOLD_SECONDS:0.1f}s",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
        )

        if hold_duration >= HOLD_SECONDS and not triggered_this_hold:
            if side == 'left':
                print("GAZE LEFT (held) - previous desktop")
                pyautogui.hotkey('ctrl', 'alt', 'left')
            else:
                print("GAZE RIGHT (held) - next desktop")
                pyautogui.hotkey('ctrl', 'alt', 'right')
            triggered_this_hold = True  # look away and back to fire again
    else:
        hold_side = None
        hold_start = None
        triggered_this_hold = False

    cv2.imshow("H.A.N.D.S - Tracking Test (Eyes)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
