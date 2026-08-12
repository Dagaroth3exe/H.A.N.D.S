"""
Shared per-user calibration for finger extended/folded detection.

Both test_tracking.py (MediaPipe) and test_tracking_wilor.py (WiLoR) produce
the same shape of data: 21 normalized (x, y) landmarks in the standard
wrist + 4-joints-per-finger order. This module doesn't care which backend
produced them, so calibration only has to be written once.

For each finger we measure `reference.y - tip.y`: positive when the tip is
above its reference joint (extended), negative when curled below it
(folded). Calibration shows you a FIST (all folded) and an OPEN hand (all
extended), records that metric for each, and sets each finger's personal
threshold at the midpoint between the two - replacing the fixed margins that
used to be hardcoded per gesture.
"""
import json
import time
from pathlib import Path

import cv2

CALIBRATION_PATH = Path(__file__).resolve().parent / "gesture_calibration.json"

# finger -> (reference joint, tip joint), same landmark indices used by the
# tracking scripts' finger-state checks
FINGER_JOINTS = {
    'index': (5, 8),
    'middle': (10, 12),
    'ring': (14, 16),
    'pinky': (18, 20),
}

HOLD_SECONDS = 3.0
DEFAULT_THRESHOLD = 0.03  # fallback if a pose never got a hand in frame


def extended_metrics(landmarks):
    """landmarks: 21x2 normalized (x, y). Returns {finger: metric}, positive
    meaning the tip is above (extended relative to) its reference joint."""
    return {
        finger: landmarks[ref][1] - landmarks[tip][1]
        for finger, (ref, tip) in FINGER_JOINTS.items()
    }


def _collect_pose(cap, get_landmarks, label):
    samples = {finger: [] for finger in FINGER_JOINTS}
    start = time.time()
    while time.time() - start < HOLD_SECONDS:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        landmarks = get_landmarks(frame)
        remaining = HOLD_SECONDS - (time.time() - start)
        cv2.putText(frame, f"Hold: {label}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"{remaining:0.1f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        if landmarks is not None:
            for finger, value in extended_metrics(landmarks).items():
                samples[finger].append(value)
            cv2.putText(frame, "hand detected", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "no hand detected", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("H.A.N.D.S - Calibration", frame)
        cv2.waitKey(1)

    return samples


def run_calibration(cap, get_landmarks):
    """Interactive two-pose calibration (fist, then open hand). Returns and
    saves {finger: threshold}."""
    print("Calibration: make a FIST and hold it...")
    fist_samples = _collect_pose(cap, get_landmarks, "make a FIST")
    print("Calibration: open your hand FLAT and hold it...")
    open_samples = _collect_pose(cap, get_landmarks, "open hand FLAT")
    cv2.destroyWindow("H.A.N.D.S - Calibration")

    thresholds = {}
    for finger in FINGER_JOINTS:
        fist_vals = fist_samples[finger]
        open_vals = open_samples[finger]
        if not fist_vals or not open_vals:
            print(f"  WARNING: no hand seen during one of the poses for '{finger}', using default threshold")
            thresholds[finger] = DEFAULT_THRESHOLD
            continue
        fist_mean = sum(fist_vals) / len(fist_vals)
        open_mean = sum(open_vals) / len(open_vals)
        thresholds[finger] = (fist_mean + open_mean) / 2.0
        print(f"  {finger}: fist={fist_mean:.3f}  open={open_mean:.3f}  -> threshold={thresholds[finger]:.3f}")

    save_calibration(thresholds)
    print(f"Calibration saved to {CALIBRATION_PATH}")
    return thresholds


def load_calibration():
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    return None


def save_calibration(thresholds):
    with open(CALIBRATION_PATH, 'w') as f:
        json.dump(thresholds, f, indent=2)


def get_or_run_calibration(cap, get_landmarks, force=False):
    if not force:
        existing = load_calibration()
        if existing is not None:
            print(f"Loaded existing calibration from {CALIBRATION_PATH} (pass --calibrate to redo it)")
            return existing
    return run_calibration(cap, get_landmarks)
