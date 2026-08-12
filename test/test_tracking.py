import subprocess
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands = 2,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera")
else:
    print("Camera opened successfully")


def process_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return hands.process(rgb)


def change_volume(delta_percent):
    """delta_percent: e.g. '+5%' or '-5%'. Goes straight through PulseAudio/
    PipeWire via pactl - GNOME's Wayland compositor grabs XF86Audio* media
    keys directly from hardware input, so synthetic key presses injected via
    XTest/Xwayland (what pyautogui.press() uses) never reach it, even though
    the same injection path works fine for normal hotkeys like Ctrl+Alt+Left."""
    subprocess.run(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', delta_percent], check=False)


# Two-hand "stop" tuning. We can't reliably detect actual crossed/interlaced
# fingers - clasped hands heavily occlude each other from the camera, which
# is a known weak spot for landmark-based tracking. Instead we detect the
# practical equivalent: both hands present and brought close together, held
# briefly, since that's inherent to the gesture anyway.
JOIN_DISTANCE_THRESHOLD = 0.15  # normalized distance between hand centroids to count as "joined"
JOIN_HOLD_SECONDS = 1.0  # how long hands must stay joined before it triggers
JOIN_CONFIRM_FRAMES = 3  # consecutive frames both hands must read as joined, filters out a momentary false read

join_streak = 0
join_hold_start = None

# Pinch-and-move volume tuning. Pinch (thumb-to-index tip distance) is just a
# single-point distance check, not multi-finger detail, so it stays reliable
# even though the fingers touch.
PINCH_DISTANCE_THRESHOLD = 0.05  # thumb-to-index distance (normalized) that counts as a pinch
PINCH_MOVE_STEP = 0.03  # normalized y-movement per volume step
VOLUME_STEP_PERCENT = 5  # volume change per step

prev_pinch_y = None  # pinch y-position when we last fired a volume step, while the pinch is held

pyautogui.PAUSE = 0  # pyautogui defaults to a 0.1s pause after every call, which would tank our frame rate

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    result = process_frame(frame)
    all_hands = (
        [np.array([[p.x, p.y] for p in hand.landmark]) for hand in result.multi_hand_landmarks]
        if result.multi_hand_landmarks else []
    )
    landmarks = all_hands[0] if all_hands else None

    hand_visible = False
    is_pinching = False
    pinch_y = None

    if landmarks is not None:
        hand_visible = True

        # Pinch: thumb tip touching index tip. A single distance check between
        # two points, not multi-finger detail, so it stays reliable even
        # though the fingers are touching.
        pinch_distance = np.linalg.norm(landmarks[4] - landmarks[8])
        is_pinching = pinch_distance < PINCH_DISTANCE_THRESHOLD
        pinch_y = (landmarks[4][1] + landmarks[8][1]) / 2

    now = time.time()

    # Pinch and move up/down -> volume up/down. Each step of movement fires
    # one key press, so speed of movement doesn't matter, only distance -
    # move further to get more steps, not faster.
    if hand_visible and is_pinching and pinch_y is not None:
        if prev_pinch_y is not None:
            dy = pinch_y - prev_pinch_y  # positive = pinch moved down the frame
            if dy < -PINCH_MOVE_STEP:
                print("PINCH UP - volume up")
                change_volume(f'+{VOLUME_STEP_PERCENT}%')
                prev_pinch_y = pinch_y
            elif dy > PINCH_MOVE_STEP:
                print("PINCH DOWN - volume down")
                change_volume(f'-{VOLUME_STEP_PERCENT}%')
                prev_pinch_y = pinch_y
        else:
            prev_pinch_y = pinch_y
    else:
        prev_pinch_y = None

    # Join both hands together and hold -> stop. Approximates "cross your
    # fingers" (see note above on why we key off hand proximity, not the
    # fingers themselves).
    hands_joined_raw = False
    if len(all_hands) == 2:
        centroid_a = all_hands[0].mean(axis=0)
        centroid_b = all_hands[1].mean(axis=0)
        hands_joined_raw = np.linalg.norm(centroid_a - centroid_b) < JOIN_DISTANCE_THRESHOLD

    join_streak = join_streak + 1 if hands_joined_raw else 0
    hands_joined = join_streak >= JOIN_CONFIRM_FRAMES

    if hands_joined:
        if join_hold_start is None:
            join_hold_start = now
        elif now - join_hold_start >= JOIN_HOLD_SECONDS:
            print("HANDS JOINED (held) - stopping")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)
    else:
        join_hold_start = None

    cv2.imshow("H.A.N.D.S - Tracking Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
