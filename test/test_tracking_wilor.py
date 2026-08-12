"""
WiLoR-backed version of test_tracking.py. Same gestures (swipe left/right,
point-to-jump-desktop), but hand tracking comes from WiLoR instead of
MediaPipe. WiLoR's 21 joints follow the same wrist + 4-per-finger ordering
as MediaPipe/OpenPose hands, so the finger extended/folded heuristics below
are unchanged from the MediaPipe version.

This needs its own environment - WiLoR requires Python 3.10, PyTorch, and a
GPU. Run with:
    source wilor-env/bin/activate
    python test/test_tracking_wilor.py
"""
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyautogui
import torch

import gesture_calibration as calib

WILOR_DIR = Path(__file__).resolve().parent.parent / "third_party" / "WiLoR"
sys.path.insert(0, str(WILOR_DIR))
# WiLoR's model config bakes in relative paths (e.g. ./mano_data/...) that only
# resolve against the repo's own directory, same as running `cd WiLoR && python demo.py`.
os.chdir(WILOR_DIR)

from wilor.models import load_wilor  # noqa: E402
from wilor.utils import recursive_to  # noqa: E402
from wilor.datasets.vitdet_dataset import ViTDetDataset  # noqa: E402
from wilor.utils.renderer import cam_crop_to_full  # noqa: E402
from ultralytics import YOLO  # noqa: E402

FAST = False  # fp16 + backbone compilation + block skipping (WiLoR's --fast path); off trades speed for accuracy
RESCALE_FACTOR = 2.3  # bbox padding - a bit more room than the demo.py default so extended fingers don't clip the crop edge
DETECTOR_CONF = 0.5  # hand-detector confidence threshold - higher rejects ambiguous/low-quality detections
CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Loading WiLoR on {device}...")

model, model_cfg = load_wilor(
    checkpoint_path=str(WILOR_DIR / 'pretrained_models' / 'wilor_final.ckpt'),
    cfg_path=str(WILOR_DIR / 'pretrained_models' / 'model_config.yaml'),
)
if FAST:
    torch.set_float32_matmul_precision('high')
    model = model.half()
    model.backbone = torch.compile(model.backbone)
    model.backbone.skip_blocks = True
model = model.to(device)
model.eval()

detector = YOLO(str(WILOR_DIR / 'pretrained_models' / 'detector.pt'))
detector.to(device)

print("Model loaded. The first detected hand will be slow (one-time backbone compile).")


def project_points(points_3d, cam_t, focal_length, img_w, img_h):
    cx, cy = img_w / 2.0, img_h / 2.0
    p = points_3d + cam_t
    x = p[:, 0] / p[:, 2] * focal_length + cx
    y = p[:, 1] / p[:, 2] * focal_length + cy
    return np.stack([x, y], axis=1)


def get_hand_keypoints(frame):
    """Runs WiLoR on an already-mirrored BGR frame. Returns a 21x2 array of
    pixel-space keypoints for the most confident detected hand, or None."""
    detections = detector(frame, conf=DETECTOR_CONF, verbose=False)[0]
    if len(detections) == 0:
        return None

    confs = detections.boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))
    bbox = detections.boxes.data.cpu().numpy()[best][:4]
    is_right = float(detections.boxes.cls.cpu().numpy()[best])

    boxes = np.array([bbox])
    right = np.array([is_right])
    dataset = ViTDetDataset(model_cfg, frame, boxes, right, rescale_factor=RESCALE_FACTOR, fp16=FAST)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = recursive_to(next(iter(dataloader)), device)

    with torch.no_grad():
        out = model(batch)

    pred_cam = out['pred_cam'].clone()
    pred_cam[:, 1] = (2 * batch['right'] - 1) * pred_cam[:, 1]
    box_center = batch['box_center'].float()
    box_size = batch['box_size'].float()
    img_size = batch['img_size'].float()
    scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    cam_t_full = cam_crop_to_full(
        pred_cam, box_center, box_size, img_size, scaled_focal_length
    ).detach().cpu().numpy()[0]

    joints = out['pred_keypoints_3d'][0].detach().float().cpu().numpy()
    is_r = batch['right'][0].cpu().numpy()
    joints[:, 0] = (2 * is_r - 1) * joints[:, 0]

    return project_points(joints, cam_t_full, float(scaled_focal_length), frame.shape[1], frame.shape[0])


def get_landmarks(frame):
    """frame: already-mirrored BGR. Returns 21x2 normalized (x, y) or None."""
    kpts = get_hand_keypoints(frame)
    if kpts is None:
        return None
    return kpts / [frame.shape[1], frame.shape[0]]


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
if not cap.isOpened():
    print("ERROR: Could not open camera")
else:
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened successfully ({actual_w}x{actual_h})")

thresholds = calib.get_or_run_calibration(cap, get_landmarks, force='--calibrate' in sys.argv)

# Swipe (workspace switch) tuning
SWIPE_HISTORY_SECONDS = 0.4   # how far back we look for wrist movement
SWIPE_DISTANCE_THRESHOLD = 0.25  # normalized x-distance needed to count as a swipe
SWIPE_COOLDOWN_SECONDS = 1.0  # min time between two swipe triggers

wrist_history = deque()  # (timestamp, x) of the wrist, most recent last
last_swipe_time = 0.0

# Point (jump to desktop 1 / next desktop) tuning
JUMP_TO_FIRST_DESKTOP_PRESSES = 8  # more "go left" presses than anyone has workspaces, so this reliably lands on the leftmost one
POINT_COOLDOWN_SECONDS = 1.0  # min time between two point triggers
POINT_CONFIRM_FRAMES = 2  # consecutive frames the raw pose must hold before we trust it as a deliberate point, not a hand mid-transition

last_point_time = 0.0
last_point_side = None  # 'left' or 'right', which side of the frame we last triggered on
was_pointing = False  # previous frame's confirmed pointing state
point_streak = 0  # consecutive frames the raw (unconfirmed) pointing pose has held

pyautogui.PAUSE = 0  # pyautogui defaults to a 0.1s pause after every call, which would tank our frame rate

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]

    kpts = get_hand_keypoints(frame)

    hand_visible = False
    is_pointing = False
    wrist_x = None
    index_x_norm = None

    if kpts is not None:
        hand_visible = True
        landmarks = kpts / [frame_w, frame_h]  # normalize to 0-1, same convention as MediaPipe

        for x, y in kpts.astype(int):
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        wrist_x = landmarks[0][0]
        index_x_norm = landmarks[8][0]

        # Per-finger extended state using this user's calibrated thresholds
        # (falls back to a sane default if calibration is missing a finger).
        metrics = calib.extended_metrics(landmarks)
        index_extended = metrics['index'] > thresholds.get('index', calib.DEFAULT_THRESHOLD)
        middle_extended = metrics['middle'] > thresholds.get('middle', calib.DEFAULT_THRESHOLD)
        ring_extended = metrics['ring'] > thresholds.get('ring', calib.DEFAULT_THRESHOLD)
        pinky_extended = metrics['pinky'] > thresholds.get('pinky', calib.DEFAULT_THRESHOLD)

        # Pointing: index finger up, other three clearly folded
        raw_pointing = index_extended and not middle_extended and not ring_extended and not pinky_extended
        point_streak = point_streak + 1 if raw_pointing else 0
        # Only trust it once the pose has held steady for several frames in a
        # row - filters out the hand briefly passing through this shape while
        # transitioning between other gestures or during ordinary movement.
        is_pointing = point_streak >= POINT_CONFIRM_FRAMES
    else:
        point_streak = 0  # hand left the frame - don't let a stale streak carry over

    # Point with your index finger on the left half of the camera frame -> jump
    # straight to desktop 1 (leftmost). Point on the right half -> move one
    # desktop to the right. Re-triggers if you keep pointing but swap sides,
    # gated by a cooldown so it doesn't rapid-fire near the frame's midline.
    now = time.time()
    if hand_visible and is_pointing and index_x_norm is not None:
        side = 'left' if index_x_norm < 0.5 else 'right'
        if (side != last_point_side or not was_pointing) and now - last_point_time > POINT_COOLDOWN_SECONDS:
            if side == 'left':
                print("POINT LEFT - jump to desktop 1")
                for _ in range(JUMP_TO_FIRST_DESKTOP_PRESSES):
                    pyautogui.hotkey('ctrl', 'alt', 'left')
                    time.sleep(0.05)
            else:
                print("POINT RIGHT - next desktop")
                pyautogui.hotkey('ctrl', 'alt', 'right')
            last_point_time = now
            last_point_side = side
            wrist_history.clear()
    else:
        last_point_side = None
    was_pointing = is_pointing if hand_visible else False

    # Swipe left/right -> switch to the workspace/desktop on that side.
    # We track the wrist position over a short rolling window and look for a
    # fast, large horizontal displacement within it. Excludes the pointing
    # shape so it doesn't double up with the point gesture above.
    if hand_visible and not is_pointing and wrist_x is not None:
        wrist_history.append((now, wrist_x))
        while wrist_history and now - wrist_history[0][0] > SWIPE_HISTORY_SECONDS:
            wrist_history.popleft()

        if now - last_swipe_time > SWIPE_COOLDOWN_SECONDS and len(wrist_history) >= 2:
            dx = wrist_x - wrist_history[0][1]
            if dx > SWIPE_DISTANCE_THRESHOLD:
                print("SWIPE RIGHT - next desktop")
                pyautogui.hotkey('ctrl', 'alt', 'right')
                last_swipe_time = now
                wrist_history.clear()
            elif dx < -SWIPE_DISTANCE_THRESHOLD:
                print("SWIPE LEFT - previous desktop")
                pyautogui.hotkey('ctrl', 'alt', 'left')
                last_swipe_time = now
                wrist_history.clear()
    else:
        wrist_history.clear()

    cv2.imshow("H.A.N.D.S - Tracking Test (WiLoR)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
