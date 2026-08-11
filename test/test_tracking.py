import cv2 
import mediapipe as mp
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.5
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open camera")
else:
    print("Camera opened successfully")

on_desktop = False  # our own source of truth for current state

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_visible = False
    is_fist = False
    is_open = False

    if result.multi_hand_landmarks:
        hand_visible = True
        for hands_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hands_landmarks, mp_hands.HAND_CONNECTIONS)

            # Pinch detection
            thumb_tip = hands_landmarks.landmark[4]
            index_tip = hands_landmarks.landmark[8]
            distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
            if distance < 0.05:
                print("PINCH DETECTED")

            # Fist / open-palm detection
            tips = [8, 12, 16, 20]
            pips = [6, 10, 14, 18]
            folded = sum(
                1 for tip, pip in zip(tips, pips)
                if hands_landmarks.landmark[tip].y > hands_landmarks.landmark[pip].y
            )
            is_fist = (folded == 4)
            is_open = (folded == 0)

    # Only act on EXPLICIT hand states. If hand isn't visible, do nothing —
    # stay in whatever state we're already in.
    if hand_visible and is_fist and not on_desktop:
        print("FIST CLOSED - showing desktop")
        pyautogui.hotkey('win', 'd')
        on_desktop = True
    elif hand_visible and is_open and on_desktop:
        print("HAND OPEN - restoring window")
        pyautogui.hotkey('win', 'd')
        on_desktop = False

    cv2.imshow("H.A.N.D.S - Tracking Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()