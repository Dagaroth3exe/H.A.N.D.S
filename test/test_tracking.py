import cv2 
import mediapipe as mp

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

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks :
        for hands_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hands_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("H.A.N.D.S - Tracking Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q") : 
        break

cap.release()
cv2.destroyAllWindows()



