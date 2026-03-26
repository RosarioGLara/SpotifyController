import cv2
import mediapipe as mp
import joblib
import numpy as np
from spotify_control import SpotifyController
import time

last_action_time = 0
cooldown = 2  # seconds

# Load trained components
model = joblib.load("models/gesture_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
spotify = SpotifyController()
gesture_map = {
    "play": spotify.play,
    "pause": spotify.pause,
    "next": spotify.next_song,
    "volume_up": spotify.volume_up,
    "volume_down": spotify.volume_down
}

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
) as hands:

     while True:
          ret, frame = cap.read()
          if not ret:
               break

          frame = cv2.flip(frame, 1)
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

          results = hands.process(frame_rgb)

          left_hand = [0]*63
          right_hand = [0]*63

          # Detect hands
          if results.multi_hand_landmarks is not None:
               for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                         frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    hand_type = results.multi_handedness[i].classification[0].label
                    data = []

                    for lm in hand_landmarks.landmark:
                         data.extend([lm.x, lm.y, lm.z])

                    if hand_type == "Left":
                         left_hand = data
                    elif hand_type == "Right":
                         right_hand = data

               # Only predict if BOTH hands detected
               if (any(left_hand) or any(right_hand)):
                    hand_flag = 0
                    if any(left_hand) and any(right_hand):
                         hand_flag = 2
                    elif any(left_hand):
                         hand_flag = 1
                    elif any(right_hand):
                         hand_flag = -1
                    sample = np.array(left_hand + right_hand + [hand_flag]).reshape(1, -1)

                    # Apply same scaling
                    sample = scaler.transform(sample)

                    prediction = model.predict(sample)
                    label = encoder.inverse_transform(prediction)[0]

                    # Show prediction
                    cv2.putText(
                         frame,
                         f"Gesture: {label}",
                         (50, 50),
                         cv2.FONT_HERSHEY_SIMPLEX,
                         1,
                         (0, 255, 0),
                         2
                    )
                    if label in gesture_map and (time.time() - last_action_time > cooldown):
                         try:
                              gesture_map[label]()
                              last_action_time = time.time()
                         except Exception as e:
                              print("Spotify error:", e)
          cv2.imshow("Gesture Control", frame)

          if cv2.waitKey(1) & 0xFF == 27:
               break

cap.release()
cv2.destroyAllWindows()

