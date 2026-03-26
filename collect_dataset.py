import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import cv2
import mediapipe as mp
import csv
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

file = open("dataset/dataset.csv", mode='a', newline='')
writer = csv.writer(file)

label = None
recording = False
start_time = None
delay = 3 # seconds before recording starts.
record_start_time = None
record_duration = 7
sample_count = 0
with mp_hands.Hands(
     static_image_mode=False,
     max_num_hands=2,
     min_detection_confidence=0.5
) as hands:

     while True:
          key = cv2.waitKey(1) & 0xFF

          if key == ord('p'): #play
               label = 'play'
               print("play started")
          elif key == ord('u'): #volume up
               label = 'volume_up'
          elif key == ord('a'): #pause music
               label = 'pause'
          elif key == ord('d'):
               label = 'volume_down' #volume low
          elif key == ord('n'):
               label = 'next' # next song
          elif key == ord('r'):
               start_time = time.time()
               recording = False
               print("Countdown started.")
          
          #read frame
          ret, frame = cap.read()
          if not ret:
               break
          
          height, width, _ = frame.shape
          frame = cv2.flip(frame,1)
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

          results = hands.process(frame_rgb)

          print("Recording:", recording, "| Label:", label, "| Hands detected:", results.multi_hand_landmarks is not None)
          left_hand = [0]*63
          right_hand = [0]*63

          if start_time is not None:
               elapsed = time.time() - start_time
               if elapsed >= delay:
                    if not recording:
                         recording = True
                         record_start_time = time.time()
               else:
                    cv2.putText(frame, f"Starting in {int(delay - elapsed)}...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
          #detect hands
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
          
          if (recording 
               and label 
               and record_start_time 
               and (any(left_hand) or any(right_hand))):               
               record_elapsed = time.time() - record_start_time
               remaining = int(record_duration - (time.time() - record_start_time))
               cv2.putText(
                    frame,
                    f"Recording... {remaining}s",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
               )
               if record_elapsed >= record_duration:
                    recording = False 
                    start_time = None
                    record_start_time = None
               else:
                    hand_flag = 0
                    if any(left_hand) and any(right_hand):
                         hand_flag = 2
                    elif any(left_hand):
                         hand_flag = 1
                    elif any(right_hand):
                         hand_flag = -1
                    row = left_hand + right_hand + [hand_flag] + [label]
                    writer.writerow(row)
                    sample_count += 1
                    print(f"Saved {sample_count} samples for {label}")
          
          cv2.imshow("Frame", frame)
          if key == 27: #press ESC to stop execution.
               break


cap.release()
file.close()
cv2.destroyAllWindows()