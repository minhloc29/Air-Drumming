import cv2
import mediapipe as mp
import time
from threading import Thread
import matplotlib.pyplot as plt
import numpy as np
import math
import simpleaudio as sa
from param import SOUNDS, upper_lip_index, lower_lip_index
from my_utils import play_sound_async, calculate_distance

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2,  # Explicitly set for 2-hand tracking
    model_complexity = 0
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    # Convert BGR to RGB
    hand_results = hands.process(frame)
    if hand_results.multi_hand_landmarks:    
        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Get hand information
            hand_label = handedness.classification[0].label
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_tip.x * w), int(index_tip.y * h)
            cv2.circle(frame, (x, y), 30, (255, 0, 0), -1)
            #print(hand_label, index_tip)

            for sound in SOUNDS[:-1]:
                if sound['hand'] == hand_label:
                    x1, y1, x2, y2 = sound['rect']
                    inside = x1 <= x <= x2 and y1 <= y <= y2

                    # Trigger sound on entry and update state
                    if inside and not sound['state']:
                        play_sound_async(sound['sound'])
                        sound['state'] = True
                    elif not inside and sound['state']:
                        sound['state'] = False  # Reset state on exit
    
    for sound in SOUNDS[:-1]:
        x1, y1, x2, y2 = sound['rect']
        sound_img = sound['image']
        roi = frame[y1:y2, x1:x2]
        alpha = sound_img[:, :, 3] / 255.0
        alpha = cv2.merge([alpha, alpha, alpha])
        
        # Composite the images
        foreground = cv2.multiply(alpha, sound_img[:, :, :3].astype(float))
        background = cv2.multiply(1.0 - alpha, roi.astype(float))
        frame[y1:y2, x1:x2] = cv2.add(foreground, background).astype(np.uint8)
        # color = (0, 255, 0) if sound['state'] else (0, 0, 255)
        # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # cv2.putText(frame, f"{sound['name']} ({sound['hand'][0]})", 
        #            (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    status = "No mouth detected!"
    face_results = face_mesh.process(frame)
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Extract the landmarks for the upper and lower lips.
            upper_lip = face_landmarks.landmark[upper_lip_index]
            lower_lip = face_landmarks.landmark[lower_lip_index]
            
            # Calculate the vertical distance between the lips.
            lip_distance = calculate_distance(upper_lip, lower_lip)
            
            # Define a threshold (this value might need tuning).
            threshold = 0.02  # Adjust based on your testing and camera settings.
            
            if lip_distance > threshold and not SOUNDS[-1]['state']:
                play_sound_async(SOUNDS[-1]['sound'])
                SOUNDS[-1]['state'] = True
                status = "Mouth Open"
                
            elif lip_distance <= threshold and SOUNDS[-1]['state']:
                status = "Mouth Closed"
                SOUNDS[-1]['state'] = False
                status = "Mouth Closed"
        
        
            # Optionally, display the status on the frame.
            cv2.putText(frame, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)                
    cv2.imshow("Webcam", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()