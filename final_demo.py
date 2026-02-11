import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


model_path = "gesture_model.task"

BaseOptions = python.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
VisionRunningMode = vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

recognizer = GestureRecognizer.create_from_options(options)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

FOLLOW_MODE = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = recognizer.recognize(mp_image)

    gesture = "None"
    if result.gestures:
        gesture = result.gestures[0][0].category_name

    
    if gesture == "thumbs_up":
        FOLLOW_MODE = True
    elif gesture == "stop":
        FOLLOW_MODE = False

    
    if FOLLOW_MODE:
        pose_result = pose.process(rgb)

        if pose_result.pose_landmarks:
            mp_draw.draw_landmarks(frame, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            
            nose = pose_result.pose_landmarks.landmark[0]
            cx, cy = int(nose.x * w), int(nose.y * h)

        
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            
            center_x = w // 2

            
            if cx < center_x - 50:
                direction = "MOVE LEFT"
            elif cx > center_x + 50:
                direction = "MOVE RIGHT"
            else:
                direction = "HOVER"

            cv2.putText(frame, direction, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    
    cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    mode_text = "FOLLOW MODE ON" if FOLLOW_MODE else "FOLLOW MODE OFF"
    cv2.putText(frame, mode_text, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("AI Drone Demo", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
