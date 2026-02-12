# Drone-Project
AI Drone Gesture Control with Human Follow-Me System

This project is an AI-based drone control simulation that allows control using hand gestures and includes an advanced human follow-me tracking feature.

The full logic is demonstrated on a computer screen (no physical drone required), but the system is designed for real drone integration in the future.

✨ Main Features

Real-time hand gesture recognition using MediaPipe.

Separate custom CNN model trained on a user-collected gesture dataset.

Supports drone-style commands:

Takeoff

Land

Left / Right / Up / Down

Stop

Thumbs Up / Thumbs Down

Smart Follow-Me AI system:

Detects a specific human target.

Continuously tracks movement.

Maintains safe distance automatically.

Hovers when the person stops.

Increases speed when the person moves fast.

Works as a complete drone behavior simulation on screen.

🧠 Project Novelty

Uses a fully custom gesture dataset, not default pretrained gestures.

Combines gesture control + intelligent human tracking in one system.

Target human can be selected using a gesture instead of face recognition.

mediapipe/
│
├── dataset/                  # Custom gesture images
├── gesture_model.task        # Trained MediaPipe model
├── detect.py                 # Gesture recognition (MediaPipe)
├── final_drone.py            # Gesture + Follow-Me integration
│
cnn/
│
├── split_data/               # Train / Validation / Test folders
├── train_cnn_custom.py       # CNN training script
├── realtime_cnn_detect.py    # Real-time CNN detection
├── custom_gesture_model.keras

Technologies Used

Python

OpenCV

MediaPipe

TensorFlow / Keras

NumPy

Matplotlib

How to Run the Project
1️⃣ MediaPipe Gesture Detection
python detect.py

2️⃣ CNN Real-Time Gesture Detection

Activate environment and run:
python realtime_cnn_detect.py

3️⃣ Final Drone Simulation (Gesture + Follow-Me)
python final_drone.py

Demo Flow

Start the camera feed.

Show a gesture → command gets recognized.

Drone movement is simulated on screen.

Select a human target → drone begins following.

If the person stops → drone hovers automatically.

Future Improvements

Integration with a real drone.

Multi-person tracking with identity selection.

Deployment on Raspberry Pi / Jetson Nano.

Adding voice + gesture hybrid control.

👨‍💻 Author

Developed as an AI + Computer Vision project demonstrating:

Gesture-based drone control with intelligent human tracking.

License

This project is created for educational and research purposes only.
Demonstrates real drone-level logic without physical hardware.

Designed for easy upgrade to real drone hardware in future.
