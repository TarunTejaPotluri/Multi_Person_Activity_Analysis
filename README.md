# Multi_Person_Activity_Analysis
A Streamlit-based application for gait recognition and multi-person activity analysis from video input. It preprocesses video frames, extracts silhouettes, and classifies activities based on contours. The application flags abnormal group conditions and visualizes frame-by-frame activity conditions over time.

Certainly! Here's a comprehensive `README.md` file containing the necessary information to guide users through setting up and using your activity recognition project:

---

# Activity Recognition using Mediapipe and OpenCV

This project demonstrates how to use Mediapipe and OpenCV to perform real-time activity recognition from a video file. The activities recognized include Walking, Clapping, Running, and detecting abnormal movements based on the angles between the shoulder, elbow, and wrist landmarks.

## Table of Contents
- [Installation](#installation)
- [How to Use](#how-to-use)
  - [Google Colab](#google-colab)
  - [Local Setup](#local-setup)
- [Project Structure](#project-structure)
- [Streamlit App](#streamlit-app)
- [Acknowledgments](#acknowledgments)

## Installation

### Google Colab
To run this project in Google Colab, ensure that you have the following libraries installed:

```bash
!pip install mediapipe opencv-python
```

### Local Setup
If you want to run the project locally, clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/activity-recognition.git
cd activity-recognition
pip install -r requirements.txt
```

## How to Use

### Google Colab

1. Upload the `v4.mp4` video file to your Colab workspace or place it in the `/content/sample_data/` directory.
2. Copy the code from the repository into a Colab notebook.
3. Run the code to start recognizing activities from the video.

### Local Setup

1. Place your video file in the `sample_data/` directory.
2. Run the `app.py` script using Streamlit:
   
   ```bash
   streamlit run app.py
   ```

3. Open the provided localhost URL in your web browser to interact with the app.

## Project Structure

- `app.py`: The main script that runs the Streamlit app for activity recognition.
- `requirements.txt`: Contains all the dependencies required for the project.
- `README.md`: Documentation and instructions for the project.
- `sample_data/v4.mp4`: Sample video file used for demonstration.

## Streamlit App

The Streamlit app provides a user-friendly UI to upload a video file and recognize activities in real-time. The app displays the processed video with landmarks and activity annotations.

Here is the code for the Streamlit app:

```python
import streamlit as st
import cv2
import mediapipe as mp
import math
import tempfile
import os

# Initialize Mediapipe Pose and Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Streamlit app
st.title("Activity Recognition using Mediapipe and OpenCV")

st.markdown("Upload a video file to recognize activities like Walking, Clapping, and Running.")

# File uploader to upload video
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Load the video using OpenCV
    cap = cv2.VideoCapture(tfile.name)
    
    stframe = st.empty()
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            pose_results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if pose_results.pose_landmarks is not None:
                left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_elbow = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                left_wrist = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]

                left_shoulder_x = left_shoulder.x * frame.shape[1]
                left_shoulder_y = left_shoulder.y * frame.shape[0]
                left_elbow_x = left_elbow.x * frame.shape[1]
                left_elbow_y = left_elbow.y * frame.shape[0]
                left_wrist_x = left_wrist.x * frame.shape[1]
                left_wrist_y = left_wrist.y * frame.shape[0]

                left_shoulder_elbow_angle = math.degrees(
                    math.atan2(left_elbow_y - left_shoulder_y, left_elbow_x - left_shoulder_x) -
                    math.atan2(left_wrist_y - left_elbow_y, left_wrist_x - left_elbow_x))

                mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                if left_shoulder_elbow_angle < 45 and left_shoulder_elbow_angle > -45:
                    activity = "Walking"
                elif left_shoulder_elbow_angle < -120 or left_shoulder_elbow_angle > 120:
                    activity = "Clapping"
                elif left_shoulder_elbow_angle < -150 or left_shoulder_elbow_angle > 150:
                    activity = "Running"
                else:
                    activity = "Abnormal"

                cv2.putText(frame, f"Activity: {activity}",
                            (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame in Streamlit
            stframe.image(frame, channels="BGR")

    cap.release()
    os.remove(tfile.name)

st.text("Upload a video file to start the recognition process.")
```

## Acknowledgments

- [Mediapipe](https://google.github.io/mediapipe/): A cross-platform framework for building multimodal applied machine learning pipelines.
- [OpenCV](https://opencv.org/): Open Source Computer Vision Library.
- [Streamlit](https://streamlit.io/): An open-source app framework for Machine Learning and Data Science teams.

