import cv2
import numpy as np
import streamlit as st
import tempfile
import matplotlib.pyplot as plt

# Preprocess video function
def preprocess_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 128))
        frames.append(resized)
        frame_count += 1

    cap.release()
    return np.array(frames)

# Extract silhouettes
def extract_silhouettes(frames):
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    silhouettes = []

    for frame in frames:
        fg_mask = bg_subtractor.apply(frame)
        silhouettes.append(fg_mask)

    return np.array(silhouettes)

# Activity classification function based on contours
def classify_activity(person_contour):
    x, y, w, h = cv2.boundingRect(person_contour)
    aspect_ratio = h / w

    if aspect_ratio > 1.8:
        return "Walking or Running"
    elif 1.2 < aspect_ratio <= 1.8:
        return "Crawling"
    elif aspect_ratio <= 1.2:
        return "Clapping"
    else:
        return "Unknown"

# Detect and classify activity with frame-by-frame output in Streamlit
def detect_and_classify_activity(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    abnormal_threshold = 3
    frame_count = 0
    results = []
    frame_conditions = []  # To track conditions for graphing

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        activities = []

        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                activity = classify_activity(contour)
                activities.append(activity)

        # Determine frame condition based on activity types
        abnormal_activities = [act for act in activities if act in ["Walking or Running", "Crawling"]]
        condition = "Normal" if len(abnormal_activities) >= abnormal_threshold else "Abnormal"
        frame_conditions.append(condition)  # Store condition for graphing

        # Append frame data to results
        results.append({"frame": frame_count + 1, "condition": condition, "activities": activities})

        # Add alert for abnormal condition
        if condition == "Abnormal":
            cv2.putText(frame, "ALERT: Abnormal Activity Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

        # Calculate timestamp
        timestamp = frame_count / frame_rate  # Calculate timestamp in seconds
        minutes, seconds = divmod(timestamp, 60)
        timestamp_str = f"{int(minutes):02}:{int(seconds):02}"  # Format timestamp

        # Display timestamp on the frame
        cv2.putText(frame, f"Timestamp: {timestamp_str}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convert frame to image format compatible with Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display frame in the sidebar
        with st.sidebar:
            st.image(frame_rgb, caption=f"Frame {frame_count + 1} - Condition: {condition}", use_column_width=True)

        frame_count += 1

    cap.release()
    return results, frame_conditions  # Return frame conditions for graphing

# Function to plot the conditions
def plot_conditions(frame_conditions):
    # Create a line graph
    plt.figure(figsize=(10, 5))

    # Convert conditions to numerical values for plotting
    numerical_conditions = [1 if cond == "Abnormal" else 0 for cond in frame_conditions]

    plt.plot(range(len(numerical_conditions)), numerical_conditions, marker='o', color='b')
    plt.xticks(range(len(frame_conditions)), range(1, len(frame_conditions) + 1))  # Frame numbers on x-axis
    plt.yticks([0, 1], ['Normal', 'Abnormal'])  # Custom y-ticks
    plt.xlabel('Frame Number')
    plt.ylabel('Condition')
    plt.title('Frame Conditions Over Time')
    plt.grid(True)

    # Display the plot in Streamlit
    st.pyplot(plt)

# Streamlit UI
st.title("Gait recognition for multi-person activity analysis")
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_video_path = temp_file.name

    # Display the uploaded video automatically
    st.video(temp_video_path)

    # Run detection and classification
    results, frame_conditions = detect_and_classify_activity(temp_video_path, max_frames=30)

    # Display final group condition based on weighted evaluation
    abnormal_frames = sum(1 for result in results if result["condition"] == "Abnormal")
    normal_frames = len(results) - abnormal_frames

    # Adjust the condition check logic
    if abnormal_frames > 0:
        # Check if the normal frames are still significant
        group_condition = "Normal" if normal_frames > (abnormal_frames * 2) else "Abnormal"
    else:
        group_condition = "Normal"

    group_condition_ratio = abnormal_frames / len(results) if results else 0
    st.write(f"Overall Group Condition: {group_condition} (Abnormal Ratio: {group_condition_ratio:.2%})")

    # Plot the conditions
    plot_conditions(frame_conditions)

    # Clean up temporary file
    temp_file.close()
