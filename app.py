import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load the trained model
model = joblib.load('deepfake_model.pkl')

# Example descriptions for features (ensure these match your actual features)
feature_descriptions = {
    0: "Mean Color Channel (Red)",
    1: "Standard Deviation Color (Green)",
    2: "Mean Brightness (Lighting)",
    3: "Texture Variance",
    4: "Mean Color Channel (Blue)",
    5: "Brightness Spike Detection",
    6: "Texture Consistency",
    7: "Color Anomaly in Frame",
    8: "Edge Density",
    9: "Lighting Consistency",
}

# Function to extract frames from a video
def extract_frames(video_path, frame_rate=5):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:  # Capture frame at specified interval
            frame = cv2.resize(frame, (224, 224))  # Resize frame
            frames.append(frame.flatten())  # Flatten frame to 1D array
        frame_count += 1

    cap.release()
    return np.array(frames)

# Function to predict video and extract features
def predict_video(video_path):
    frames = extract_frames(video_path)
    if frames.size > 0:
        feature_vector = np.mean(frames, axis=0).reshape(1, -1)
        prediction = model.predict(feature_vector)
        feature_importances = model.feature_importances_
        return prediction[0], feature_vector.flatten(), feature_importances
    else:
        st.error("No frames were extracted from the video.")
        return None, None, None

# Export features to CSV
def export_features_to_csv(features, feature_importances, output_path, prediction_label):
    feature_data = []
    sorted_indices = np.argsort(feature_importances)[::-1]  # Sort features by importance

    for idx in sorted_indices:
        feature_data.append({
            "Feature Index": idx,
            "Feature Description": feature_descriptions.get(idx, f"Feature {idx}"),
            "Feature Value": features[idx],
            "Feature Importance": feature_importances[idx],
            "Prediction Label": "Fake" if prediction_label == 1 else "Real"
        })

    df = pd.DataFrame(feature_data)
    df.to_csv(output_path, index=False)
    st.success(f"Feature data saved to {output_path}")

# Function to identify feature location in frames
def identify_feature(index, frame_width=224, frame_height=224, channels=3):
    pixels_per_channel = frame_width * frame_height
    total_features = pixels_per_channel * channels

    if index >= total_features:
        return "Index out of bounds for the given frame dimensions."

    # Determine the channel
    channel = index // pixels_per_channel
    relative_index = index % pixels_per_channel

    # Calculate row and column
    row = relative_index // frame_width
    col = relative_index % frame_width

    # Map channel number to color
    channel_name = ["Red", "Green", "Blue"][channel]

    return f"Channel: {channel_name}, Row: {row}, Column: {col}"

# Streamlit app layout
st.title("Deepfake Detection")
st.write("Upload a video to predict if it is Fake or Real and analyze features.")

# Video uploader
video_file = st.file_uploader("Upload a video file (mp4, avi, webm):", type=["mp4", "avi", "webm"])

if video_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = "uploaded_video.mp4"
    with open(temp_file_path, "wb") as f:
        f.write(video_file.read())

    # Predict and analyze
    prediction, features, feature_importances = predict_video(temp_file_path)

    if prediction is not None:
        # Display prediction
        st.subheader("Prediction Result")
        st.write(f"The video is **{'Fake' if prediction == 1 else 'Real'}**.")

        # Feature importance analysis
        st.subheader("Feature Importance Analysis")
        sorted_indices = np.argsort(feature_importances)[::-1][:10]  # Top 10 features
        st.write("Top Contributing Features:")
        for idx in sorted_indices:
            description = feature_descriptions.get(idx, f"Feature {idx}")
            location = identify_feature(idx)
            st.write(f"- {description}: Importance {feature_importances[idx]:.4f}, "
                     f"Value {features[idx]:.4f}, {location}")

        # Save feature data to CSV
        output_csv_path = r'path\to\deepfake_features_analysis.csv'
        export_features_to_csv(features, feature_importances, output_csv_path, prediction)

    # Optional: Display the video
    st.subheader("Video Preview")
    st.video(video_file)
