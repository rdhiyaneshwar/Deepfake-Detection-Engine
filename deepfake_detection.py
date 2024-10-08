import streamlit as st
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import tempfile
import numpy as np

mtcnn = MTCNN(keep_all=True, device='cpu')

# Load InceptionResnetV1 pre-trained model for feature extraction (face embeddings)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Simulated deepfake classifier (replace with actual model)
def classify_face(face_tensor):
    with torch.no_grad():
        # Extract embeddings using InceptionResnetV1
        embedding = model(face_tensor)
        return "Real" if torch.rand(1).item() > 0.5 else "Fake"

# Detect faces and classify each face as real or fake
def detect_and_classify_faces(video_file):
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    results = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the frame
        faces = mtcnn(rgb_frame)
        
        if faces is not None:
            for i, face_tensor in enumerate(faces):
                if face_tensor is not None:
                    face_tensor = face_tensor.unsqueeze(0)
                    result = classify_face(face_tensor)
                    results.append(result)
                    st.write(f"Frame {frame_count + 1}: Face {i + 1} - Result: {result}")
        
        frame_count += 1
    
    cap.release()
    
    fake_count = results.count("Fake")
    real_count = results.count("Real")
    
    if fake_count > 0:
        return "Deepfake Detected"
    else:
        return "Real Video"
    
st.title("Deepfake Detection Engine")
st.write("Upload a video to analyze for deepfakes.")

uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())
    
    st.video(uploaded_video)
    
    # Run the detection and classification
    st.write("Analyzing the video...")
    detection_result = detect_and_classify_faces(temp_file.name)
    
    st.write(f"Final Result: {detection_result}")
