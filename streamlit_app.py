# streamlit_app.py
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Drowsiness Detection", page_icon="🚗")

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import os
import time
import subprocess


# Load the pre-trained model with error handling
@st.cache_resource
def load_drowsiness_model():
    try:
        model = load_model('drowsiness_detection_model.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_drowsiness_model()

# Load Haar cascade classifiers with correct paths
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Check if cascades loaded successfully
if face_cascade.empty():
    st.error("Error: Could not load face cascade classifier")
if eye_cascade.empty():
    st.error("Error: Could not load eye cascade classifier")

# Initialize score
score = 0

def play_alarm():
    try:
        # Try different audio players based on platform
        import platform
        system = platform.system()
        
        if system == "Darwin":  # macOS
            subprocess.Popen(['afplay', 'alarm.mp3'])
        elif system == "Linux":  # Linux (Streamlit Cloud)
            # Try aplay first, then fallback to other options
            try:
                subprocess.Popen(['aplay', 'alarm.mp3'])
            except FileNotFoundError:
                try:
                    subprocess.Popen(['paplay', 'alarm.mp3'])
                except FileNotFoundError:
                    # If no audio player available, just show visual alert
                    st.warning("🚨 DROWSINESS DETECTED! 🚨")
        elif system == "Windows":  # Windows
            import winsound
            winsound.PlaySound('alarm.mp3', winsound.SND_FILENAME)
    except Exception as e:
        # Fallback to visual alert if audio fails
        st.warning("🚨 DROWSINESS DETECTED! 🚨")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.score = 0
        self.last_alarm_time = time.time()
        self.alarm_interval = 2  # Minimum time between alarms in seconds
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Check if model is loaded
        if model is None:
            cv2.putText(img, "Model not loaded", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Check if cascades are loaded
        if face_cascade.empty() or eye_cascade.empty():
            cv2.putText(img, "Cascade not loaded", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        # For each detected face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Get ROI for eyes
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            
            # Detect eyes in face ROI
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
            
            for (ex, ey, ew, eh) in eyes:
                # Draw rectangle around eyes
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                
                # Get eye ROI
                eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
                
                if eye_roi.size > 0:
                    # Preprocess eye image
                    eye_roi = cv2.resize(eye_roi, (80, 80))
                    eye_roi = eye_roi.reshape(-1, 80, 80, 1)
                    eye_roi = eye_roi / 255.0
                    
                    try:
                        # Make prediction
                        prediction = model.predict(eye_roi, verbose=0)
                        
                        if prediction > 0.5:  # Eye is open
                            cv2.putText(img, "Open", (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                            self.score = max(0, self.score - 1)
                        else:  # Eye is closed
                            cv2.putText(img, "Closed", (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                            self.score += 1
                    except Exception as e:
                        cv2.putText(img, "Prediction error", (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        
        # Display score
        cv2.putText(img, f'Score: {self.score}', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
        
        # Check if drowsy (lowered threshold to 15)
        if self.score > 15:
            # Display warning
            cv2.putText(img, "DROWSINESS ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Play alarm sound if enough time has passed since last alarm
            current_time = time.time()
            if current_time - self.last_alarm_time >= self.alarm_interval:
                play_alarm()  # Using the new alarm function
                self.last_alarm_time = current_time
            
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Add title
st.title("Drowsiness Detection System")

# Add description
st.write("""
This application detects drowsiness by monitoring your eyes through the webcam.
If your eyes remain closed for too long, it will trigger an alert with sound.
""")

# Check if model and cascades are loaded
if model is None:
    st.error("Model could not be loaded. Please check if 'drowsiness_detection_model.h5' exists.")
else:
    # RTC Configuration (using Google's STUN server)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Create WebRTC streamer
    webrtc_streamer(
        key="drowsiness-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_configuration
    )
    
    # Add instructions
    st.markdown("""
    ### Instructions:
    1. Click the 'START' button to begin webcam streaming
    2. Position yourself so your face is clearly visible
    3. The system will monitor your eyes for signs of drowsiness
    4. If drowsiness is detected (score > 15), a warning will appear with sound alert
    5. Click 'STOP' to end the session
    
    ### Note:
    - Ensure good lighting for better detection
    - Keep your face centered and visible
    - Wear glasses? Remove them if detection isn't working well
    - Make sure your system volume is turned on to hear the alerts
    """)
