# Drowsiness Detection System

A real-time drowsiness detection system built with Streamlit, OpenCV, and TensorFlow. The application uses your webcam to monitor your eyes and alerts you if signs of drowsiness are detected.

## Features

- Real-time face and eye detection
- Drowsiness detection based on eye state
- Audio alerts when drowsiness is detected
- User-friendly web interface
- Works with any webcam

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd drowsiness-detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

## How it Works

1. The app uses your webcam to capture video in real-time
2. OpenCV detects faces and eyes in each frame
3. A pre-trained deep learning model analyzes the eye state (open/closed)
4. If eyes remain closed for too long, an audio alert is triggered
5. A drowsiness score is displayed to indicate the current state

## Requirements

- Python 3.8+
- Webcam
- See requirements.txt for Python package dependencies

## License

MIT License
