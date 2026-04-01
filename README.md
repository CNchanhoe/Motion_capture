# Hand Landmark-based Motion Capture & Classification

This project implements a hand gesture recognition system using **MediaPipe** for landmark extraction and **K-Nearest Neighbors (KNN)** for classification. It consists of a data collection pipeline and a real-time inference system.

## 📁 Project Structure
- `collect.py`: Script to extract 3D hand landmarks from images and save them to a CSV file.
- `print.py`: Real-time hand gesture recognition script using a webcam and KNN classifier.
- `dataset_landmarks.csv`: The generated dataset containing normalized hand landmark coordinates and their corresponding labels.

## ✨ Features
- **3D Landmark Extraction**: Uses MediaPipe's Hand Landmarker to detect 21 hand joints.
- **Data Normalization**: Custom normalization logic to ensure translation and scale invariance (relative to the wrist).
- **Real-time Inference**: Live classification of hand gestures via webcam using the KNN algorithm.

## 🛠️ Requirements
- Python 3.8+
- OpenCV
- MediaPipe
- scikit-learn
- NumPy

Install the dependencies using:
```bash
pip install opencv-python mediapipe scikit-learn numpy
