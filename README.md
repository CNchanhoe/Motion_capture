# Real-Time Sign Language Recognition

A real-time hand gesture and sign language recognition project using Google's MediaPipe Tasks API and scikit-learn. This system extracts 3D hand landmarks from images or a webcam feed, applies geometric normalization, and classifies the gestures using a K-Nearest Neighbors (KNN) model.

## 📂 Project Files

* **`collect.py`**
    Reads a dataset of sign language images, detects hand landmarks using MediaPipe, and saves the 3D coordinate data into a CSV file.
* **`dataset_landmarks.csv`**
    The generated dataset containing the labels (folder names) and the extracted 63 ($x, y, z$) landmark coordinates for each image.
* **`print.py`**
    The main execution script. It loads the CSV data, trains a KNN classifier, and opens the webcam to perform real-time sign language translation on screen. It includes a normalization function to ensure accuracy regardless of hand size or distance from the camera.

## 🛠 Prerequisites

Make sure you have the required Python libraries installed:

```bash
pip install opencv-python mediapipe scikit-learn# Motion_capture
