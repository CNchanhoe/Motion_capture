import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import os
from sklearn.neighbors import KNeighborsClassifier

csv_file = 'dataset_landmarks.csv' 

def normalize_landmarks(landmark_list):
    base_x, base_y, base_z = landmark_list[0], landmark_list[1], landmark_list[2]
    translated_list = []
    
    for i in range(0, len(landmark_list), 3):
        translated_list.append(landmark_list[i] - base_x)
        translated_list.append(landmark_list[i+1] - base_y)
        translated_list.append((landmark_list[i+2] - base_z) * 0.2)

    max_value = max([abs(val) for val in translated_list])
    if max_value > 0:
        normalized_list = [val / max_value for val in translated_list]
    else:
        normalized_list = translated_list
        
    return normalized_list

X = []
y = []

if not os.path.exists(csv_file):
    print(f"{csv_file} 파일이 없습니다. 데이터셋 추출을 먼저 진행해 주세요!")
    exit()

print("데이터를 불러오고 학습하는 중입니다. (데이터가 많으면 몇 초 걸릴 수 있습니다...)")
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        label = row[0]
        raw_landmarks = [float(val) for val in row[1:]]
        normalized_landmarks = normalize_landmarks(raw_landmarks)
        
        y.append(label)
        X.append(normalized_landmarks)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
print("학습 완료! 웹캠을 켭니다.")

model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            raw_landmark_list = []
            for landmark in hand_landmarks:
                raw_landmark_list.extend([landmark.x, landmark.y, landmark.z])
                
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            
            normalized_live_landmarks = normalize_landmarks(raw_landmark_list)
            
            pred_label = knn.predict([normalized_live_landmarks])[0]

            cv2.putText(frame, str(pred_label), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
                
    cv2.imshow('Gesture Recognition', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()