import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv

dataset_path = '/Users/chan050714/asl_dataset' 
output_csv = 'dataset_landmarks.csv'

model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE, 
    num_hands=1,
    min_hand_detection_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = ['label'] + [f'{axis}_{i}' for i in range(21) for axis in ('x', 'y', 'z')]
    writer.writerow(header)

print("이미지에서 랜드마크 추출을 시작합니다...")

total_images = 0
success_images = 0

for label_folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, label_folder)
    
    if not os.path.isdir(folder_path):
        continue
        
    print(f"[{label_folder}] 폴더 처리 중...")

    for image_name in os.listdir(folder_path):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(folder_path, image_name)
        total_images += 1

        image = cv2.imread(image_path)
        if image is None:
            continue
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        detection_result = detector.detect(mp_image)

        if detection_result.hand_landmarks:
            success_images += 1
            landmark_list = []

            for landmark in detection_result.hand_landmarks[0]:
                landmark_list.extend([landmark.x, landmark.y, landmark.z])

            with open(output_csv, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([label_folder] + landmark_list)

print("========================================")
print(f"작업 완료! 총 {total_images}장의 이미지 중 {success_images}장에서 손을 찾아 저장했습니다.")
print(f"결과물: {output_csv}")