import os
import cv2

video_path = "./data/testing_data/detection_extraction/IMG_1550.mov"

print("cwd:", os.getcwd())
print("exists:", os.path.exists(video_path))
print("abs:", os.path.abspath(video_path))

cap = cv2.VideoCapture(video_path)
print("opened:", cap.isOpened())
cap.release()