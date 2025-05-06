import cv2
import os
from ultralytics import YOLO
import numpy as np
from pathlib import Path

# Centralized project root
project_root = Path(__file__).parent

# Define Red Light and ROI regions
RedLight = np.array([[998, 125], [998, 155], [972, 152], [970, 127]])
ROI = np.array([[910, 372], [388, 365], [338, 428], [917, 441]])

# Load YOLO model
model = YOLO("yolov8m.pt")
coco = model.model.names
TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck"]  # Only vehicles

# Function to check if red light is on
def is_red_light_on(image, polygon, brightness_threshold=128):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_image)
    cv2.fillPoly(mask, [polygon], 255)
    roi = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    mean_brightness = cv2.mean(roi, mask=mask)[0]
    return mean_brightness > brightness_threshold

# Start video capture
cap = cv2.VideoCapture("sample.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (frame_width, frame_height))

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video processing complete.")
        break

    frame = cv2.resize(frame, (1100, 700))
    cv2.polylines(frame, [RedLight], True, [0, 0, 255], 2)
    cv2.polylines(frame, [ROI], True, [255, 0, 0], 2)

    results = model.predict(frame, conf=0.75)
    red_light_on = is_red_light_on(frame, RedLight)

    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            if coco[int(cls)] in TargetLabels:
                x, y, w, h = map(int, box)
                in_roi = cv2.pointPolygonTest(ROI, (x, y), False) >= 0 or cv2.pointPolygonTest(ROI, (w, h), False) >= 0
                
                if red_light_on and in_roi:
                    cv2.rectangle(frame, (x, y), (w, h), [0, 0, 255], 3)  # Red box for violators
                    cv2.putText(frame, f"{coco[int(cls)].capitalize()} violated red light!", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Traffic Signal Violation", frame)
    out.write(frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
