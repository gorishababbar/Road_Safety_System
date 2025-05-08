import cv2
from ultralytics import YOLO
from datetime import timedelta
import os

# Load YOLOv8 model (downloads automatically if not present)
model = YOLO("yolov8n.pt")  # lightweight YOLOv8 nano model

# Video path (replace with your video)
video_path = os.path.join(os.path.dirname(__file__), 'sample.mp4')

# Open video capture
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video file")
    exit()

# Frame rate to process every 1 second
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)

frame_count = 0
log = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # Current timestamp
        time_sec = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        timestamp = str(timedelta(seconds=time_sec))

        # Run YOLO detection
        results = model(frame)
        
        # Get class names from model, not from results
        # The names dictionary is stored in the model, not in results
        detected_objects = []
        
        if len(results) > 0 and hasattr(results[0], 'boxes'):
            for box in results[0].boxes:
                # Get class ID and use model.names to get the class name
                cls_id = int(box.cls.item())
                class_name = model.names[cls_id]
                detected_objects.append(class_name)

        # Log detected objects or no activity
        if detected_objects:
            event = f"At {timestamp} — {', '.join(set(detected_objects))} detected"
        else:
            event = f"At {timestamp} — No activity detected"

        print(event)
        log.append(event)

    frame_count += 1

cap.release()

# Save events to a text file
output_path = os.path.join(os.path.dirname(__file__), 'surveillance_summary.txt')
with open(output_path, "w") as f:
    for event in log:
        f.write(event + "\n")

print(f"Summary saved to {output_path}")