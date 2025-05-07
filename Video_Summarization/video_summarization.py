import cv2
from ultralytics import YOLO
from datetime import timedelta

# Load YOLOv8 model (downloads automatically if not present)
model = YOLO("yolov8n.pt")  # lightweight YOLOv8 nano model

# Video path (replace with your video)
video_path = r'sample.mp4'  # Path to your video file"

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
        detections = results[0].names
        detected_objects = [results[0].names[int(cls)] for cls in results[0].boxes.cls]

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
with open(r'C:\Users\ASUS\Desktop\Video Summarizer\surveillance_summary.txt', "w") as f:
    for event in log:
        f.write(event + "\n")

print("Summary saved to surveillance_summary.txt")
