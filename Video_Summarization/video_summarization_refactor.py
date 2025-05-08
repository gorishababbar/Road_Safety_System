import cv2
from ultralytics import YOLO
from datetime import timedelta
import os
import torch
import sys

# Create a patched version of torch.load before importing YOLO
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    # Force weights_only to False
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

# Replace torch.load with our patched version
torch.load = patched_torch_load

# Load YOLOv8 model (downloads automatically if not present)
try:
    model = YOLO("yolov8n.pt")  # lightweight YOLOv8 nano model
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

def summarize_video(video_path, output_path=None):
    """
    Summarize video by detecting objects in regular intervals
    
    Args:
        video_path (str): Path to the video file
        output_path (str, optional): Path to save the summary text file
        
    Returns:
        list: List of event strings (the summary)
    """
    # Open video capture
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return ["Error opening video file"]

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
            
            # Use the approach from the working video_summarization.py file
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

            log.append(event)

        frame_count += 1

    cap.release()

    # Save events to a text file if output_path is provided
    if output_path:
        with open(output_path, "w") as f:
            for event in log:
                f.write(event + "\n")
    
    return log

# For standalone execution
if __name__ == "__main__":
    video_path = os.path.join(os.path.dirname(__file__), 'sample.mp4')
    output_path = os.path.join(os.path.dirname(__file__), 'surveillance_summary.txt')
    
    summary = summarize_video(video_path, output_path)
    
    for event in summary:
        print(event)
    print(f"Summary saved to {output_path}")