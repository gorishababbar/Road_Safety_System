import cv2
import os
from ultralytics import YOLO, solutions
import numpy as np
import time
from pathlib import Path
from datetime import datetime

# Use correct project path instead of hardcoded one
project_dir = Path(__file__).parent
os.chdir(project_dir)  # Change working directory to script location

# Create output directory if it doesn't exist
output_dir = project_dir / "output"
output_dir.mkdir(exist_ok=True)

# Rest of your code remains unchanged
RedLight = np.array([[998, 125],[998, 155],[972, 152],[970, 127]])
GreenLight = np.array([[971, 200],[996, 200],[1001, 228],[971, 230]])
ROI = np.array([[910, 372],[388, 365],[338, 428],[917, 441]])

model = YOLO("yolov8m.pt")
coco = model.model.names
TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]

# Track violations
violations = []
violation_count = 0
violation_frames = []  # Store frames with violations

# Your functions remain unchanged
def is_region_light(image, polygon, brightness_threshold=128):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_image)
    cv2.fillPoly(mask, [np.array(polygon)], 255)
    roi = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    mean_brightness = cv2.mean(roi, mask=mask)[0]
    return mean_brightness > brightness_threshold

def draw_text_with_background(frame, text, position, font, scale, text_color, background_color, border_color, thickness=2, padding=5):
    """Draw text with background and border on the frame."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    # Background rectangle
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  background_color, 
                  cv2.FILLED)
    # Border rectangle
    cv2.rectangle(frame, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  border_color, 
                  thickness)
    # Text
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)

# Function to create summary frame
def create_summary(violation_count, violations, violation_frames):
    if not violation_frames:
        # Create a blank summary if no violations
        summary = np.ones((700, 1100, 3), dtype=np.uint8) * 255
        cv2.putText(summary, "No violations detected", (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return summary
    
    # Create summary image with black background
    summary = np.zeros((700, 1100, 3), dtype=np.uint8)
    
    # Add title
    cv2.putText(summary, "Traffic Violation Summary", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(summary, f"Total Violations: {violation_count}", (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(summary, f"Processed on: {timestamp}", (300, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Show up to 4 violation frames in a grid
    max_frames = min(4, len(violation_frames))
    if max_frames > 0:
        grid_size = 2  # 2x2 grid
        for i in range(max_frames):
            row = i // grid_size
            col = i % grid_size
            
            # Calculate position for this thumbnail
            x_offset = 50 + col * 500
            y_offset = 180 + row * 230
            
            # Get the violation frame and resize it to fit in the grid
            thumb = cv2.resize(violation_frames[i], (450, 200))
            
            # Place the thumbnail on the summary
            summary[y_offset:y_offset+200, x_offset:x_offset+450] = thumb
            
            # Add violation info below the thumbnail
            if i < len(violations):
                violation_text = f"Violation {i+1}: {violations[i]['type']}"
                cv2.putText(summary, violation_text, (x_offset, y_offset+220), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return summary

# Change the video filename to sample.mp4
cap = cv2.VideoCapture(str(project_dir / "sample.mp4"))
if not cap.isOpened():
    print(f"Error: Could not open video file")
    exit()

# Get original video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Set up output video writer
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = str(output_dir / f"traffic_violation_{timestamp}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (1100, 700))

# Track frame count
frame_count = 0

print("Processing video... Press ESC to stop early.")
start_time = time.time()

# The rest of your code remains unchanged
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Video processing complete.")
        break
    
    frame_count += 1
    
    # Resize frame
    frame = cv2.resize(frame, (1100, 700))
    
    # Add frame counter
    cv2.putText(frame, f"Frame: {frame_count}", (10, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw regions
    cv2.polylines(frame, [RedLight], True, [0, 0, 255], 1)
    cv2.polylines(frame, [GreenLight], True, [0, 255, 0], 1)
    cv2.polylines(frame, [ROI], True, [255, 0, 0], 2)
    
    # Check if red light is on
    red_light_on = is_region_light(frame, RedLight)
    
    # Process results
    results = model.predict(frame, conf=0.75)
    violation_in_frame = False
    
    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        classes = result.boxes.cls
        
        for box, conf, cls in zip(boxes, confs, classes):
            if coco[int(cls)] in TargetLabels:
                x1, y1, x2, y2 = map(int, box)  # Use x1,y1,x2,y2 format for xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 255, 0], 2)
                draw_text_with_background(frame, 
                                  f"{coco[int(cls)].capitalize()}, conf:{(conf)*100:0.2f}%", 
                                  (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_COMPLEX, 
                                  0.6, 
                                  (255, 255, 255),
                                  (0, 0, 0),
                                  (0, 0, 255))

                # Check for violation
                if red_light_on:
                    # Check if the center of the vehicle is in ROI
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    if cv2.pointPolygonTest(ROI, (center_x, center_y), False) >= 0:
                        violation_in_frame = True
                        violation_count += 1
                        
                        # Store violation info
                        violations.append({
                            'frame': frame_count,
                            'type': coco[int(cls)],
                            'confidence': float(conf)
                        })
                        
                        # Store this frame for summary
                        if len(violation_frames) < 10:  # Limit to 10 frames to save memory
                            violation_frames.append(frame.copy())
                        
                        draw_text_with_background(frame, 
                                      f"The {coco[int(cls)].capitalize()} violated the traffic signal!", 
                                      (10, 30), 
                                      cv2.FONT_HERSHEY_COMPLEX, 
                                      0.6, 
                                      (255, 255, 255),
                                      (0, 0, 0),
                                      (0, 0, 255))

                        cv2.polylines(frame, [ROI], True, [0, 0, 255], 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 0, 255], 2)
    
    # Add traffic light status indicator
    status = "RED" if red_light_on else "GREEN"
    status_color = (0, 0, 255) if red_light_on else (0, 255, 0)
    draw_text_with_background(frame, 
                      f"Traffic Light: {status}", 
                      (frame.shape[1] - 220, 30), 
                      cv2.FONT_HERSHEY_COMPLEX, 
                      0.6, 
                      (255, 255, 255),
                      (0, 0, 0),
                      status_color)
    
    # Add violation counter
    draw_text_with_background(frame, 
                      f"Total Violations: {violation_count}", 
                      (10, 70), 
                      cv2.FONT_HERSHEY_COMPLEX, 
                      0.6, 
                      (255, 255, 255),
                      (0, 0, 0),
                      (0, 0, 255))
        
    # Write frame to output video
    out.write(frame)
    
    # Display the frame
    cv2.imshow("Traffic Violation Detection", frame)
    
    # Break on ESC key
    if cv2.waitKey(1) == 27:
        print("Processing stopped by user.")
        break

# Calculate processing time
processing_time = time.time() - start_time
fps_processed = frame_count / processing_time if processing_time > 0 else 0

# Generate and show the summary
summary = create_summary(violation_count, violations, violation_frames)

# Add the summary frame to the end of the video
for _ in range(int(fps * 5)):  # Show summary for 5 seconds
    out.write(summary)

# Display summary
cv2.imshow("Traffic Violation Summary", summary)
cv2.waitKey(0)  # Wait for key press

# Save summary image
summary_path = str(output_dir / f"violation_summary_{timestamp}.jpg")
cv2.imwrite(summary_path, summary)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Print final statistics
print(f"\nProcessing completed:")
print(f"- Processed {frame_count} frames in {processing_time:.2f} seconds ({fps_processed:.2f} FPS)")
print(f"- Detected {violation_count} traffic violations")
print(f"- Output video saved to: {output_path}")
print(f"- Summary image saved to: {summary_path}")