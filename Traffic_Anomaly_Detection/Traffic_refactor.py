import cv2
import os
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

def detect_traffic_violations(video_path, output_folder, red_light_region=None, green_light_region=None, roi_region=None):
    """
    Process a video to detect traffic violations at a stoplight.
    
    Parameters:
    -----------
    video_path : str
        Path to the input video file
    output_folder : str
        Directory to save output files
    red_light_region : numpy.ndarray, optional
        Coordinates for red light area [[x1,y1], [x2,y2], ...]
    green_light_region : numpy.ndarray, optional
        Coordinates for green light area [[x1,y1], [x2,y2], ...]
    roi_region : numpy.ndarray, optional
        Coordinates for region of interest (intersection) [[x1,y1], [x2,y2], ...]
        
    Returns:
    --------
    dict
        Results including violation count, output paths, and processing statistics
    """
    # Results dictionary to return
    results = {
        'violation_count': 0,
        'processed_video': None,
        'summary_image': None,
        'violations': [],
        'processing_time': 0,
        'error': None
    }
    
    try:
        # Use default regions if not specified
        if red_light_region is None:
            red_light_region = np.array([[998, 125], [998, 155], [972, 152], [970, 127]])
        
        if green_light_region is None:
            green_light_region = np.array([[971, 200], [996, 200], [1001, 228], [971, 230]])
        
        if roi_region is None:
            roi_region = np.array([[910, 372], [388, 365], [338, 428], [917, 441]])
        
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Initialize model
        model = YOLO("yolov8m.pt")
        coco = model.model.names
        target_labels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]
        
        # Track violations
        violations = []
        violation_count = 0
        violation_frames = []
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            results['error'] = f"Could not open video file: {video_path}"
            return results
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = 1100
        frame_height = 700
        
        # Create output filenames with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"traffic_violation_{timestamp}.mp4"
        summary_filename = f"violation_summary_{timestamp}.jpg"
        output_video_path = os.path.join(output_folder, video_filename)
        summary_path = os.path.join(output_folder, summary_filename)
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        # Track frame count and start time
        frame_count = 0
        start_time = time.time()
        
        # Helper functions
        def is_region_light(image, polygon, brightness_threshold=128):
            """Check if a region (traffic light) is illuminated based on brightness."""
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(gray_image)
            cv2.fillPoly(mask, [polygon], 255)
            roi = cv2.bitwise_and(gray_image, gray_image, mask=mask)
            mean_brightness = cv2.mean(roi, mask=mask)[0]
            return mean_brightness > brightness_threshold
        
        def draw_text_with_background(frame, text, position, font, scale, text_color, bg_color, border_color, thickness=2, padding=5):
            """Draw text with background and border on the frame."""
            (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
            x, y = position
            # Background rectangle
            cv2.rectangle(frame, 
                         (x - padding, y - text_height - padding), 
                         (x + text_width + padding, y + baseline + padding), 
                         bg_color, 
                         cv2.FILLED)
            # Border rectangle
            cv2.rectangle(frame, 
                         (x - padding, y - text_height - padding), 
                         (x + text_width + padding, y + baseline + padding), 
                         border_color, 
                         thickness)
            # Text
            cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, lineType=cv2.LINE_AA)
        
        # Process video frames
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Resize frame
            frame = cv2.resize(frame, (frame_width, frame_height))
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 670), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw regions
            cv2.polylines(frame, [red_light_region], True, [0, 0, 255], 1)
            cv2.polylines(frame, [green_light_region], True, [0, 255, 0], 1)
            cv2.polylines(frame, [roi_region], True, [255, 0, 0], 2)
            
            # Check if red light is on
            red_light_on = is_region_light(frame, red_light_region)
            
            # Process results
            results_yolo = model.predict(frame, conf=0.75)
            
            for result in results_yolo:
                boxes = result.boxes.xyxy
                confs = result.boxes.conf
                classes = result.boxes.cls
                
                for box, conf, cls in zip(boxes, confs, classes):
                    if coco[int(cls)] in target_labels:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), [0, 255, 0], 2)
                        
                        draw_text_with_background(frame, 
                                          f"{coco[int(cls)].capitalize()}, conf:{(conf)*100:0.2f}%", 
                                          (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_COMPLEX, 
                                          0.6, 
                                          (255, 255, 255),
                                          (0, 0, 0),
                                          (0, 0, 255))
                        
                        # Check for violation - only for vehicles, not traffic lights
                        if red_light_on and coco[int(cls)] in ["bicycle", "car", "motorcycle", "bus", "truck"]:
                            # Calculate center of vehicle
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            # Check if in intersection zone
                            if cv2.pointPolygonTest(roi_region, (center_x, center_y), False) >= 0:
                                violation_count += 1
                                
                                # Store violation info
                                violations.append({
                                    'frame': frame_count,
                                    'type': coco[int(cls)],
                                    'confidence': float(conf),
                                    'time': datetime.now().strftime("%H:%M:%S")
                                })
                                
                                # Store frame for summary (limit to 10)
                                if len(violation_frames) < 10:
                                    violation_frames.append(frame.copy())
                                
                                # Highlight violation
                                draw_text_with_background(frame, 
                                              f"The {coco[int(cls)].capitalize()} violated the traffic signal!", 
                                              (10, 30), 
                                              cv2.FONT_HERSHEY_COMPLEX, 
                                              0.6, 
                                              (255, 255, 255),
                                              (0, 0, 0),
                                              (0, 0, 255))
                                
                                cv2.polylines(frame, [roi_region], True, [0, 0, 255], 2)
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
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create summary image
        summary = np.zeros((700, 1100, 3), dtype=np.uint8)  # Black background
        
        if len(violation_frames) > 0:
            # Add title
            cv2.putText(summary, "Traffic Violation Summary", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(summary, f"Total Violations: {violation_count}", (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add timestamp
            timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(summary, f"Processed on: {timestamp_text}", (300, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Show up to 4 violation frames in a grid
            max_frames = min(4, len(violation_frames))
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
        else:
            # Create a blank summary if no violations
            summary = np.ones((700, 1100, 3), dtype=np.uint8) * 255  # White background
            cv2.putText(summary, "No violations detected", (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add the summary frame to the end of the video
        for _ in range(int(fps * 5)):  # Show summary for 5 seconds
            out.write(summary)
        
        # Save summary image
        cv2.imwrite(summary_path, summary)
        
        # Release resources
        cap.release()
        out.release()
        
        # Update results
        results['violation_count'] = violation_count
        results['processed_video'] = video_filename
        results['summary_image'] = summary_filename
        results['violations'] = violations
        results['processing_time'] = processing_time
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        results['error'] = f"Error: {str(e)}\n{traceback_str}"
    
    return results

# This allows the module to be run directly for testing
if __name__ == "__main__":
    project_dir = Path(__file__).parent
    video_path = str(project_dir / "sample.mp4")
    output_folder = str(project_dir / "output")
    
    print(f"Testing traffic violation detection on {video_path}")
    print(f"Output will be saved to {output_folder}")
    
    results = detect_traffic_violations(video_path, output_folder)
    
    if results['error']:
        print(f"Error: {results['error']}")
    else:
        print(f"\nProcessing completed:")
        print(f"- Detected {results['violation_count']} traffic violations")
        print(f"- Processing time: {results['processing_time']:.2f} seconds")
        print(f"- Output video: {results['processed_video']}")
        print(f"- Summary image: {results['summary_image']}")