import cv2
import os
from pathlib import Path
import time
from datetime import datetime
import argparse

def extract_frames(video_path, output_dir=None, frame_interval=1):
    """
    Extract frames from a video file at specified intervals
    
    Parameters:
    -----------
    video_path : str
        Path to the input video file
    output_dir : str, optional
        Directory to save the extracted frames. If None, creates a directory based on video name
    frame_interval : int, optional
        Extract frames at this interval (1 means every frame, 30 means every 30th frame)
        
    Returns:
    --------
    dict
        Results including frame count, output directory, and processing statistics
    """
    results = {
        'success': False,
        'output_dir': None,
        'frame_count': 0,
        'processing_time': 0,
        'error': None
    }
    
    try:
        # Validate video path
        if not os.path.exists(video_path):
            results['error'] = f"Video file not found: {video_path}"
            return results
        
        # Create output directory
        if output_dir is None:
            video_name = Path(video_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f"extracted_frames_{video_name}_{timestamp}")
        else:
            output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Open the video file
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            results['error'] = f"Failed to open video file: {video_path}"
            return results
        
        # Video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"Video properties: {frame_count} frames, {fps:.2f} fps, {duration:.2f}s duration")
        print(f"Extracting frames (interval: {frame_interval})")
        
        # Start timing
        start_time = time.time()
        
        # Extract frames
        count = 0
        saved_count = 0
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            if count % frame_interval == 0:
                # Save the frame
                frame_file = output_dir / f"frame_{saved_count:05d}.jpg"
                cv2.imwrite(str(frame_file), frame)
                saved_count += 1
                
                # Show progress
                if saved_count % 10 == 0:
                    print(f"Extracted {saved_count} frames...", end="\r")
            
            count += 1
        
        # Release the video capture object
        video.release()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update results
        results['success'] = True
        results['output_dir'] = str(output_dir)
        results['frame_count'] = saved_count
        results['processing_time'] = processing_time
        results['total_frames'] = count
        
        print(f"\nExtracted {saved_count} frames from {count} total frames in {processing_time:.2f}s")
        print(f"Frames saved to: {output_dir}")
        
    except Exception as e:
        results['error'] = str(e)
        import traceback
        print(f"Error extracting frames: {e}")
        print(traceback.format_exc())
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--output", help="Directory to save extracted frames")
    parser.add_argument("--interval", type=int, default=30, help="Extract every Nth frame (default: 30)")
    
    args = parser.parse_args()
    
    print(f"Extracting frames from {args.video_path}")
    extract_frames(args.video_path, args.output, args.interval)