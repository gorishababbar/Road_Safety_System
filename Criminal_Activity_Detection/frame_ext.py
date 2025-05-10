import cv2
import os
from datetime import datetime
from pathlib import Path
import uuid

def extract_frames(video_path, output_dir=None, frame_interval=1, resize_dim=None):
    """
    Extract frames from a video file and save them to a unique directory.
    
    Parameters:
    -----------
    video_path : str
        Path to the input video file
    output_dir : str, optional
        Base directory to save the extracted frames. If None, creates a directory in the same location as the script
    frame_interval : int, optional
        Extract frames at this interval (1 means every frame, 2 means every other frame, etc.)
    resize_dim : tuple, optional
        Resize frames to this dimension (width, height) before saving
        
    Returns:
    --------
    dict
        Results including frame count, output directory, and processing statistics
    """
    # Results dictionary to return
    results = {
        'success': False,
        'frame_count': 0,
        'output_dir': None,
        'processing_time': 0,
        'error': None
    }
    
    try:
        # Validate video path
        if not os.path.exists(video_path):
            results['error'] = f"Video file not found: {video_path}"
            return results
            
        # Create unique output directory if not specified
        if output_dir is None:
            script_dir = Path(__file__).parent
            base_output_dir = script_dir / "extracted_frames"
        else:
            base_output_dir = Path(output_dir)
            
        # Create a unique subdirectory with timestamp and UUID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        unique_id = str(uuid.uuid4())[:8]  # First 8 chars of UUID for brevity
        frames_dir = base_output_dir / f"{video_name}_{timestamp}_{unique_id}"
        
        # Create output directory
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Open the video file
        vidcap = cv2.VideoCapture(video_path)
        if not vidcap.isOpened():
            results['error'] = f"Failed to open video file: {video_path}"
            return results
            
        # Get video properties
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Start extraction
        import time
        start_time = time.time()
        
        frame_count = 0
        saved_count = 0
        
        # Read and save frames
        success, image = vidcap.read()
        while success:
            if frame_count % frame_interval == 0:
                # Resize if needed
                if resize_dim is not None:
                    image = cv2.resize(image, resize_dim)
                
                # Save the frame
                frame_path = frames_dir / f"frame_{saved_count:05d}.jpg"
                cv2.imwrite(str(frame_path), image)
                saved_count += 1
                
            frame_count += 1
            success, image = vidcap.read()
            
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Clean up
        vidcap.release()
        
        # Update results
        results['success'] = True
        results['frame_count'] = saved_count
        results['output_dir'] = str(frames_dir)
        results['processing_time'] = processing_time
        results['total_video_frames'] = total_frames
        results['fps'] = fps
        
    except Exception as e:
        results['error'] = str(e)
    
    return results

# This allows the module to be run directly for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract frames from a video file')
    parser.add_argument('video_path', type=str, help='Path to the input video file')
    parser.add_argument('--output_dir', type=str, help='Base directory to save the extracted frames')
    parser.add_argument('--interval', type=int, default=1, help='Extract frames at this interval')
    parser.add_argument('--resize', type=str, help='Resize frames to this dimension (width,height)')
    
    args = parser.parse_args()
    
    # Parse resize argument if provided
    resize_dim = None
    if args.resize:
        try:
            width, height = map(int, args.resize.split(','))
            resize_dim = (width, height)
        except:
            print("Error parsing resize dimensions. Format should be 'width,height'")
            exit(1)
    
    print(f"Extracting frames from {args.video_path}")
    results = extract_frames(args.video_path, args.output_dir, args.interval, resize_dim)
    
    if results['error']:
        print(f"Error: {results['error']}")
    else:
        print(f"Successfully extracted {results['frame_count']} frames")
        print(f"Frames saved to: {results['output_dir']}")
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        print(f"Video info: {results['total_video_frames']} frames, {results['fps']:.2f} fps")