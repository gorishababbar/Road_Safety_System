import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import os
import pandas as pd
from tabulate import tabulate
from datetime import timedelta

def detect_video_forgery(video_path, threshold=3, output_dir=None):
    """
    Detect potential video forgery using optical flow analysis.
    
    Args:
        video_path (str): Path to the video file
        threshold (int): Threshold for anomaly detection (default: 3)
        output_dir (str): Directory to save results (default: same as script)
        
    Returns:
        dict: Dictionary containing analysis results including:
            - video_name: Name of the analyzed video
            - forgery_count: Number of potentially forged frames
            - anomaly_frames: List of frame numbers with anomalies
            - anomaly_timestamps: List of timestamps for anomalies
            - anomaly_score: List of anomaly scores for all frames
            - frame_numbers: List of frame numbers
            - plot_path: Path to the saved plot
            - result_path: Path to the saved results file
            - is_forged: Boolean indicating if forgery was detected
    """
    # Check if file exists
    if not os.path.isfile(video_path):
        return {"error": "Video file not found at the specified path."}
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ret, frame1 = cap.read()
    if not ret:
        return {"error": "Could not read the first frame of the video."}
    
    # Initialize variables
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    frame_no = []
    op_flow_per_frame = []
    m = 1
    b = 1
    a = frame1.size
    
    # Process all frames
    while True:
        s = 0
        ret, frame2 = cap.read()
        if ret:
            next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            frame_no.append(m)
            m += 1

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            op_flow_1D = np.resize(mag, (1, a))

            for i in op_flow_1D[0]:
                s += i
            op_flow_per_frame.append(s)
            prvs = next_frame
            b += 1
        else:
            break

    # Compute variation factor
    vrt_factor = [1]
    j = 1
    for o in range(m - 3):
        c = (2 * op_flow_per_frame[j]) / (op_flow_per_frame[j - 1] + op_flow_per_frame[j + 1])
        vrt_factor.append(c)
        j += 1
    vrt_factor.append(1)

    # Round off variation factor
    vrt_factor_round_2 = [round(i, 2) for i in vrt_factor]

    # Compute mean and standard deviation
    mean = round(np.sum(vrt_factor_round_2) / b, 3)
    y = sum((i - mean) ** 2 for i in vrt_factor_round_2)
    st = round(y / b, 3)
    st = math.sqrt(st)

    # Compute anomaly score
    anomaly_score = [abs(i - mean) / st for i in vrt_factor_round_2]

    # Find frames exceeding threshold
    anomaly_frames = [i for i, score in enumerate(anomaly_score) if score > threshold]
    anomaly_timestamps = [frame_num / fps for frame_num in anomaly_frames]
    forgery_count = len(anomaly_frames)
    
    # Format timestamps as human-readable
    formatted_timestamps = []
    for ts in anomaly_timestamps:
        minutes = int(ts // 60)
        seconds = int(ts % 60)
        milliseconds = int((ts % 1) * 1000)
        formatted_timestamps.append(f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}")
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.title('Video Forgery Detection Analysis')
    plt.xlabel('Frame Number')
    plt.ylabel('Anomaly Score')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.plot(frame_no, anomaly_score, label='Anomaly Score')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    video_name = os.path.basename(video_path).split('.')[0]
    plot_path = os.path.join(output_dir, f"{video_name}_forgery_analysis.png")
    plt.savefig(plot_path)
    plt.close()  # Close the plot to avoid displaying it in non-interactive environments
    
    # Prepare result data
    data = [[video_name, forgery_count]]
    df = pd.DataFrame(data, columns=["Video Name", "No. of Forgery Frames"])
    
    # Save results to file
    result_path = os.path.join(output_dir, f"{video_name}_forgery_result.txt")
    with open(result_path, "w") as f:
        f.write(tabulate(df, headers="keys", tablefmt="pretty"))
        f.write("\n\n=== Detected Anomalies ===\n")
        if anomaly_frames:
            for i, (frame, ts) in enumerate(zip(anomaly_frames, formatted_timestamps)):
                f.write(f"Anomaly {i+1}: Frame {frame} (Time: {ts})\n")
        else:
            f.write("No anomalies detected.\n")
    
    # Clean up
    cap.release()
    
    # Prepare return data
    return {
        "video_name": video_name,
        "forgery_count": forgery_count,
        "anomaly_frames": anomaly_frames,
        "anomaly_timestamps": anomaly_timestamps,
        "formatted_timestamps": formatted_timestamps,
        "anomaly_score": anomaly_score,
        "frame_numbers": frame_no,
        "plot_path": plot_path,
        "result_path": result_path,
        "threshold_used": threshold,
        "is_forged": forgery_count > 0
    }

# Test the function if run directly
if __name__ == "__main__":
    # Ask for threshold value
    threshold_input = input("Enter threshold (default is 3): ")
    threshold = int(threshold_input) if threshold_input.strip() else 3
    
    # Get video path
    default_path = os.path.join(os.path.dirname(__file__), "sample.mp4")
    video_path = input(f"Enter video path (default: {default_path}): ")
    
    # Use default path if none provided
    if not video_path.strip():
        video_path = default_path
    
    # Run the analysis
    print(f"Analyzing video: {video_path}")
    print(f"Using threshold: {threshold}")
    
    results = detect_video_forgery(video_path, threshold)
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"\nResults for {results['video_name']}:")
        print(f"Forgery detected: {'Yes' if results['is_forged'] else 'No'}")
        print(f"Total suspicious frames: {results['forgery_count']}")
        
        # Print anomalies if any
        if results['anomaly_frames']:
            print("\nDetected anomalies:")
            for i, (frame, ts) in enumerate(zip(results['anomaly_frames'][:5], results['formatted_timestamps'][:5])):
                print(f"  Anomaly {i+1}: Frame {frame} (Time: {ts})")
            
            if len(results['anomaly_frames']) > 5:
                print(f"  ... and {len(results['anomaly_frames']) - 5} more")
        else:
            print("\nNo anomalies detected.")
        
        print(f"\nResults saved to: {results['result_path']}")
        print(f"Plot saved to: {results['plot_path']}")
        
        # Display the plot
        plt.figure(figsize=(10, 6))
        plt.title('Video Forgery Detection Analysis')
        plt.xlabel('Frame Number')
        plt.ylabel('Anomaly Score')
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
        plt.plot(results['frame_numbers'], results['anomaly_score'], label='Anomaly Score')
        plt.legend()
        plt.grid(True)
        plt.show()