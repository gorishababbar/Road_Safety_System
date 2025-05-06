import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import math
import os
import pandas as pd
from tabulate import tabulate

# Get threshold value from user
th = int(input("Enter threshold: "))

# Manually set the path to your video file
video_path = r"C:\Users\ASUS\Desktop\Major Project\Tamper-detection-in-digital-videos\dataset\video.mp4"

# Check if file exists
if not os.path.isfile(video_path):
    print("Video file not found at the specified path.")
    exit()

# Initialize lists
no_of_forgery = []
video_name = []

# Process the single video
cap = cv2.VideoCapture(video_path)
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
frame_no = []
op_flow_per_frame = []
m = 1
b = 1
a = frame1.size
s = np.arange(a)

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

# Count frames exceeding threshold
bv = sum(1 for i in anomaly_score if i > th)
no_of_forgery.append(bv)
video_name.append(os.path.basename(video_path))

# Add snippet to detect and print anomalies (frames exceeding threshold)
fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
threshold = th  # Example threshold

anomaly_frames = [i for i, score in enumerate(anomaly_score) if score > threshold]

for frame_num in anomaly_frames:
    timestamp = frame_num / fps  # Convert frame number to timestamp
    print(f"Anomaly detected at frame {frame_num} (Time: {timestamp:.2f} seconds)")

# Prepare result data
data = [[video_name[0], no_of_forgery[0]]]
df = pd.DataFrame(data, columns=["Video Name", "No. of Forgery Frames"])
filename = "video_forgery_result.txt"

# Plot anomaly score
plt.title('Video forgery')
plt.xlabel('Frame Number')
plt.ylabel('Anomaly Score')
plt.plot(frame_no, anomaly_score)
plt.show()

# Write result to file
with open(filename, "w") as f:
    f.write(tabulate(df, headers="keys", tablefmt="pretty"))

print(df)

# Clean up
cv2.destroyAllWindows()
cap.release()
