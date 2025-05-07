# 📹 Video Surveillance Summarizer

A lightweight Python project that uses the **YOLOv8 object detection model** to scan surveillance video footage and log detected objects at regular 1-second intervals. It helps generate a quick summary of activity happening in a video.

---

## 📖 What It Does

- Loads a video file.
- Runs YOLOv8 object detection on one frame per second.
- Logs detected objects with their timestamp.
- Saves a summary log to a text file.

---

## 📦 Project Structure

VideoSummarizer/
├── sample.mp4 # Input video file
├── surveillance_summary.txt # Text file with detection summary
├── summarizer_script.py # Python script running YOLOv8 detection


---

## 📚 Dependencies

Install the required packages using:

pip install opencv-python ultralytics

---

## 🚀 How to Run

1️⃣ Install the dependencies:

pip install opencv-python ultralytics

2️⃣ Place your video file (e.g., sample.mp4) in the project directory.

3️⃣ Run the Python script:

python summarizer_script.py

4️⃣ View the generated surveillance_summary.txt for logged detections.