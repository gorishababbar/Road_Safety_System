# 🚦 Traffic Anomaly Detection

A system that detects unusual or anomolous traffic behaviors in surveillance videos using deep learning and object detection.

---

## 📦 Project Structure

Traffic_Anomaly_Detection/
├── main.py # Core anomaly detection script
├── requirements.txt # Required Python libraries
├── readme.md # Project documentation
├── sample.mp4 # Sample test video
├── yolov8m.pt # YOLOv8 pretrained model for object detection

---

## 📖 What It Does

- Detects red light violation and trespassing 
- Uses YOLOv8 object detection for vehicle tracking.

---

## 📥 How to Run

1️⃣ Install dependencies:
pip install -r requirements.txt

2️⃣ Run anomaly detection:
python main.py

3️⃣ Check output video with alerts.

---

## 📊 Expected Output

| Frame | Anomaly Detected |
|:------|:----------------|
| 45    | Red light violated        |
| 112   | Lane Violation            |

---

## 📌 Future Scope

- Add further functionalities such as helmet detection, overspeeding, wrong lane trespassing etc
- Integrate GPS data for location-aware alerts.
- Push real-time notifications to traffic control systems.

---
