# 📹 Video Authentication System

A tool to verify the authenticity of surveillance or recorded videos using metadata analysis and frame validation.

---

## 📦 Project Structure

Video_Authentication/
├── main.py # Authentication logic and frame analysis
├── readme.md # Project documentation
├── sample.mp4 # Sample video for demo

---

## 📖 What It Does

- Validates video authenticity by:
  - Checking frame sequence integrity.
  - Analyzing metadata (FPS, codec, etc.).
- Detects tampering like frame drops or edits.
- Plots a graph for tampering vs frame number.

---

## 📥 How to Run

1️⃣ Install the dependencies:
Run pip install -r requirements.txt


2️⃣ Authenticate a video:
python main.py

3️⃣ Review the authenticity report generated.

---

## 📊 Expected Output

| Time              | Result                |
|:------------------|:----------------------|
| 01.51 sec         | Tampering detected    |
| 03.05 sec         | Tampering detected    |

---

## 📊 Expected Output (after further improvements)

| Check              | Result |
|:------------------|:--------|
| Frame Consistency  | Pass    |
| Metadata Tampering | Detected|

---

## 📌 Future Scope

- Integrating blockchain-based video fingerprinting.
- Integrating deepfake techniques.
- Real-time video integrity monitoring.

---