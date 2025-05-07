# 🚘 Vehicle Number Plate Identification

A YOLOv8-powered project that identifies vehicle number plates from video footage for traffic management and law enforcement. Logs the relevant data into a database.

---

## 📦 Project Structure

Vehicle_Number_Plate_Identification/
├── main.py # Number plate detection and recognition script
├── best.pt # Trained YOLOv8 model for number plate detection
├── readme.md # Project documentation
├── sample.mp4 # Sample video for testing

---

## 📖 What It Does

- Detects and localizes number plates in video frames.
- Extracts and displays plate regions.
- Leverages OCR for text extraction so as to store the number plate text.
- Predicts vehicle speed, timeframe etc for each vehicle in the video file.
- Stores all the data inside an sqlite database.

---

## 📥 How to Run

1️⃣ Install dependencies:
pip install -r requirements.tx

2️⃣ Run detection:
python main.py

3️⃣ Check the database with detected number plates.

---

## 📊 Example Output

| Frame | Number Plate Detected | Time frame | Estimated Speed |
|:------|:---------------------|:------------|:----------------|
| 23    | MH12AB1234            | 14:00 sec  | 40 kmph         |
| 87    | DL4CAF5678            | 16:00 sec  | 37 kmph         |

---

## 📌 Future Scope

- Real-time video stream support.
- Linking to traffic violation database.

---