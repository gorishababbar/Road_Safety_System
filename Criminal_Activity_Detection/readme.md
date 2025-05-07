# 📹 Crime Detection in Videos

This is a video classification project designed to detect criminal activities in video footage. It uses a deep learning model trained on the **UCF Crime Dataset**, a popular dataset for crime video classification research. The model predicts whether a video belongs to a certain crime category like fighting, robbery, normal activity, etc.

---

## 📦 Project Structure

Videoclassification/
├── assets/ # Images or resources (to be added)
├── extracted_frames/ # Frames extracted from input videos (created after the extraction script is run)
├── UCF/ # UCF Crime Dataset (organized in folders)
├── frame_extraction.ipynb # Notebook to extract video frames
├── train.ipynb # Model training notebook
├── main.ipynb # Crime Classification and prediction notebook
├── trained_model.h5 # Trained Keras model file
├── label_classes.npy # Numpy array containing class labels
├── features.pkl # Pickle file of extracted features

yaml
Copy
Edit

---

## 📖 What It Does

- Extracts frames from videos.
- Uses a trained deep learning model to classify each video.
- Detects possible criminal activity based on frame sequences.
- Outputs prediction labels such as **Normal**, **Fight**, **Robbery**, etc.

---

## 📚 Dataset: UCF Crime Dataset

The project uses the **UCF-Crime dataset** — a large-scale real-world surveillance dataset with **8 crime categories** including:

- Assault
- Burglary
- Robbery
- Fighting
- Normal activities  
…and others.

> **Note:** Dataset needs to be downloaded and placed in the `UCF/` directory.

**📥 [Download dataset here](https://www.kaggle.com/datasets/mission-ai/crimeucfdataset?resource=download)**

---

## 🚀 How to Run

### 1️⃣ Clone or download the repo

### 2️⃣ Setup Environment

- Install Anaconda (for Jupyter Notebook)
- Install Python
- Run: pip install -r requirements.txt

### 3️⃣ Extract Frames

- Run `frame_extraction.ipynb` to convert videos into image frames.

### 4️⃣ Train Model (optional)

- Run `train.ipynb` to train your model.
- *(Skip if you're using `trained_model.h5` provided in the repo)*

### 5️⃣ Classify Videos

- Run `main.ipynb`
- Load a video, process it, and get crime detection prediction.

---

## 📊 Example Output

| Video            | Prediction |
|:----------------|:------------|
| fight_001.mp4    | Fight       |
| burglary_014.mp4 | Burglary    |
| normal_003.mp4   | Normal      |

---

## 📌 Future Scope

- Integrating real-time CCTV feed detection
- Adding action summarization using vision-language models