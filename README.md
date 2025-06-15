# Posture Detection Project

A real-time ergonomic posture analyzer built with Python, MediaPipe Pose & OpenCV. Computes joint angles, classifies “good” vs. “slouching” posture, overlays visual feedback on webcam feed, and logs session statistics for offline analysis.

---

## 🔍 Features

- **Pose Estimation**  
  Uses MediaPipe Pose to extract 33 3D landmarks per frame (shoulders, spine, neck, hips, etc.).

- **Kinematic Computations**  
  Calculates key joint angles (e.g. back tilt, neck flexion) via NumPy and compares against calibrated thresholds.

- **Real‑Time Visualization**  
  Overlays landmarks, angle values, and posture status (✅ Good / ⚠️ Slouching) on live webcam feed with OpenCV.

- **Session Logging & Analytics**  
  Streams timestamped posture labels into Pandas DataFrames, exports CSV for time‑series analysis or dashboarding.

- **Demo Notebook**  
  Jupyter Notebook walkthrough showing setup, landmark plotting, threshold tuning, and interactive session reporting.


## 🚀 Quick Start

1. **Clone the repo**  
   ```bash
   git clone https://github.com/zainibaloch/Posture-Detection-Project.git
   cd Posture-Detection-Project


2. **Create & activate virtual environment**

   ```bash
   python3 -m venv venv        #conda create -n newenv python==3.10 --y
   source venv/bin/activate    # Linux/Mac
   venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install streamlit opencv-python mediapipe tensorflow numpy pillow
   ```

4. **Run real‑time detection**

   ```bash
   streamlit run app.py
   ```

5. **Explore the demo Notebook**
   Open `demo/Posture_Detection_Demo.ipynb` in Jupyter to see angle calculations, threshold tuning, and data logging examples.

---

## 📁 Project Structure Overview

- **Driver Safety**: CSVs, preprocessing.pkl, analysis notebook
- **Exercise**: 
  - Yoga pose folders (train/test)
  - Model training script, evaluation report & confusion matrix
- **Handley**: YOLOv5 model config, training output
- **Sitting**: Image dataset (train/test/val), result folders, model script
- **Main Files**: `main.py`, README, and a screenshot

> ⚠️ Images and model weights are excluded. DM for full access.



---

## ⚙️ Dependencies

* Python 3.10+
* `opencv-python`
* `mediapipe`
* `numpy`
* `pandas`

Install via:

```bash
pip install opencv-python mediapipe numpy pandas
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/awesome`)
3. Commit your changes (`git commit -m "Add awesome feature"`)
4. Push to branch (`git push origin feature/awesome`)
5. Open a Pull Request


