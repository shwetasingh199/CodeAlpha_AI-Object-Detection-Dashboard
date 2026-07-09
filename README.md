# 🎯 AI Object Detection & Tracking Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

An industry-inspired AI dashboard for **Real-Time Object Detection and Tracking** using **YOLOv8**, **OpenCV**, and **Streamlit**.

</div>

---

# 📌 Project Overview

This project is a modern Computer Vision application capable of detecting and tracking multiple objects from **images** and **videos** using the Ultralytics YOLOv8 model.

The dashboard provides an intuitive interface where users can upload media, visualize detections, analyze detected objects, and download the processed results.

---

# ✨ Features

✅ Image Object Detection

✅ Video Object Detection

✅ YOLOv8 Pre-trained COCO Model

✅ Object Labels

✅ Confidence Scores

✅ Object Counting

✅ Interactive Streamlit Dashboard

✅ Detection Summary Table

✅ Object Distribution Chart

✅ Download Processed Video

✅ Responsive UI

---

# 🖼 Dashboard Preview

## Dashboard

<img width="1115" height="882" alt="Screenshot 2026-07-09 140954" src="https://github.com/user-attachments/assets/e865e753-489a-4e7d-bdb5-b84c8d89e99e" />

---

## Object Detection

<img width="1007" height="880" alt="Screenshot 2026-07-09 141136" src="https://github.com/user-attachments/assets/b43318d9-b75d-4aee-ab35-f9a78dce5e8f" />
<img width="1091" height="808" alt="Screenshot 2026-07-09 141017" src="https://github.com/user-attachments/assets/3a0c6c75-576a-450b-9d08-fe4f0ee843ed" />
<img width="717" height="822" alt="Screenshot 2026-07-09 141155" src="https://github.com/user-attachments/assets/59d60187-9f23-46e8-a12f-9e540ccfeadd" />

---

# 📂 Folder Structure

```text
ObjectDetectionDashboard/
│
├── app.py
├── detector.py
├── processor.py
├── utils.py
├── requirements.txt
├── README.md
│
├── assets/
│   └── logo.png
│
├── outputs/
│
├── screenshots/
│   ├── dashboard.png
│   ├── image_detection.png
│   └── video_detection.png
│
└── sample/
    └── sample_video.mp4
```

---

# ⚙ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Dashboard Development |
| OpenCV | Image & Video Processing |
| YOLOv8 | Object Detection |
| NumPy | Numerical Operations |
| Pandas | Data Analysis |
| Plotly | Interactive Charts |
| Pillow | Image Handling |

---

# 🧠 Model

Model Used:

```
YOLOv8 Nano (yolov8n.pt)
```

Dataset:

```
MS COCO Dataset
```

Supported Objects:

- Person
- Car
- Bus
- Truck
- Dog
- Cat
- Bicycle
- Motorcycle
- Chair
- Bottle

and 80 COCO classes.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Object-Detection-Dashboard.git
```

Move into the project

```bash
cd AI-Object-Detection-Dashboard
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🎮 How to Use

### Image Mode

1. Select **Image** from the sidebar.
2. Upload an image.
3. View detected objects.
4. Analyze object statistics.

### Video Mode

1. Select **Video**.
2. Upload a video.
3. Click **Start Detection**.
4. Wait for processing.
5. Download the processed video.

---

# 📊 Dashboard Features

✔ Total Objects

✔ Object Categories

✔ Detection Confidence

✔ Object Distribution Graph

✔ Detection Summary Table

✔ Download Processed Results

---

# 📈 Sample Output

| Object | Count |
|---------|------:|
| Person | 5 |
| Car | 3 |
| Dog | 1 |

---

# 💻 Requirements

```text
streamlit
opencv-python
ultralytics
numpy
pandas
plotly
Pillow
torch
```

Install using:

```bash
pip install -r requirements.txt
```

---

# 🔮 Future Improvements

- Real-Time Webcam Detection
- ByteTrack Object Tracking
- Persistent Tracking IDs
- Live FPS Counter
- Object Trails
- Region of Interest (ROI)
- Line Crossing Counter
- CSV Report Export
- Detection History
- Dark/Light Theme Support

---

# 👩‍💻 Developer

**Shweta Singh**

Electronics & Computer Engineering

Passionate about Artificial Intelligence, Computer Vision, and Machine Learning.

---

# 📜 License

This project is licensed under the MIT License.

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
