# 🚀 NeuroVision: AI-Powered Face Recognition

**NeuroVision** is a robust facial recognition system leveraging deep neural networks to identify individuals with high precision. This project focuses on a complete data engineering pipeline—from raw social media data collection to real-time inference.

![AI](https://img.shields.io/badge/AI-MTCNN%20%2B%20FaceNet-orange)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-emerald)
![Dataset](https://img.shields.io/badge/Data-Facebook%20Sourced-blue)

## 🧠 AI & Data Engineering Pipeline

The core of NeuroVision is its multi-stage neural pipeline, designed to handle real-world variations in lighting, pose, and image quality.

### 1. Data Collection & Preprocessing
- **Source:** Raw image datasets collected from Facebook profiles of friends and colleagues.
- **Cleaning:** Automated script to remove low-quality frames and non-face images.
- **Face Detection (MTCNN):** Uses **Multi-task Cascaded Convolutional Networks** to detect faces and align them by identifying 5 key facial landmarks (eyes, nose, mouth corners).
- **Cropping:** Faces are extracted and resized to a standard 160x160 pixels for the embedding engine.

### 2. Feature Extraction (FaceNet)
- **Deep Learning:** Utilizes the **FaceNet** model (Inception ResNet v1) to map face images to a 128-dimensional Euclidean space.
- **Embeddings:** These "face signatures" ensure that images of the same person have small distances between them, while different people are far apart.

### 3. Classification & Training
- **Machine Learning:** A **Support Vector Machine (SVM)** classifier is trained on the generated 128D embeddings.
- **Optimization:** Probability calibration is enabled to provide accurate confidence scores for each prediction.

---

## 🏗️ Technical Architecture

### 🛡️ Backend Engine
- **Framework:** FastAPI for high-concurrency request handling.
- **Service Layer:** Modular recognition service that initializes the neural models once and reuses them for inference.
- **Inference Latency:** Optimized for sub-100ms response times on standard CPU hardware.

### 🎨 Frontend Interface
- **Modern UI:** A sleek "Ember" themed React interface for interacting with the AI.
- **Features:** Supports both local file uploads and **Real-Time Camera** snapshots.
- **Visuals:** Features neural-themed background animations and real-time scanning feedback.

---

## 🛠️ Setup & Execution

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Execute Neural Pipeline
Initialize the system by processing the Facebook-sourced dataset:
```bash
# 1. Clean raw data
python scripts/clean_dataset.py

# 2. Detect, align, and crop faces (MTCNN)
python scripts/detect_and_crop.py

# 3. Generate 128D embeddings (FaceNet)
python scripts/generate_embeddings.py

# 4. Train the SVM classifier
python scripts/train_model.py
```

### 3. Deployment
```bash
# Start Backend (Port 8001)
uvicorn backend.app.main:app --host 0.0.0.0 --port 8001

# Start Frontend
cd frontend && pnpm run dev
```

---

## ⚡ Stack
- **AI/ML:** TensorFlow, MTCNN, FaceNet, Scikit-Learn.
- **API:** FastAPI, Uvicorn.
- **UI:** React, Tailwind v4, Lucide Icons.

---

Powered by **MTCNN + FaceNet + SVM**
