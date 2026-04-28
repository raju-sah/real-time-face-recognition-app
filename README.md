# 🚀 NeuroVision: Advanced AI Face Recognition

**NeuroVision** is a state-of-the-art, production-ready facial recognition ecosystem. It combines high-performance neural networks with a premium, immersive user interface to provide real-time biometric identification.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-rose)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-emerald)
![AI](https://img.shields.io/badge/AI-MTCNN%20%2B%20FaceNet-orange)

## ✨ Features

### 🖥️ Premium Frontend (NeuroVision UI)
- **Ember Theme:** A modern Rose & Orange aesthetic with deep-layer dark modes.
- **Glassmorphism:** High-end translucent containers with backdrop blur effects.
- **Animated Backgrounds:** Immersive mesh gradients, floating data particles, and dynamic neural network SVG animations.
- **Real-Time Camera:** Support for direct webcam capture and biometric snapshots within the browser.
- **Responsive Design:** Optimized for both desktop and high-resolution displays with balanced card layouts.

### 🧠 Advanced AI Pipeline
- **Neural Processing:** Automated face detection using **MTCNN** and high-precision embedding generation via **FaceNet**.
- **Accuracy:** SVM-based classification for lighting-fast and highly accurate identity matching.
- **Live Scanning:** Visual "neural scan" animations during inference to represent active computation.

---

## 📂 Project Structure
- `datasets/`: Raw source images grouped by subject.
- `scripts/`: Full data lifecycle scripts (Clean → Crop → Embed → Train).
- `processed_dataset/`: Binary artifacts, neural embeddings, and the trained model.
- `backend/`: Scalable FastAPI application with modular service architecture.
- `frontend/`: Premium React + TypeScript + Tailwind v4 application.

---

## 🛠️ Setup Instructions

### 1. Environment Setup
Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Neural Pipeline Execution
Initialize the system by processing your dataset:
```bash
# 1. Clean raw data
python scripts/clean_dataset.py

# 2. Detect and crop faces
python scripts/detect_and_crop.py

# 3. Generate embeddings
python scripts/generate_embeddings.py

# 4. Train the classifier
python scripts/train_model.py
```

### 3. Deploy Backend (FastAPI)
The backend service handles the heavy lifting of neural inference.
```bash
# From project root
uvicorn backend.app.main:app --host 0.0.0.0 --port 8001
```
*API will run at http://localhost:8001*

### 4. Deploy Frontend (NeuroVision)
The immersive UI provides the portal for biometric interaction.
```bash
cd frontend
pnpm install
pnpm run dev
```
*Frontend will run at http://localhost:5173 (or 5174)*

---

## 📖 Usage Guide
1. **Initialize Identity:** Choose between **File Upload** or **Live Camera** mode.
2. **Neural Capture:** Select an image or capture a live biometric snapshot.
3. **Execute Recognition:** Click the "Initialize Identification" button to begin neural analysis.
4. **Analysis Snapshot:** View the identified subject, confidence scores, and processing latency in the result panel.

---

## ⚡ Tech Stack
- **Frontend:** React 18, Vite, Tailwind CSS v4, Lucide Icons, Framer-style CSS Animations.
- **Backend:** FastAPI, Python 3.10+, TensorFlow, Keras-FaceNet.
- **Inference:** MTCNN (Face Detection), FaceNet (Embeddings), Scikit-Learn (Classification).

---

Powered by **FastAPI + React + MTCNN + FaceNet**
