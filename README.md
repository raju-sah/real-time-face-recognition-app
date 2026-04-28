# Real-Time Face Recognition System

A production-ready face recognition system built with FastAPI, React, MTCNN, and FaceNet.

## Features
- **Data Pipeline:** Automated cleaning, face detection (MTCNN), and embedding generation (FaceNet).
- **ML Model:** SVM classifier for fast and accurate identity prediction.
- **Backend:** FastAPI with modular service-based architecture.
- **Frontend:** Modern React + TypeScript + TailwindCSS UI.

---

## Project Structure
- `datasets/`: Raw images grouped by person name.
- `scripts/`: Data pipeline scripts (clean, crop, embed, train).
- `processed_dataset/`: Cleaned data, embeddings, and trained model.
- `backend/`: FastAPI application.
- `frontend/`: React application.

---

## Setup Instructions

### 1. Environment Setup
Create a virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scikit-learn Pillow opencv-python mtcnn tensorflow keras-facenet joblib fastapi uvicorn python-multipart
```

### 2. Run Data Pipeline
Execute these in order to prepare the system:
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

### 3. Start Backend (FastAPI)
```bash
# From project root
export PYTHONPATH=$PYTHONPATH:$(pwd)
python backend/app/main.py
```
*API will run at http://localhost:8000*

### 4. Start Frontend (React)
```bash
cd frontend
pnpm install
pnpm run dev
```
*Frontend will run at http://localhost:5173*

---

## Usage
1. Open the frontend in your browser.
2. Upload an image containing a face.
3. Click "Identify Person".
4. View the predicted identity and confidence score.
