# 🚀 NeuroVision: Real-Time Face Recognition

**NeuroVision** is a real-time face recognition system with **live enrollment** and **instant multi-face identification**. Users enroll by capturing guided multi-angle samples from their camera — no pre-training on a static dataset required. The system learns new faces on the fly via face embeddings and identifies every person in the live camera feed simultaneously.

![AI](https://img.shields.io/badge/AI-MTCNN%20%2B%20FaceNet-orange)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-emerald)
![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-blue)

## ✨ Features

- **Live Guided Enrollment** — camera-based capture of 6 head poses (front, left, right, up, down, front). The system automatically captures a sample only when your head is at the correct angle.
- **Real-Time Training** — no model retraining. New faces are added instantly as 512-D FaceNet embeddings into a persistent gallery.
- **Automatic Naming** — leave the name empty and the backend generates a unique human-friendly label (`TestUser-1`, `TestUser-2`, ...). Counter persists across restarts.
- **Add Samples to Existing Users** — improve recognition accuracy by appending more angle/lighting samples to an existing profile. Near-identical duplicates are skipped automatically.
- **Real-Time Multi-Face Recognition** — the live camera identifies **all** faces in the frame at once, drawing a name + confidence label above each bounding box (green = known, red = unknown). No captures needed.
- **Persistent Storage** — enrolled faces survive server restarts (stored in `backend/gallery/*.json`).
- **Duplicate Detection** — re-enrolling a face that already exists is detected and reported.
- **Quality Gates** — blurry, too-small, or badly-lit frames are rejected with clear feedback.
- **Upload Mode** — single-image identification as a fallback.

## 🧠 How It Works

1. **Face Detection (MTCNN)** — locates faces and extracts 5 facial keypoints.
2. **Pose Estimation** — head yaw/pitch computed from keypoints to guide enrollment angles and validate sample quality.
3. **Embedding (FaceNet)** — each face is mapped to a 512-D vector ("face signature").
4. **Similarity Matching** — cosine similarity against the gallery; above a threshold → matched person, otherwise `Unknown`.
5. **Enrollment** — the guided flow appends embeddings per pose and persists the profile to JSON on completion.

## 🏗️ Technical Architecture

### 🛡️ Backend (FastAPI)

| Endpoint | Purpose |
|---|---|
| `POST /enroll/start` | Create enrollment session (`name`, optional `existing_user_id`) |
| `POST /enroll/sample` | Send camera frame + target pose; auto-captures when pose/quality OK |
| `POST /enroll/complete` | Persist new profile or append samples to existing one |
| `POST /enroll/abort` | Discard enrollment session |
| `GET /users` | List enrolled profiles |
| `DELETE /users/{user_id}` | Remove a profile |
| `POST /recognize` | Single-image recognition (multipart) |
| `POST /recognize/base64` | Live-frame recognition returning **all** faces |

### 🎨 Frontend (React + Vite + Tailwind)

- **Enroll New Face** — pose-guided capture with live yaw/pitch feedback, progress grid, optional name input, and existing-profile dropdown to add more samples.
- **Live Recognition** — streams frames every ~400 ms and overlays a labeled bounding box on every face.
- **Upload & Recognize** — identifies all faces in an uploaded image with per-face confidence bars.

### 📁 Persistence

Each enrolled person is stored as JSON in `backend/gallery/`:
```json
{
  "name": "TestUser-1",
  "user_id": "a1b2c3...",
  "created_at": "...",
  "poses": ["front", "left", "right", "up", "down", "front"],
  "embeddings": [[512 floats...], ...]
}
```

## 🛠️ Setup & Execution

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend && pnpm install
```

### 2. Deployment
```bash
# Start Backend (default port 8000 — set BACKEND_PORT in .env to change)
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# Start Frontend
cd frontend && pnpm run dev
```

> Frontend calls `VITE_API_URL` (default `http://localhost:8000`). Keep it aligned with the backend port in `frontend/.env`.

## ⚡ Stack
- **AI/ML:** TensorFlow, Keras, MTCNN, keras-facenet (FaceNet Inception ResNet v1).
- **API:** FastAPI, Uvicorn, OpenCV, NumPy.
- **UI:** React 18, Vite, Tailwind CSS v4, Lucide Icons, Axios.

---

## 🤝 Contact

- **Portfolio:** [sahraju.com.np](https://sahraju.com.np)
- **Email:** [try.rajusah@gmail.com](mailto:try.rajusah@gmail.com)
- **LinkedIn:** [linkedin.com/in/rajusah18](https://linkedin.com/in/rajusah18)
- **GitHub:** [github.com/raju-sah](https://github.com/raju-sah)
- **Instagram:** [instagram.com/okay.raju](https://instagram.com/okay.raju)

---

Powered by **MTCNN + FaceNet**
