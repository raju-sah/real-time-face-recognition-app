# skill: fastapi-backend

## description

Builds a modular FastAPI backend for face recognition with clean architecture and ML service integration.

---

## workflow

### 1. PROJECT STRUCTURE

backend/
app/
main.py
routes/
services/
models/
utils/

---

### 2. SETUP

* Install:
  fastapi
  uvicorn
  python-multipart
  opencv-python
  deepface
  numpy
  scikit-learn

---

### 3. CORE ENDPOINTS

#### POST /upload

* Accept image file
* Save to uploads/

#### POST /recognize

* Input image
* Detect face
* Generate embedding
* Load model.pkl
* Predict identity
* Return:
  {
  "person": "name",
  "confidence": 0.92
  }

---

### 4. SERVICE LAYER

Move logic to services/:

* face_detection.py
* embedding_service.py
* recognition_service.py

Rules:

* No ML logic in routes
* Keep routes thin

---

### 5. MODEL LOADING

* Load model at startup
* Cache embeddings
* Avoid recomputation

---

### 6. CORS

* Enable for frontend access

---

## constraints

* Do NOT mix business logic in routes
* Do NOT reload model per request
* Handle invalid images safely

---

## validation

* Test endpoints with Postman
* Ensure response format consistency
* Handle errors gracefully
