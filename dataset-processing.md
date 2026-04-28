# skill: dataset-processing

## description

Handles dataset preparation for face recognition systems including cleaning, face detection, augmentation, and embedding generation.

---

## workflow

### 1. DATA CLEANING

* Remove corrupted images
* Remove duplicates
* Filter images < 100x100
* Convert all to JPG
* Normalize filenames

---

### 2. FACE DETECTION & CROPPING

* Use MTCNN or OpenCV
* Detect faces in each image
* Crop face region
* Resize to 160x160
* Save in processed/ directory

---

### 3. DATA AUGMENTATION (optional)

* Horizontal flip
* Small rotations
* Brightness/contrast adjustment

---

### 4. LABEL ENCODING

* Assign numeric labels to each person
* Save mapping.json

---

### 5. EMBEDDING GENERATION

* Use DeepFace / FaceNet
* Convert each image to embedding vector
* Save:

  * embeddings.npy
  * labels.npy

---

### 6. MODEL TRAINING

* Use SVM or KNN
* Train on embeddings
* Save model.pkl

---

## folder-structure

dataset/
raw/
processed/
embeddings/

scripts/
clean_dataset.py
detect_and_crop.py
generate_embeddings.py
train_model.py

---

## constraints

* Do NOT skip face detection
* Do NOT train deep CNN from scratch
* Ensure consistent image size

---

## validation

* Check dataset balance
* Verify embeddings shape consistency
* Ensure model loads without error
