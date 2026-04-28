import os
import cv2
import numpy as np
import joblib
import json
from mtcnn import MTCNN
from keras_facenet import FaceNet
import gc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FaceRecognitionService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceRecognitionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        print("Initializing FaceRecognitionService...")
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        # Configuration from environment variables
        model_dir_relative = os.getenv("MODEL_DIR", "processed_dataset")
        self.model_dir = os.path.join(self.project_root, model_dir_relative)
        
        # Load Model
        model_path = os.path.join(self.model_dir, 'model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        self.model = joblib.load(model_path)
        
        # Load Label Mapping
        mapping_path = os.path.join(self.model_dir, 'label_mapping.json')
        with open(mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
            
        # Initialize ML Models
        self.detector = MTCNN()
        self.embedder = FaceNet()
        
        self._initialized = True
        print("FaceRecognitionService initialized successfully.")

    def recognize(self, image_path):
        """
        Processes an image, detects a face, generates an embedding, and predicts identity.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not read image"}

            # Resize if large (to match training logic and save memory)
            h, w = image.shape[:2]
            if max(h, w) > 1024:
                scale = 1024 / max(h, w)
                image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 1. Detect Face
            detections = self.detector.detect_faces(image_rgb)
            if not detections:
                return {"error": "No face detected"}

            # Take the largest/first face
            x, y, width, height = detections[0]['box']
            x, y = max(0, x), max(0, y)
            face_image = image[y:y+height, x:x+width]
            
            if face_image.size == 0:
                return {"error": "Invalid face crop"}

            # 2. Resize for Embedding (FaceNet expects 160x160 usually, keras-facenet handles resizing internally but we'll do it to be safe)
            face_image_resized = cv2.resize(face_image, (160, 160), interpolation=cv2.INTER_AREA)
            face_rgb = cv2.cvtColor(face_image_resized, cv2.COLOR_BGR2RGB)
            
            # 3. Generate Embedding
            face_batch = np.expand_dims(face_rgb, axis=0)
            embedding = self.embedder.embeddings(face_batch)[0]
            
            # 4. Predict
            # model.predict_proba returns probabilities for each class
            embedding = embedding.reshape(1, -1)
            probabilities = self.model.predict_proba(embedding)[0]
            prediction_idx = np.argmax(probabilities)
            confidence = probabilities[prediction_idx]
            
            person_name = self.label_mapping[str(prediction_idx)]

            # Cleanup
            del image
            del image_rgb
            del face_image
            gc.collect()

            return {
                "person": person_name,
                "confidence": float(confidence),
                "box": [int(x), int(y), int(width), int(height)]
            }

        except Exception as e:
            print(f"Error in recognition service: {e}")
            return {"error": str(e)}

# Export a single instance
recognition_service = FaceRecognitionService()
