import os
import cv2
import numpy as np
import json
from mtcnn import MTCNN
from keras_facenet import FaceNet
import gc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

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
        
        # Load Embeddings and Labels
        embeddings_path = os.path.join(self.model_dir, 'embeddings.npy')
        labels_path = os.path.join(self.model_dir, 'labels.npy')
        
        if not os.path.exists(embeddings_path) or not os.path.exists(labels_path):
            raise FileNotFoundError("Embeddings or labels not found. Please train/generate embeddings first.")
            
        self.known_embeddings = np.load(embeddings_path)
        self.known_labels = np.load(labels_path)
        
        # Load Label Mapping
        mapping_path = os.path.join(self.model_dir, 'label_mapping.json')
        with open(mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
            
        # Initialize ML Models
        self.detector = MTCNN()
        self.embedder = FaceNet()
        
        self.SIMILARITY_THRESHOLD = 0.40  # Standard threshold for FaceNet
        
        self._initialized = True
        print("FaceRecognitionService initialized successfully with Cosine Similarity.")

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
            
            # 4. Predict using Cosine Similarity
            similarities = cosine_similarity(self.known_embeddings, embedding)
            
            max_idx = np.argmax(similarities)
            max_similarity = similarities[max_idx]
            
            if max_similarity >= self.SIMILARITY_THRESHOLD:
                prediction_idx = self.known_labels[max_idx]
                person_name = self.label_mapping[str(prediction_idx)]
            else:
                person_name = "Unknown"

            # Cleanup
            del image
            del image_rgb
            del face_image
            gc.collect()

            return {
                "person": person_name,
                "confidence": float(max_similarity),
                "box": [int(x), int(y), int(width), int(height)]
            }

        except Exception as e:
            print(f"Error in recognition service: {e}")
            return {"error": str(e)}

# Export a single instance
recognition_service = FaceRecognitionService()
