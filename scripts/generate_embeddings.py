import os
import cv2
import numpy as np
from keras_facenet import FaceNet
import json

def generate_embeddings(input_dir, output_dir):
    """
    Generates embeddings for cropped face images using FaceNet.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading FaceNet model...")
    embedder = FaceNet()
    print("FaceNet model loaded.")

    embeddings = []
    labels = []
    label_mapping = {}
    
    people = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
    
    for i, person_name in enumerate(people):
        label_mapping[i] = person_name
        person_path = os.path.join(input_dir, person_name)
        
        print(f"Generating embeddings for: {person_name}")
        
        for filename in os.listdir(person_path):
            file_path = os.path.join(person_path, filename)
            
            if not os.path.isfile(file_path):
                continue
                
            try:
                # Load image
                image = cv2.imread(file_path)
                if image is None:
                    continue
                
                # Convert to RGB (FaceNet expects RGB)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # FaceNet.embeddings() expects a batch of images
                # Since we have one image, we add an extra dimension
                face_batch = np.expand_dims(image_rgb, axis=0)
                
                # Generate embedding
                det_embeddings = embedder.embeddings(face_batch)
                
                embeddings.append(det_embeddings[0])
                labels.append(i)
                
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                continue

    # Convert to numpy arrays
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Save results
    np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    
    with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
        json.dump(label_mapping, f, indent=4)

    print(f"\nEmbedding generation complete.")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cropped_faces_dir = os.path.join(project_root, 'cropped_faces')
    embeddings_output_dir = os.path.join(project_root, 'processed_dataset')
    
    generate_embeddings(cropped_faces_dir, embeddings_output_dir)
