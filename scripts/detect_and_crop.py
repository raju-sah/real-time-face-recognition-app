import os
import cv2
import numpy as np
from mtcnn import MTCNN
import gc

def resize_image_if_large(image, max_dim=1024):
    """Resizes image if its height or width exceeds max_dim, maintaining aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
    return image, 1.0

def detect_and_crop_faces(input_dir, output_dir, target_size=(160, 160)):
    """
    Detects faces using MTCNN with memory optimizations and pre-resizing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing MTCNN detector...")
    detector = MTCNN()
    print(f"MTCNN Initialized. Processing {input_dir}")

    total_images = 0
    faces_found = 0

    people = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    for person_name in people:
        person_input_path = os.path.join(input_dir, person_name)
        person_output_path = os.path.join(output_dir, person_name)
        os.makedirs(person_output_path, exist_ok=True)
        
        print(f"\n---> Person: {person_name}")
        
        filenames = sorted(os.listdir(person_input_path))
        for filename in filenames:
            file_path = os.path.join(person_input_path, filename)
            output_file_path = os.path.join(person_output_path, filename)
            
            if not os.path.isfile(file_path) or filename.startswith('.'):
                continue

            if os.path.exists(output_file_path):
                continue

            try:
                # 1. Load Image
                image = cv2.imread(file_path)
                if image is None:
                    continue

                # 2. Resize for memory efficiency and better detection performance
                # This fixes the "Allocation of 288000000 exceeds 10% of free system memory" warning
                working_image, scale = resize_image_if_large(image, max_dim=1024)
                
                # 3. Convert to RGB
                image_rgb = cv2.cvtColor(working_image, cv2.COLOR_BGR2RGB)
                
                print(f"  Detecting: {filename} (Resized to {working_image.shape[1]}x{working_image.shape[0]})...", flush=True)
                
                # 4. Detect Faces
                detections = detector.detect_faces(image_rgb)
                total_images += 1

                if detections:
                    # Take the first detected face
                    x, y, width, height = detections[0]['box']
                    
                    # Ensure coordinates are positive
                    x, y = max(0, x), max(0, y)
                    
                    # Crop from the working image
                    face_image = working_image[y:y+height, x:x+width]
                    
                    if face_image.size > 0:
                        face_image_resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_AREA)
                        cv2.imwrite(output_file_path, face_image_resized)
                        faces_found += 1
                else:
                    print(f"    [!] No face found in {filename}.")

                # 5. Cleanup memory
                del image
                del working_image
                del image_rgb
                del detections
                gc.collect()

            except Exception as e:
                print(f"    [Error] {filename}: {e}")
                gc.collect()
                continue
    
    print(f"\nFinished. Total processed: {total_images}, Faces found: {faces_found}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dataset_dir = os.path.join(project_root, 'processed_dataset')
    cropped_faces_dir = os.path.join(project_root, 'cropped_faces')
    
    detect_and_crop_faces(processed_dataset_dir, cropped_faces_dir)
