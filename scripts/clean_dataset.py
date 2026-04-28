import os
import cv2
from PIL import Image
import numpy as np

def clean_and_process_dataset(input_dir, output_dir, min_size=(100, 100)):
    """
    Cleans the raw dataset, filters images, converts them to JPG,
    and saves them to a new directory structure.

    Args:
        input_dir (str): Path to the raw dataset directory (e.g., 'datasets').
        output_dir (str): Path to the processed dataset directory (e.g., 'processed_dataset').
        min_size (tuple): Minimum width and height for images (default: 100x100).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting data cleaning and processing from {input_dir} to {output_dir}")

    for person_name in os.listdir(input_dir):
        person_input_path = os.path.join(input_dir, person_name)
        person_output_path = os.path.join(output_dir, person_name)

        if not os.path.isdir(person_input_path):
            print(f"Skipping {person_input_path} as it is not a directory.")
            continue

        os.makedirs(person_output_path, exist_ok=True)
        print(f"Processing images for: {person_name}")

        for filename in os.listdir(person_input_path):
            file_path = os.path.join(person_input_path, filename)
            
            # Skip if not a file or if it's a hidden file
            if not os.path.isfile(file_path) or filename.startswith('.'):
                continue

            try:
                # Open image using Pillow
                with Image.open(file_path) as img:
                    # Convert to RGB if not already (handles PNG with alpha, etc.)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Check image size
                    if img.width < min_size[0] or img.height < min_size[1]:
                        print(f"  Skipping {filename}: size {img.width}x{img.height} is smaller than {min_size[0]}x{min_size[1]}.")
                        continue

                    # Generate a consistent filename (e.g., original_name.jpg)
                    output_filename = os.path.splitext(filename)[0] + '.jpg'
                    output_file_path = os.path.join(person_output_path, output_filename)
                    
                    # Save as JPG
                    img.save(output_file_path, "JPEG")
                    # print(f"  Processed and saved: {output_file_path}")

            except Exception as e:
                print(f"  Error processing {filename}: {e}")
                continue
    print("Data cleaning and processing complete.")

if __name__ == "__main__":
    # Define paths relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasets_dir = os.path.join(project_root, 'datasets')
    processed_output_dir = os.path.join(project_root, 'processed_dataset')
    
    clean_and_process_dataset(datasets_dir, processed_output_dir)
