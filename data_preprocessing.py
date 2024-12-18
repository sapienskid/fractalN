import os
import cv2
import numpy as np
from pathlib import Path

def resize_image(image, target_size=(224, 224)):
    """Resize image maintaining aspect ratio and pad if necessary."""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_h/h, target_w/w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create blank target image
    final_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate padding
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    # Place resized image in center
    final_img[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    return final_img

def process_dataset(input_dir, output_dir, target_size=(224, 224)):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "edible").mkdir(exist_ok=True)
    (output_path / "poisonous").mkdir(exist_ok=True)
    
    # Simplified folder mapping
    folder_mapping = {
        "edible mushroom sporocarp": "edible",
        "edible sporocarp": "edible",
        "poisonous mushroom sporocarp": "poisonous",
        "poisonous sporocarp": "poisonous"
    }
    
    total_processed = {"edible": 0, "poisonous": 0}
    
    for folder_name, output_class in folder_mapping.items():
        folder_path = input_path / folder_name
        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} not found")
            continue
            
        # Process all image files
        for img_path in folder_path.glob('*.[jJ][pP][gG]*'):
            try:
                # Read image using OpenCV
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Failed to read {img_path}")
                    continue
                
                # Resize image using OpenCV
                processed_img = resize_image(img, target_size)
                
                # Simpler naming: class_counter.jpeg
                total_processed[output_class] += 1
                new_filename = f"{output_class}_{total_processed[output_class]:04d}.jpeg"
                
                # Save using OpenCV
                out_path = output_path / output_class / new_filename
                cv2.imwrite(str(out_path), processed_img)
                print(f"Processed: {new_filename}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Print results
    print("\nProcessing complete!")
    for class_name, count in total_processed.items():
        print(f"{class_name}: {count} images")

if __name__ == "__main__":
    # Update paths to include data folder
    input_directory = "data/raw_mushroom_dataset"
    output_directory = "data/processed_mushroom_dataset"
    target_size = (224, 224)
    
    # Validate input directory exists
    if not os.path.exists(input_directory):
        print(f"Error: Input directory '{input_directory}' not found!")
        print("Make sure the dataset is in the correct location:")
        print("  Expected path: /home/sapienskid/Development/FractalN/data/raw_mushroom_dataset")
        exit(1)
    
    print("Current working directory:", os.getcwd())
    print(f"Processing images from: {input_directory}")
    process_dataset(input_directory, output_directory, target_size)
