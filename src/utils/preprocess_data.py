#preprocess_data.py
import os
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

def preprocess_dataset(data_dir='data/mushroom_data', output_dir='data/processed', test_size=0.2, img_size=(224, 224)):
    """
    Preprocess the dataset by:
    1. Splitting into train/test sets
    2. Resizing all images
    3. Organizing into proper directory structure
    """
    base_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'test']:
        for category in ['poisonous', 'edible']:
            os.makedirs(output_path / split / category, exist_ok=True)
    
    # Process each category
    for category in ['poisonous', 'edible']:
        print(f"\nProcessing {category} images...")
        image_files = list((base_path / category).glob('*.[Jj][Pp][Gg]'))
        
        # Split into train/test
        train_files, test_files = train_test_split(
            image_files, 
            test_size=test_size,
            random_state=42,
            shuffle=True
        )
        
        # Process training images
        for img_path in train_files:
            process_image(img_path, output_path / 'train' / category, img_size)
            
        # Process test images
        for img_path in test_files:
            process_image(img_path, output_path / 'test' / category, img_size)
        
        print(f"{category}: {len(train_files)} train, {len(test_files)} test")

def process_image(img_path, output_dir, img_size):
    """Process a single image: resize and save"""
    try:
        with Image.open(img_path) as img:
            # Convert to RGB if needed
            if (img.mode != 'RGB'):
                img = img.convert('RGB')
            
            # Resize maintaining aspect ratio
            img.thumbnail(img_size, Image.Resampling.LANCZOS)
            
            # Create new image with padding if needed
            new_img = Image.new('RGB', img_size, (0, 0, 0))
            paste_x = (img_size[0] - img.size[0]) // 2
            paste_y = (img_size[1] - img.size[1]) // 2
            new_img.paste(img, (paste_x, paste_y))
            
            # Save processed image
            new_img.save(output_dir / img_path.name, 'JPEG', quality=95)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    preprocess_dataset()
