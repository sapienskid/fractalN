import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import albumentations as A
import shutil

class MushroomDataProcessor:
    def __init__(self, data_dir='data/mushroom_data', output_dir='data/processed',
                 img_size=(224, 224), validation_split=0.15, test_split=0.15):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.validation_split = validation_split
        self.test_split = test_split

    def process_image(self, img_path, output_path):
        """Process a single image with basic preprocessing"""
        try:
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                print(f"Warning: Corrupted image found: {img_path}")
                return False
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LANCZOS4)
            img = Image.fromarray(img)
            img.save(output_path, format='JPEG', quality=95, optimize=True)
            return True
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return False

    def split_dataset(self):
        """Split dataset into train/validation/test sets"""
        print("\nSplitting dataset...")
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            for category in ['poisonous', 'edible']:
                os.makedirs(self.output_dir / split / category, exist_ok=True)
        
        # Process each category
        for category in ['poisonous', 'edible']:
            print(f"\nProcessing {category}...")
            image_files = list((self.data_dir / category).glob('*.[Jj][Pp][Gg]'))
            
            # First split: separate test set
            train_val_files, test_files = train_test_split(
                image_files,
                test_size=self.test_split,
                random_state=42
            )
            
            # Second split: separate validation set
            train_files, val_files = train_test_split(
                train_val_files,
                test_size=self.validation_split,
                random_state=42
            )
            
            # Process and copy files
            for files, split in [(train_files, 'train'), 
                               (val_files, 'val'), 
                               (test_files, 'test')]:
                print(f"{split}: {len(files)} images")
                for img_path in tqdm(files, desc=f"Processing {split}"):
                    output_path = self.output_dir / split / category / img_path.name
                    self.process_image(img_path, output_path)

    def verify_dataset(self):
        """Verify dataset integrity and balance"""
        print("\nVerifying dataset...")
        for split in ['train', 'val', 'test']:
            print(f"\n{split.capitalize()} set:")
            for category in ['poisonous', 'edible']:
                count = len(list((self.output_dir / split / category).glob('*.[Jj][Pp][Gg]')))
                print(f"{category.capitalize()}: {count} images")

def main():
    processor = MushroomDataProcessor()
    
    # Step 1: Split and preprocess dataset
    processor.split_dataset()
    
    # Step 2: Verify final dataset
    processor.verify_dataset()

if __name__ == "__main__":
    main()