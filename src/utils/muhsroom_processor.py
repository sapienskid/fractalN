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
        
        # Create stronger augmentation pipeline
        self.augmentation = A.Compose([
            # Geometric transformations
            A.RandomRotate90(p=0.5),    # Randomly rotate image by 90 degrees
            A.VerticalFlip(p=0.5),      # Flip image vertically
            A.HorizontalFlip(p=0.5),    # Flip image horizontally
            A.Transpose(p=0.5),         # Transpose image axes
            
            # Noise injection (20% chance)
            A.OneOf([
                A.GaussNoise(),         # Add gaussian noise
                A.GaussNoise(),         # Alternative gaussian noise
            ], p=0.2),
            
            # Blur effects (20% chance)
            # A.OneOf([
            #     A.MotionBlur(p=0.2),    # Add motion blur
            #     A.MedianBlur(blur_limit=3, p=0.1),  # Add median blur
            #     A.Blur(blur_limit=3, p=0.1),        # Add gaussian blur
            # ], p=0.2),
            
            # Geometric distortions (20% chance)
            # A.OneOf([
            #     A.OpticalDistortion(p=0.3),    # Simulate lens distortion
            #     A.GridDistortion(p=0.1),       # Apply grid distortion
            #     A.PiecewiseAffine(p=0.3),      # Local elastic deformation
            # ], p=0.2),
            
            # Color and contrast adjustments (30% chance)
            A.OneOf([
                A.CLAHE(clip_limit=2),         # Contrast Limited Adaptive Histogram Equalization
                A.Sharpen(),                   # Increase image sharpness
                A.Emboss(),                    # Create emboss effect
                A.RandomBrightnessContrast(),  # Adjust brightness and contrast
            ], p=0.3),
            
            A.HueSaturationValue(p=0.3),      # Adjust hue, saturation, and value
            
            # Normalize pixel values to standard ImageNet stats
            A.Normalize(
                mean=[0.485, 0.456, 0.406],    # ImageNet means
                std=[0.229, 0.224, 0.225],     # ImageNet standard deviations
            )
        ])

    def process_image(self, img_path, output_path, augment=False):
        """Process a single image with advanced preprocessing"""
        try:
            # Read image with cv2 for better performance
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                print(f"Warning: Corrupted image found: {img_path}")
                return False
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Ensure image is uint8 before augmentation
            img = img.astype(np.uint8)
            
            if augment:
                # Apply augmentation without normalization
                transform = A.Compose([
                    aug for aug in self.augmentation if not isinstance(aug, A.Normalize)
                ])
                augmented = transform(image=img)
                img = augmented['image']
            
            # Convert to PIL and save
            img = Image.fromarray(img.astype(np.uint8))
            img.save(output_path, format='JPEG', quality=95, optimize=True)
            return True
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return False

    def create_balanced_dataset(self, target_count=5000):
        """Create a balanced dataset with augmentation"""
        print("Creating balanced dataset...")
        
        # Count existing images
        stats = {}
        for category in ['poisonous', 'edible']:
            category_path = self.data_dir / category
            if category_path.exists():
                stats[category] = len(list(category_path.glob('*.[Jj][Pp][Gg]')))
        
        # Calculate augmentations needed
        for category, count in stats.items():
            if count < target_count:
                source_files = list((self.data_dir / category).glob('*.[Jj][Pp][Gg]'))
                augmentations_needed = target_count - count
                
                print(f"\nAugmenting {category}: {count} → {target_count}")
                with tqdm(total=augmentations_needed) as pbar:
                    while augmentations_needed > 0:
                        for source_file in source_files:
                            if augmentations_needed <= 0:
                                break
                            
                            output_path = self.data_dir / category / f"aug_{augmentations_needed}_{source_file.name}"
                            if self.process_image(source_file, output_path, augment=True):
                                augmentations_needed -= 1
                                pbar.update(1)

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
    
    # Step 1: Create balanced dataset with augmentation
    processor.create_balanced_dataset(target_count=5000)
    
    # Step 2: Split and preprocess dataset
    processor.split_dataset()
    
    # Step 3: Verify final dataset
    processor.verify_dataset()

if __name__ == "__main__":
    main()