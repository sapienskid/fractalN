#mushroom_data_processor.py
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import math
from tqdm import tqdm

class MushroomDataProcessor:
    def __init__(self, 
                 data_dir='data/mushroom_data',
                 processed_dir='data/processed',
                 img_size=(224, 224),
                 validation_split=0.1,
                 test_split=0.1):
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir)
        self.img_size = img_size
        self.validation_split = validation_split
        self.test_split = test_split
        
        # Create necessary directories
        for split in ['train', 'validation', 'test']:
            for category in ['poisonous', 'edible']:
                (self.processed_dir / split / category).mkdir(parents=True, exist_ok=True)

    def process_dataset(self):
        """Main processing pipeline"""
        # Get class distributions
        class_counts = self._get_class_distribution()
        print("\nOriginal class distribution:")
        for cls, count in class_counts.items():
            print(f"{cls}: {count} images")

        # Split data while maintaining stratification
        splits = self._split_dataset()
        
        # Process and augment training data
        self._process_training_data(splits['train'])
        
        # Process validation and test data (no augmentation)
        self._process_evaluation_data(splits['validation'], 'validation')
        self._process_evaluation_data(splits['test'], 'test')

    def _get_class_distribution(self):
        """Count images in each class"""
        return {
            category: len(list(self.data_dir.glob(f"{category}/*.[Jj][Pp][Gg]")))
            for category in ['poisonous', 'edible']
        }

    def _split_dataset(self):
        """Split dataset into train/validation/test sets"""
        datasets = {}
        for category in ['poisonous', 'edible']:
            images = list((self.data_dir / category).glob('*.[Jj][Pp][Gg]'))
            
            # First split out test set
            train_val, test = train_test_split(
                images,
                test_size=self.test_split,
                random_state=42
            )
            
            # Then split remaining data into train and validation
            train, validation = train_test_split(
                train_val,
                test_size=self.validation_split / (1 - self.test_split),
                random_state=42
            )
            
            datasets[category] = {
                'train': train,
                'validation': validation,
                'test': test
            }
        
        # Combine into final splits
        return {
            'train': {category: data['train'] 
                     for category, data in datasets.items()},
            'validation': {category: data['validation'] 
                          for category, data in datasets.items()},
            'test': {category: data['test'] 
                    for category, data in datasets.items()}
        }

    def _process_training_data(self, train_splits):
        """Process and augment training data"""
        # Calculate target counts for balanced classes
        max_count = max(len(files) for files in train_splits.values())
        
        for category, files in train_splits.items():
            print(f"\nProcessing {category} training images...")
            needed = max_count - len(files)
            
            # Process original files with progress bar
            for img_path in tqdm(files, desc=f"Processing original {category} images"):
                self._process_image(img_path, 
                                  self.processed_dir / 'train' / category,
                                  augment=False)
            
            # Add augmented versions if needed
            if needed > 0:
                print(f"Generating {needed} augmented images...")
                augmentations_per_image = math.ceil(needed / len(files))
                
                with tqdm(total=needed, desc=f"Generating augmented {category} images") as pbar:
                    for img_path in files:
                        for i in range(augmentations_per_image):
                            if needed <= 0:
                                break
                            self._process_image(img_path,
                                              self.processed_dir / 'train' / category,
                                              augment=True,
                                              suffix=f'_aug_{i}')
                            needed -= 1
                            pbar.update(1)

    def _process_evaluation_data(self, split_data, split_name):
        """Process validation/test data without augmentation"""
        for category, files in split_data.items():
            print(f"\nProcessing {category} {split_name} images...")
            for img_path in tqdm(files, desc=f"Processing {split_name} {category} images"):
                self._process_image(img_path,
                                  self.processed_dir / split_name / category,
                                  augment=False)

    def _process_image(self, img_path, output_dir, augment=False, suffix=''):
        """Modified to handle multiple augmentations"""
        try:
            with Image.open(img_path) as img:
                if (img.mode != 'RGB'):
                    img = img.convert('RGB')
                
                img_tensor = tf.convert_to_tensor(np.array(img))
                img_tensor = tf.image.resize(img_tensor, self.img_size)
                img_tensor = tf.cast(img_tensor, tf.float32) / 255.0
                
                if augment:
                    augmented_images = self._augment_image(img_tensor)
                    # Save each augmented version
                    for i, aug_img in enumerate(augmented_images):
                        aug_img = tf.cast(aug_img * 255, tf.uint8)
                        output_path = output_dir / f"{img_path.stem}_aug_{i}{img_path.suffix}"
                        Image.fromarray(aug_img.numpy()).save(output_path, 'JPEG', quality=95)
                else:
                    # Save original processed image
                    img_tensor = tf.cast(img_tensor * 255, tf.uint8)
                    output_path = output_dir / f"{img_path.stem}{suffix}{img_path.suffix}"
                    Image.fromarray(img_tensor.numpy()).save(output_path, 'JPEG', quality=95)
                    
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    def _augment_image(self, image):
        """Fix rotation angle issue"""
        augmented_images = []
        
        # 1. Horizontal flip + brightness
        aug1 = tf.image.random_flip_left_right(image)
        aug1 = tf.image.random_brightness(aug1, 0.3)
        augmented_images.append(aug1)
        
        # 2. Vertical flip + contrast
        aug2 = tf.image.random_flip_up_down(image)
        aug2 = tf.image.random_contrast(aug2, 0.7, 1.3)
        augmented_images.append(aug2)
        
        # 3. Fixed rotation + saturation
        aug3 = tf.image.rot90(image, k=1)  # 90-degree rotation
        aug3 = tf.image.random_saturation(aug3, 0.7, 1.3)
        augmented_images.append(aug3)
        
        # 4. Color jitter
        aug4 = tf.image.random_hue(image, 0.2)
        aug4 = tf.image.random_saturation(aug4, 0.8, 1.2)
        aug4 = tf.image.random_brightness(aug4, 0.2)
        augmented_images.append(aug4)
        
        # 5. Zoom only
        h = tf.cast(tf.shape(image)[0], tf.float32)
        w = tf.cast(tf.shape(image)[1], tf.float32)
        crop_h = tf.cast(tf.round(h * 0.8), tf.int32)
        crop_w = tf.cast(tf.round(w * 0.8), tf.int32)
        crop_h = tf.maximum(crop_h, 1)
        crop_w = tf.maximum(crop_w, 1)
        aug5 = tf.image.random_crop(
            image, 
            size=[crop_h, crop_w, 3]
        )
        aug5 = tf.image.resize(aug5, (tf.shape(image)[0], tf.shape(image)[1]))
        augmented_images.append(aug5)
        
        return [tf.clip_by_value(img, 0.0, 1.0) for img in augmented_images]


if __name__ == "__main__":
    processor = MushroomDataProcessor()
    processor.process_dataset()