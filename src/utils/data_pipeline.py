import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import shutil
from typing import Dict, Tuple

class DataPipeline:
    def __init__(self, 
                 raw_data_dir: str = 'data/mushroom_data',
                 processed_dir: str = 'data/processed',
                 img_size: Tuple[int, int] = (160, 160),  # Updated default size
                 validation_split: float = 0.1,
                 test_split: float = 0.2,
                 batch_size: int = 32):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.img_size = img_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.batch_size = batch_size

    def setup_data(self, preprocess: bool = False) -> Dict[str, tf.data.Dataset]:
        """
        Main method to setup data pipeline
        Returns: Dictionary containing train, validation, and test datasets
        """
        if not preprocess and self._check_processed_data():
            print("Using existing processed data...")
            return self._create_datasets()

        print("Running complete preprocessing pipeline...")
        self._preprocess_data()
        return self._create_datasets()

    def _check_processed_data(self) -> bool:
        """Check if processed data exists and is valid"""
        if not self.processed_dir.exists():
            return False

        # Check all required directories exist and contain data
        required_dirs = ['train', 'validation', 'test']
        required_classes = ['poisonous', 'edible']
        
        for split in required_dirs:
            for cls in required_classes:
                dir_path = self.processed_dir / split / cls
                if not dir_path.exists():
                    return False
                if not list(dir_path.glob('*.[Jj][Pp][Gg]')):
                    return False

        return True

    def _preprocess_data(self):
        """Run the complete preprocessing pipeline"""
        print("Cleaning up existing processed directories...")
        self._clean_processed_dir()
        
        # Clean up and recreate mushroom_data directory
        mushroom_data_dir = Path('data/mushroom_data')
        if mushroom_data_dir.exists():
            print("Removing existing mushroom_data directory...")
            shutil.rmtree(mushroom_data_dir)
        
        # Reorganize raw data
        print("Reorganizing raw data...")
        from utils.reorganize_data import reorganize_mushroom_data
        reorganize_mushroom_data()
        
        # Create directory structure
        self._organize_raw_data()
        
        # Process and split data
        self._process_and_split_data()

    def _clean_processed_dir(self):
        """Clean existing processed directory"""
        if self.processed_dir.exists():
            shutil.rmtree(self.processed_dir)
        self.processed_dir.mkdir(parents=True)

    def _organize_raw_data(self):
        """Create organized directory structure"""
        for split in ['train', 'validation', 'test']:
            for category in ['poisonous', 'edible']:
                (self.processed_dir / split / category).mkdir(parents=True)

    def _process_and_split_data(self):
        """Process and split the data into train/validation/test sets"""
        # Initialize data processor
        from utils.mushroom_data_processor import MushroomDataProcessor
        processor = MushroomDataProcessor(
            data_dir=self.raw_data_dir,
            processed_dir=self.processed_dir,
            img_size=self.img_size,
            validation_split=self.validation_split,
            test_split=self.test_split
        )
        processor.process_dataset()

    def _create_datasets(self) -> Dict[str, tf.data.Dataset]:
        """Create TensorFlow datasets from processed data"""
        datasets = {}

        def preprocess_image(path, label):
            # Read image
            img = tf.io.read_file(path)
            # Decode and resize
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, self.img_size)
            # Normalize
            img = tf.cast(img, tf.float32) / 255.0
            return img, tf.cast(label, tf.float32)

        def create_dataset(directory: Path, is_training: bool = False):
            # Get all image paths and labels
            image_paths = []
            labels = []
            for idx, class_name in enumerate(['poisonous', 'edible']):
                class_paths = list((directory / class_name).glob('*.[Jj][Pp][Gg]'))
                image_paths.extend([str(p) for p in class_paths])
                labels.extend([idx] * len(class_paths))

            # Convert to tensors
            ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
            
            # Map preprocessing
            ds = ds.map(
                preprocess_image,
                num_parallel_calls=2  # Reduced from AUTOTUNE
            )

            # Configure dataset
            ds = ds.cache()
            
            # For training, shuffle before batching
            if is_training:
                ds = ds.shuffle(min(500, len(image_paths)))
                
            # Batch the dataset
            ds = ds.batch(self.batch_size)
            
            # For training, repeat the dataset AFTER batching
            if is_training:
                ds = ds.repeat()
                
            ds = ds.prefetch(1)
            return ds, len(image_paths)  # Return length as well

        # Create datasets with their lengths
        train_ds, train_size = create_dataset(self.processed_dir / 'train', is_training=True)
        val_ds, val_size = create_dataset(self.processed_dir / 'validation')
        test_ds, test_size = create_dataset(self.processed_dir / 'test')

        # Store sizes in the datasets dict
        datasets = {
            'train': train_ds,
            'validation': val_ds,
            'test': test_ds,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size
        }

        # Calculate class weights
        class_weights = self._calculate_class_weights()
        datasets['class_weights'] = class_weights
        
        return datasets

    def _calculate_class_weights(self) -> dict:
        """Calculate class weights based on training data distribution."""
        train_dir = self.processed_dir / 'train'
        poisonous_samples = len(list((train_dir / 'poisonous').glob('*.[Jj][Pp][Gg]')))
        edible_samples = len(list((train_dir / 'edible').glob('*.[Jj][Pp][Gg]')))
        
        total_samples = poisonous_samples + edible_samples
        print(f"\nClass distribution in training set:")
        print(f"Poisonous samples: {poisonous_samples}")
        print(f"Edible samples: {edible_samples}")
        
        # Handle cases where there are zero samples in a class
        if poisonous_samples == 0 or edible_samples == 0:
            raise ValueError("One of the classes has zero samples. Adjust your dataset.")

        # Calculate weights
        max_samples = max(poisonous_samples, edible_samples)
        weights = {
            0: max_samples / poisonous_samples,
            1: max_samples / edible_samples
        }
        
        # Optional: Normalize weights to sum to 2.0
        weight_sum = sum(weights.values())
        weights = {k: (v * 2.0 / weight_sum) for k, v in weights.items()}
        
        return weights


def setup_training_data(preprocess: bool = False) -> Dict[str, tf.data.Dataset]:
    """Main function to setup training data"""
    pipeline = DataPipeline()
    return pipeline.setup_data(preprocess)

# Usage in train.py
if __name__ == "__main__":
    # Example usage in your training script
    datasets = setup_training_data(preprocess=False)
    
    # Your existing training code would start here
    train_ds = datasets['train']
    val_ds = datasets['validation']
    test_ds = datasets['test']
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")