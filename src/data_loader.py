import os
import numpy as np
from PIL import Image

class DataLoader:
    def __init__(self, data_path, batch_size=8):
        self.data_path = data_path
        self.batch_size = batch_size
        self.image_paths = []
        self.labels = []
        self._load_paths()
        
    def _load_paths(self):
        """Load image paths and labels instead of actual images"""
        # Load edible mushrooms
        edible_path = os.path.join(self.data_path, 'edible')
        for img_name in os.listdir(edible_path):
            self.image_paths.append(os.path.join(edible_path, img_name))
            self.labels.append([1, 0])
            
        # Load poisonous mushrooms
        poisonous_path = os.path.join(self.data_path, 'poisonous')
        for img_name in os.listdir(poisonous_path):
            self.image_paths.append(os.path.join(poisonous_path, img_name))
            self.labels.append([0, 1])
        
        # Convert labels to numpy array
        self.labels = np.array(self.labels)
        self.num_samples = len(self.image_paths)
        
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img)
        img_array = np.transpose(img_array, (2, 0, 1))
        return img_array / 255.0
    
    def get_batches(self, shuffle=True):
        """Generator that yields batches of images"""
        indices = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            
            # Load images for this batch
            batch_images = []
            batch_labels = []
            
            for idx in batch_indices:
                img = self.load_and_preprocess_image(self.image_paths[idx])
                batch_images.append(img)
                batch_labels.append(self.labels[idx])
            
            yield np.array(batch_images), np.array(batch_labels)
    
    def split_indices(self, train_ratio=0.7, val_ratio=0.2):
        """Split data into train/val/test sets"""
        indices = np.random.permutation(self.num_samples)
        train_size = int(self.num_samples * train_ratio)
        val_size = int(self.num_samples * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        return train_indices, val_indices, test_indices
