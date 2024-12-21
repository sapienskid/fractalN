import tensorflow as tf
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
import cv2

def augment_image(img_array):
    """Create multiple augmented versions of a single image"""
    augmented_images = []
    
    # Remove batch dimension if present
    if len(img_array.shape) == 4:
        img_array = img_array[0]
    
    # Ensure proper shape and normalization
    if img_array.dtype != np.float32:
        img_array = img_array.astype(np.float32)
    if img_array.max() > 1.0:
        img_array = img_array / 255.0

    # Apply augmentations
    for _ in range(15):  # Generate 15 variations
        img = img_array.copy()
        
        # Random rotation
        angle = np.random.uniform(-180, 180)
        img = tf.image.rot90(img, k=int(angle/90))
        
        # Random flip
        if np.random.random() > 0.5:
            img = tf.image.flip_left_right(img)
        if np.random.random() > 0.5:
            img = tf.image.flip_up_down(img)
            
        # Color augmentations
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        
        augmented_images.append(img.numpy())
    
    return augmented_images

def prepare_dataset(base_dir='data', output_dir='data/processed'):
    """Prepare and augment the dataset"""
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for category in ['poisonous', 'edible']:
        print(f"\nProcessing {category} mushrooms...")
        
        # Create category directories
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)
        
        # Get all images from both train and test
        all_images = []
        for split in ['train', 'test']:
            src_dir = os.path.join(base_dir, split, category)
            if os.path.exists(src_dir):
                all_images.extend([
                    os.path.join(src_dir, f) 
                    for f in os.listdir(src_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg'))
                ])
        
        # Split into train/test
        train_images, test_images = train_test_split(
            all_images, test_size=0.2, random_state=42
        )
        
        # Process train images
        print(f"Processing {len(train_images)} training images...")
        for idx, img_path in enumerate(train_images):
            try:
                # Load and preprocess image
                img = tf.keras.preprocessing.image.load_img(img_path)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                
                # Generate augmented versions
                augmented_images = augment_image(img_array)
                
                # Save original and augmented images
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Save original
                dst_path = os.path.join(train_dir, category, f"{base_name}_orig.jpg")
                tf.keras.preprocessing.image.save_img(dst_path, img_array)
                
                # Save augmented versions
                for i, aug_img in enumerate(augmented_images):
                    aug_path = os.path.join(train_dir, category, f"{base_name}_aug_{i+1}.jpg")
                    tf.keras.preprocessing.image.save_img(aug_path, aug_img)
                
                print(f"\rProcessed {idx+1}/{len(train_images)} training images", end="")
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
                continue
        
        # Process test images (no augmentation)
        print(f"\nCopying {len(test_images)} test images...")
        for img_path in test_images:
            try:
                dst_path = os.path.join(test_dir, category, os.path.basename(img_path))
                shutil.copy2(img_path, dst_path)
            except Exception as e:
                print(f"Error copying {img_path}: {str(e)}")
        
        # Print statistics
        final_train_count = len(os.listdir(os.path.join(train_dir, category)))
        final_test_count = len(os.listdir(os.path.join(test_dir, category)))
        print(f"\n{category} final counts:")
        print(f"Training images: {final_train_count}")
        print(f"Test images: {final_test_count}")

if __name__ == "__main__":
    try:
        # Verify source data exists
        if not os.path.exists('data'):
            raise ValueError("Data directory not found!")
            
        # Prepare dataset
        prepare_dataset()
        print("\nDataset preparation completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
