import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_config import setup_gpu

setup_gpu()


def create_augmented_image(image, seed=None):
    """Create an augmented version of a single image with extensive variations"""
    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)
    
    # Ensure proper format
    image = tf.cast(image, tf.float32)
    if tf.reduce_max(image) > 1.0:
        image = image / 255.0

    # Enhanced augmentation pipeline
    augmentation = tf.keras.Sequential([
        # Geometric transformations
        tf.keras.layers.RandomRotation(0.5, fill_mode='reflect'),  # More rotation
        tf.keras.layers.RandomTranslation(0.3, 0.3, fill_mode='reflect'),  # More translation
        tf.keras.layers.RandomZoom(0.3, fill_mode='reflect'),  # More zoom
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        
        # Added new transformations
        tf.keras.layers.RandomCrop(height=image.shape[0], width=image.shape[1]),
        tf.keras.layers.RandomContrast(0.4),  # Increased contrast range
    ])
    
    # Apply base augmentations
    augmented = augmentation(tf.expand_dims(image, 0))[0]
    
    # Additional color augmentations
    augmented = tf.image.random_brightness(augmented, 0.4)  # Increased brightness range
    augmented = tf.image.random_saturation(augmented, 0.6, 1.6)  # Increased saturation range
    augmented = tf.image.random_hue(augmented, 0.2)  # Added hue variation
    
    # Convert to numpy for OpenCV operations
    augmented_np = augmented.numpy()
    augmented_np = (augmented_np * 255).astype(np.uint8)
    
    # Random OpenCV filters (apply one randomly)
    if np.random.random() < 0.3:  # 30% chance for each image
        filter_choice = np.random.choice(['blur', 'sharpen', 'edge'])
        
        if filter_choice == 'blur':
            # Gaussian blur with random kernel size
            kernel_size = np.random.choice([3, 5, 7])
            augmented_np = cv2.GaussianBlur(augmented_np, (kernel_size, kernel_size), 0)
        
        elif filter_choice == 'sharpen':
            # Sharpen filter
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            augmented_np = cv2.filter2D(augmented_np, -1, kernel)
        
        elif filter_choice == 'edge':
            # Edge enhancement
            kernel = np.array([[0,-1,0],
                             [-1,5,-1],
                             [0,-1,0]])
            augmented_np = cv2.filter2D(augmented_np, -1, kernel)
    
    # Random noise (20% chance)
    if np.random.random() < 0.2:
        noise = np.random.normal(0, 15, augmented_np.shape).astype(np.uint8)
        augmented_np = cv2.add(augmented_np, noise)
    
    # Ensure proper range
    augmented_np = np.clip(augmented_np, 0, 255)
    
    return augmented_np

def augment_mushroom_data(data_dir='data/mushroom_data', target_count=14000):
    """Augment mushroom images to reach exactly target_count images per class"""
    try:
        base_path = Path(data_dir)
        if not base_path.exists():
            raise FileNotFoundError(f"Directory {data_dir} not found!")
        
        # Count existing images in each category
        stats = {}
        for category in ['poisonous', 'edible']:
            category_path = base_path / category
            if not category_path.exists():
                print(f"Warning: Category directory {category} not found!")
                continue
            stats[category] = len(list(category_path.glob('*.[Jj][Pp][Gg]')))
        
        print("\nInitial class distribution:")
        for category, count in stats.items():
            print(f"{category.capitalize()}: {count} images")
        
        # Calculate needed augmentations for each category
        augmentations_needed = {
            category: target_count - count 
            for category, count in stats.items()
        }
        
        # Process each category
        for category, needed in augmentations_needed.items():
            if needed <= 0:
                print(f"\nWarning: {category} already has {stats[category]} images.")
                print(f"Removing {abs(needed)} images to reach target...")
                
                # Remove excess images if we have more than target
                category_path = base_path / category
                all_images = list(category_path.glob('*.[Jj][Pp][Gg]'))
                images_to_remove = all_images[target_count:]
                for img_path in images_to_remove:
                    img_path.unlink()
                continue
            
            category_path = base_path / category
            print(f"\nProcessing {category} images...")
            print(f"Need to generate {needed} new images to reach {target_count}")
            
            # Get list of original images
            original_images = list(category_path.glob('*.[Jj][Pp][Gg]'))
            
            # Calculate augmentations per image
            augmentations_per_image = needed // len(original_images)
            extra_augmentations = needed % len(original_images)
            
            print(f"Will create {augmentations_per_image} variations per image")
            print(f"Plus {extra_augmentations} extra variations")
            
            # Create augmented versions with fixed image saving
            with tqdm(total=needed, desc=f"Augmenting {category}") as pbar:
                for idx, img_path in enumerate(original_images):
                    try:
                        # Load image using PIL first
                        with Image.open(img_path) as img:
                            # Convert to RGB to ensure consistent color space
                            img = img.convert('RGB')
                            # Convert to numpy array
                            img_array = np.array(img)
                        
                        # Calculate variations for this image
                        num_variations = augmentations_per_image
                        if idx < extra_augmentations:
                            num_variations += 1
                        
                        # Generate variations
                        for i in range(num_variations):
                            # Generate augmented image
                            augmented = create_augmented_image(
                                img_array, 
                                seed=idx * augmentations_per_image + i
                            )
                            
                            # Save using PIL to preserve colors
                            aug_filename = f"aug_{idx}_{i}_{Path(img_path).stem}.jpg"
                            aug_path = category_path / aug_filename
                            
                            # Convert tensor to PIL Image and save
                            Image.fromarray(augmented).save(
                                aug_path,
                                'JPEG',
                                quality=95
                            )
                            pbar.update(1)
                            
                    except Exception as e:
                        print(f"\nError processing {img_path.name}: {str(e)}")
                        continue
            
            # Verify final count
            final_count = len(list(category_path.glob('*.[Jj][Pp][Gg]')))
            print(f"\n{category.capitalize()} final count: {final_count} images")
            if final_count != target_count:
                print(f"Warning: Expected {target_count} images but got {final_count}")
    
    except Exception as e:
        print(f"Error during augmentation: {str(e)}")
        raise

if __name__ == "__main__":
    print("Starting balanced mushroom data augmentation...")
    augment_mushroom_data(target_count=5000)
    print("\nAugmentation complete!")
