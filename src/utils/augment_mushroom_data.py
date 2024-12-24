#data_augmentation.py
import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from PIL import Image
import sys
import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_config import setup_gpu

setup_gpu()

# Add custom layer definitions
class RandomSaturation(tf.keras.layers.Layer):
    def __init__(self, lower=0.7, upper=1.3, **kwargs):
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper
    
    def call(self, inputs, training=None):
        if training:
            return tf.image.random_saturation(inputs, self.lower, self.upper)
        return inputs

class RandomHue(tf.keras.layers.Layer):
    def __init__(self, factor=0.1, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor
    
    def call(self, inputs, training=None):
        if training:
            return tf.image.random_hue(inputs, self.factor)
        return inputs

class RandomBlur(tf.keras.layers.Layer):
    def __init__(self, prob=0.3, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob
    
    def call(self, inputs, training=None):
        if training and tf.random.uniform([]) < self.prob:
            return tf.nn.avg_pool2d(inputs, ksize=3, strides=1, padding='SAME')
        return inputs

def create_augmented_image(image, seed=None):
    """Create truly augmented version of the image without tensorflow-addons"""
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    # Convert to float32
    image = tf.cast(image, tf.float32) / 255.0
    
    # Basic transformations using tf.image
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Color transformations
    image = tf.image.random_brightness(image, 0.4)
    image = tf.image.random_contrast(image, 0.5, 1.5)
    image = tf.image.random_saturation(image, 0.5, 1.5)
    image = tf.image.random_hue(image, 0.2)
    
    # Add random noise
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.05)
    image = image + noise
    
    # Random rotation (using tf.image instead of tfa)
    # Convert angle from degrees to radians
    angle = np.random.uniform(-45, 45) * np.pi / 180
    image = tf.image.rot90(image, k=int(angle/(np.pi/2)))
    
    # Random crop and resize (similar to zoom)
    crop_size = tf.random.uniform([], 0.8, 1.0, dtype=tf.float32)
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    crop_h = tf.cast(tf.cast(h, tf.float32) * crop_size, tf.int32)
    crop_w = tf.cast(tf.cast(w, tf.float32) * crop_size, tf.int32)
    image = tf.image.random_crop(image, [crop_h, crop_w, 3])
    image = tf.image.resize(image, [h, w])
    
    # Random blur using conv2d
    if tf.random.uniform([]) < 0.3:
        gaussian_kernel = tf.cast([[1, 2, 1], [2, 4, 2], [1, 2, 1]], tf.float32) / 16
        gaussian_kernel = gaussian_kernel[:, :, tf.newaxis, tf.newaxis]
        gaussian_kernel = tf.tile(gaussian_kernel, [1, 1, 3, 1])
        image = tf.nn.depthwise_conv2d(
            tf.expand_dims(image, 0),
            gaussian_kernel,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )[0]
    
    # Ensure values are in valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return tf.cast(image * 255, tf.uint8).numpy()

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
                            # Generate augmented image with multiple attempts
                            augmented = create_augmented_image(
                                img_array, 
                                seed=idx * augmentations_per_image + i
                            )
                            
                            # Save using PIL
                            aug_filename = f"aug_{idx}_{i}_{Path(img_path).stem}.jpg"
                            aug_path = category_path / aug_filename
                            
                            # Convert numpy array to PIL Image and save
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
