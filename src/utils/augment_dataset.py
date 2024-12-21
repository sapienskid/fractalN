import tensorflow as tf
import numpy as np
import os
import cv2

def create_augmented_variations(image_array, num_variations=15):
    """Create multiple augmented versions of a single image"""
    augmented_images = []
    
    # Remove batch dimension and ensure correct format
    image = tf.squeeze(image_array)  # Remove batch dimension
    
    # Create augmentation pipeline
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.5, fill_mode='reflect'),
        tf.keras.layers.RandomZoom(0.2, fill_mode='reflect'),
        tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode='reflect')
    ])
    
    for _ in range(num_variations):
        # Add batch dimension for augmentation
        img_batch = tf.expand_dims(image, 0)
        
        # Basic augmentations
        augmented = augmentation(img_batch)
        augmented = tf.squeeze(augmented)  # Remove batch dimension
        
        # Color augmentations
        augmented = tf.image.random_brightness(augmented, 0.2)
        augmented = tf.image.random_contrast(augmented, 0.8, 1.2)
        augmented = tf.image.random_saturation(augmented, 0.8, 1.2)
        
        # Convert to numpy and ensure correct format for OpenCV
        img_np = augmented.numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Apply OpenCV operations
        if np.random.rand() > 0.5:
            # Sharpen
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img_np = cv2.filter2D(img_np, -1, kernel)
        else:
            # Blur
            img_np = cv2.GaussianBlur(img_np, (3,3), 0.5)
        
        # Convert back to float32 and normalize
        img_np = img_np.astype(np.float32) / 255.0
        augmented_images.append(img_np)
    
    return augmented_images

def augment_dataset(base_dir):
    """Augment images in place within train and test directories"""
    
    # Process each directory
    for split in ['train', 'test']:
        split_dir = os.path.join(base_dir, split)
        for category in ['poisonous', 'edible']:
            category_dir = os.path.join(split_dir, category)
            print(f"\nProcessing {split}/{category}")
            
            # Get original images
            original_images = [
                f for f in os.listdir(category_dir) 
                if f.lower().endswith(('.jpg', '.jpeg'))
                and not f.startswith('aug_')
            ]
            
            # Process each image
            total = len(original_images)
            for idx, img_name in enumerate(original_images, 1):
                try:
                    img_path = os.path.join(category_dir, img_name)
                    
                    # Load and preprocess image
                    img = tf.keras.utils.load_img(img_path)
                    img_array = tf.keras.utils.img_to_array(img)
                    img_array = img_array / 255.0
                    
                    # Generate augmented versions
                    augmented_images = create_augmented_variations(img_array)
                    
                    # Save augmented images
                    base_name = os.path.splitext(img_name)[0]
                    for i, aug_img in enumerate(augmented_images):
                        aug_name = f"aug_{i+1}_{base_name}.jpg"
                        aug_path = os.path.join(category_dir, aug_name)
                        
                        # Convert to uint8 for saving
                        save_img = (aug_img * 255).astype(np.uint8)
                        cv2.imwrite(aug_path, cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))
                    
                    print(f"\rProgress: {idx}/{total} images processed", end="")
                except Exception as e:
                    print(f"\nError processing {img_name}: {str(e)}")
                    continue
                    
            print()  # New line after progress

if __name__ == "__main__":
    # Print dataset statistics
    base_dir = 'data'
    print("Initial dataset statistics:")
    for split in ['train', 'test']:
        for category in ['poisonous', 'edible']:
            path = os.path.join(base_dir, split, category)
            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg'))]
            print(f"{split}/{category}: {len(files)} images")
    
    # Perform augmentation
    augment_dataset(base_dir)
    
    # Print final dataset statistics
    print("\nFinal dataset statistics:")
    for split in ['train', 'test']:
        for category in ['poisonous', 'edible']:
            path = os.path.join(base_dir, split, category)
            files = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg'))]
            print(f"{split}/{category}: {len(files)} images")
