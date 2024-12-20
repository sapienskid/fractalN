import os
import json
import numpy as np
from datetime import datetime
from model import CNN
from PIL import Image
import cv2
from pathlib import Path

def create_test_dirs():
    """Create necessary directories for testing"""
    dirs = [
        'data/test/edible',
        'data/test/poisonous',
        'models/saved_models',
        'results/inferences'
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_and_preprocess_image(image_path):
    """Load and preprocess a single test image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Resize to match training size
    img = cv2.resize(img, (224, 224))
    
    # Convert to float and normalize
    img = img.astype(np.float32) / 255.0
    
    # Reshape for CNN (batch_size=1, channels=3, height, width)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    
    return img

def run_inference(model_path, test_dir, results_dir):
    """Run inference on test images and save results"""
    model = CNN()
    model.load(model_path)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'predictions': []
    }
    
    # Process each class directory
    for class_name in ['edible', 'poisonous']:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        # Process each image in the class directory
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            try:
                # Load and preprocess image
                img = load_and_preprocess_image(img_path)
                
                # Get prediction
                predictions = model.forward(img)
                pred_class = "edible" if np.argmax(predictions[0]) == 0 else "poisonous"
                confidence = float(predictions[0][np.argmax(predictions[0])])
                
                # Store result
                result = {
                    'image_name': img_name,
                    'true_class': class_name,
                    'predicted_class': pred_class,
                    'confidence': confidence,
                    'correct': pred_class == class_name
                }
                results['predictions'].append(result)
                
                print(f"Processed {img_name}: {pred_class} ({confidence:.2%})")
                
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
    
    # Calculate accuracy
    correct = sum(1 for p in results['predictions'] if p['correct'])
    total = len(results['predictions'])
    results['accuracy'] = correct / total if total > 0 else 0
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = os.path.join(results_dir, f'inference_results_{timestamp}.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nTest Results:")
    print(f"Total images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Results saved to: {result_path}")

if __name__ == "__main__":
    create_test_dirs()
    
    model_path = "models/saved_models/best_model"
    test_dir = "data/test"
    results_dir = "results/inferences"
    
    run_inference(model_path, test_dir, results_dir)
