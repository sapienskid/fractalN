import os
import cv2
import numpy as np
from model import CNN
from colorama import Fore, Style
from PIL import Image

def preprocess_image(image_path):
    """Preprocess a single image for prediction"""
    # Load and resize image to match training size (224x224)
    img = Image.open(image_path)
    img = img.resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(img)
    img_array = np.transpose(img_array, (2, 0, 1))  # CHW format
    img_array = img_array / 255.0  # Normalize
    
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

def predict_single_image(image_path, model_path="models/saved_models/best_model"):
    """Make prediction on a single mushroom image"""
    try:
        # Load and initialize model
        model = CNN()
        model.load(model_path)
        
        # Preprocess image
        img = preprocess_image(image_path)
        
        # Make prediction
        predictions = model.forward(img)
        class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][class_index])
        
        # Format results
        result = {
            'class': "Edible" if class_index == 0 else "Poisonous",
            'confidence': confidence,
            'probabilities': {
                'Edible': float(predictions[0][0]),
                'Poisonous': float(predictions[0][1])
            }
        }
        
        # Print results
        print(f"\n{Fore.CYAN}{'='*50}")
        print(f"{Fore.WHITE}Mushroom Classification Results")
        print(f"{Fore.CYAN}{'='*50}")
        print(f"\n{Fore.WHITE}Image: {Fore.YELLOW}{os.path.basename(image_path)}")
        print(f"\n{Fore.WHITE}Classification: {Fore.GREEN if result['class'] == 'Edible' else Fore.RED}{result['class']}")
        print(f"{Fore.WHITE}Confidence: {Fore.YELLOW}{confidence:.2%}")
        print(f"\n{Fore.WHITE}Detailed Probabilities:")
        print(f"{Fore.GREEN}Edible: {result['probabilities']['Edible']:.2%}")
        print(f"{Fore.RED}Poisonous: {result['probabilities']['Poisonous']:.2%}")
        
        # Warning for low confidence predictions
        if confidence < 0.8:
            print(f"\n{Fore.YELLOW}Warning: Low confidence prediction!")
            print(f"Please consult an expert mycologist for verification.{Style.RESET_ALL}")
        
        return result
        
    except Exception as e:
        print(f"{Fore.RED}Error making prediction: {str(e)}{Style.RESET_ALL}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Predict if a mushroom is edible or poisonous')
    parser.add_argument('image_path', help='Path to the mushroom image')
    parser.add_argument('--model', default='models/saved_models/best_model', 
                      help='Path to the trained model')
    
    args = parser.parse_args()
    predict_single_image(args.image_path, args.model)
