import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os
from gpu_config import setup_gpu

# Configure GPU at startup
setup_gpu()

# Add constants for image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224

def load_and_prep_image(image_path):
    # Read JPEG file
    raw_img = tf.io.read_file(image_path)
    # Decode JPEG
    img = tf.io.decode_jpeg(raw_img, channels=3)
    # Resize to match model's expected input
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method='bilinear')
    # Convert to float32 and normalize
    img = tf.cast(img, tf.float32) / 255.0
    # Add batch dimension
    img = tf.expand_dims(img, 0)
    return img

def predict_mushroom(model_path, image_path):
    try:
        # Ensure we're using the GPU
        with tf.device('/GPU:0'):
            model = tf.keras.models.load_model(model_path, compile=False)
            
            model.compile(
                optimizer='adam', 
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            processed_image = load_and_prep_image(image_path)
            prediction = model.predict(processed_image, verbose=0)
            
            # Get probabilities for each class
            edible_prob = float(prediction[0][0])
            poisonous_prob = float(prediction[0][1])
            
            # Return prediction based on highest probability
            if poisonous_prob > edible_prob:
                return "Poisonous", poisonous_prob
            else:
                return "Edible", edible_prob
            
    except Exception as e:
        print(f"Error making prediction: {e}")
        raise

def analyze_prediction_confidence(confidence):
    """Helper function to provide more detailed analysis"""
    if confidence > 0.8:
        confidence_level = "Very High"
    elif confidence > 0.6:
        confidence_level = "High"
    elif confidence > 0.4:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"
    return confidence_level

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        print("Note: Image should be in JPEG format")
        sys.exit(1)
    
    if not sys.argv[1].lower().endswith(('.jpg', '.jpeg')):
        print("Warning: Image file should be in JPEG format")
    
    # Change model path to match what train.py saves
    result, confidence = predict_mushroom('best_model.keras', sys.argv[1])
    
    confidence_level = analyze_prediction_confidence(confidence)
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Confidence Level: {confidence_level}")
    
    if confidence < 0.7:
        print("\nWarning: Low confidence prediction. Please seek expert verification.")
