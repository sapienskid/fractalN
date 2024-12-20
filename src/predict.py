import tensorflow as tf
import numpy as np
from PIL import Image
import sys

def load_and_prep_image(image_path):
    # Read JPEG file
    raw_img = tf.io.read_file(image_path)
    # Decode JPEG
    img = tf.io.decode_jpeg(raw_img, channels=3)
    # Resize
    img = tf.image.resize(img, [224, 224], method='bilinear')
    # Convert to float32 and normalize
    img = tf.cast(img, tf.float32) / 255.0
    # Add batch dimension
    img = tf.expand_dims(img, 0)
    return img

def predict_mushroom(model_path, image_path):
    # Update model path to use .keras extension
    model = tf.keras.models.load_model(model_path.replace('.h5', '.keras'))
    
    # Prepare the image
    processed_image = load_and_prep_image(image_path)
    
    # Make prediction with adjusted threshold
    prediction = model.predict(processed_image)
    
    # Use standard threshold since data is balanced
    THRESHOLD = 0.5
    
    # Interpret results
    if prediction[0][0] > THRESHOLD:
        return "Poisonous", prediction[0][0]
    else:
        return "Edible", 1 - prediction[0][0]

def analyze_prediction_confidence(prediction):
    """Helper function to provide more detailed analysis"""
    raw_score = prediction[0][0]
    if raw_score > 0.8:
        confidence_level = "Very High"
    elif raw_score > 0.6:
        confidence_level = "High"
    elif raw_score > 0.4:
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
    
    result, confidence = predict_mushroom('mushroom_classifier.keras', sys.argv[1])
    confidence_level = analyze_prediction_confidence(confidence)
    print(f"Prediction: {result}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Confidence Level: {confidence_level}")
    
    if confidence < 0.7:
        print("\nWarning: Low confidence prediction. Please seek expert verification.")
