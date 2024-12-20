from model import CNN
import numpy as np
from PIL import Image

def predict_image(image_path, model_path="models/saved_models/best_model"):
    # Load the trained model
    model = CNN()
    model.load(model_path)
    
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match training size
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(img)
    img_array = np.transpose(img_array, (2, 0, 1))  # CHW format
    img_array = img_array / 255.0  # Normalize
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get prediction
    predictions = model.forward(img_array)
    class_index = np.argmax(predictions[0])
    confidence = float(predictions[0][class_index])
    
    # Return result
    return {
        "class": "Edible" if class_index == 0 else "Poisonous",
        "confidence": confidence
    }

# Example usage
if __name__ == "__main__":
    result = predict_image("path/to/your/mushroom/image.jpg")
    print(f"Prediction: {result['class']} with {result['confidence']:.2%} confidence")
