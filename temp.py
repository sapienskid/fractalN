import cv2
import numpy as np
from PIL import Image, ImageDraw
from rembg import remove

def create_crosshatch(image, angle, spacing):
    height, width = image.shape[:2]
    lines = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(-width, height + width, spacing):
        pt1 = (i, 0)
        pt2 = (i + width, width)
        cv2.line(lines, pt1, pt2, 255, 1)
    
    M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    return cv2.warpAffine(lines, M, (width, height))

def create_halftone(image, dot_size=3):
    height, width = image.shape[:2]
    halftone = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(0, height, dot_size):
        for x in range(0, width, dot_size):
            region = image[y:min(y+dot_size, height), x:min(x+dot_size, width)]
            if region.size > 0:
                avg_intensity = np.mean(region)
                radius = int((255 - avg_intensity) * dot_size / 800)  # Reduced intensity
                if radius > 0:
                    cv2.circle(halftone, (x + dot_size//2, y + dot_size//2), 
                             radius, 255, -1)
    return halftone

def style_image(input_path, output_path, accent_color=(70, 50, 255)):
    # Remove background
    input_img = Image.open(input_path)
    output = remove(input_img)
    
    # Convert to OpenCV format
    img_array = np.array(output)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    alpha_mask = img_array[:, :, 3]
    
    # Create base layers
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast while preserving details
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Create lighter engraving effect layers
    edges = cv2.Canny(enhanced, 30, 100)
    halftone = create_halftone(enhanced, dot_size=3)  # Smaller dots
    crosshatch1 = create_crosshatch(enhanced, 45, 12)  # Increased spacing
    crosshatch2 = create_crosshatch(enhanced, -45, 12)
    
    # Combine engraving effects with lower opacity
    engraving = cv2.addWeighted(crosshatch1, 0.3, crosshatch2, 0.3, 0)
    engraving = cv2.addWeighted(engraving, 0.4, halftone, 0.2, 0)
    engraving = cv2.addWeighted(engraving, 0.5, edges, 0.2, 0)
    
    # Start with the original image as base
    result = img.copy()
    
    # Create background with gradient
    background = np.full((height, width, 3), accent_color, dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(width, height) // 2
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    gradient = np.clip(1 - dist_from_center/radius, 0.2, 0.8)
    
    # Apply gradient to background
    for i in range(3):
        background[:,:,i] = background[:,:,i] * gradient
    
    # New blending approach
    engraving_color = cv2.cvtColor(engraving, cv2.COLOR_GRAY2BGR)
    
    # Layer composition (bottom to top):
    # 1. Original image (base)
    # 2. Light engraving overlay
    # 3. Background color with gradient
    subject = cv2.addWeighted(result, 0.7, engraving_color, 0.3, 0)  # More weight to original
    mask = np.stack([alpha_mask/255.0]*3, axis=-1)
    result = background * (1 - mask) + subject * mask
    
    # Save result
    cv2.imwrite(output_path, result.astype(np.uint8))

if __name__ == "__main__":
    # Example usage
    input_path = "sabin_pokharel.jpeg"
    output_path = "styled_output.jpg"
    style_image(input_path, output_path, accent_color=(70, 50, 255))