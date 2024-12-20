import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = hasattr(cp, 'cuda') and cp.cuda.is_available()
except ImportError:
    cp = None
    GPU_AVAILABLE = False

from data_loader import DataLoader  # Updated import
import cv2
from model import CNN
from datetime import datetime
import os
from PIL import Image
import psutil
import GPUtil
from gpu_utils import configure_gpu, clear_gpu_memory, to_gpu, to_cpu, GPU_AVAILABLE
from progress_monitor import EnhancedProgressMonitor, EnhancedBatchProgress
from colorama import Fore, Style, init
from logger import TrainingLogger
init(autoreset=True)

def print_header(text):
    print("\n" + "="*50)
    print(f" {text} ".center(50))
    print("="*50 + "\n")

def print_progress_bar(iteration, total, prefix='', suffix='', length=50):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

def get_memory_usage():
    process = psutil.Process(os.getpid())
    if GPU_AVAILABLE:
        gpu = GPUtil.getGPUs()[0]
        return {
            'RAM': f"{process.memory_info().rss / 1024 / 1024:.1f}MB",
            'GPU': f"{gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)"
        }
    return {
        'RAM': f"{process.memory_info().rss / 1024 / 1024:.1f}MB",
        'GPU': "Not Available"
    }

def calculate_accuracy(predictions, labels):
    """Calculate accuracy ensuring consistent array types"""
    predictions = to_cpu(predictions)
    labels = to_cpu(labels)
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))

def train():
    # Initialize logger and monitor
    logger = TrainingLogger()
    
    # Configure GPU first and get status
    is_gpu_available = configure_gpu()
    
    # Training parameters
    batch_size = 8
    epochs = 10
    learning_rate = 0.01
    
    # Initialize data loader
    data_loader = DataLoader('data/processed_mushroom_dataset', batch_size=batch_size)
    
    # Get data splits
    train_indices, val_indices, test_indices = data_loader.split_indices()
    
    # Initialize model and monitor
    model = CNN()
    monitor = EnhancedProgressMonitor(total_epochs=epochs)
    model.monitor = monitor  # Set monitor in model
    
    # Initialize batch tracking
    monitor.current_batch = 0
    monitor.total_batches = len(train_indices) // batch_size
    
    # Log training start
    logger.log_training_start(batch_size, epochs, learning_rate)
    
    # Add debug print
    print(f"{Fore.CYAN}Total training batches: {monitor.total_batches}")
    print(f"Total validation batches: {len(val_indices) // batch_size}{Style.RESET_ALL}\n")
    
    # Training loop
    print_header("Starting Training")
    monitor.print_training_params(batch_size, epochs, learning_rate)
    
    start_time = datetime.now()
    best_accuracy = 0
    
    for epoch in range(epochs):
        monitor.print_epoch_header(epoch + 1)
        
        # Training phase
        train_progress = EnhancedBatchProgress(
            total_batches=len(train_indices) // batch_size,
            total_samples=len(train_indices),
            desc=f"🚂 Training"  # Using emoji instead of color codes
        )
        
        # Add batch counter
        batch_counter = 0
        batch_metrics = []
        
        for batch_images, batch_labels in data_loader.get_batches():
            monitor.current_batch += 1  # Update batch counter
            batch_counter += 1
            # Add debug print every few batches
            if batch_counter % 5 == 0:
                print(f"\rProcessing batch {batch_counter}", end="")
                
            if is_gpu_available:
                batch_images = to_gpu(batch_images)
                batch_labels = to_gpu(batch_labels)
                
            loss, predictions = model.train_step(batch_images, batch_labels)
            accuracy = calculate_accuracy(predictions, batch_labels)
            
            train_progress.update(
                samples_in_batch=len(batch_images),
                loss=loss,
                acc=accuracy,
                batch=batch_counter
            )
            
            # Add current metrics to monitor
            monitor.update_batch_progress(
                monitor.current_batch,
                monitor.total_batches,
                {'loss': loss, 'acc': accuracy}
            )
            
            batch_metrics.append({'loss': loss, 'acc': accuracy})
            
            # Memory cleanup
            if is_gpu_available:
                clear_gpu_memory()
        train_progress.close()
        
        # Validation phase
        val_progress = EnhancedBatchProgress(
            total_batches=len(val_indices) // batch_size,
            total_samples=len(val_indices),
            desc=f"🔍 Validation"  # Using emoji instead of color codes
        )
        
        val_metrics = []
        for batch_images, batch_labels in data_loader.get_batches():
            if is_gpu_available:
                batch_images = to_gpu(batch_images)
                batch_labels = to_gpu(batch_labels)
                
            predictions = model.forward(batch_images, training=False)
            val_loss = -np.sum(batch_labels * np.log(predictions + 1e-7)) / len(batch_labels)
            val_accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(batch_labels, axis=1))
            
            val_progress.update(
                samples_in_batch=len(batch_images),
                loss=val_loss,
                acc=val_accuracy
            )
            val_metrics.append({'loss': val_loss, 'acc': val_accuracy})
        val_progress.close()
        
        # Update and display metrics
        current_metrics = {
            'train_loss': np.mean([m['loss'] for m in batch_metrics]),
            'train_acc': np.mean([m['acc'] for m in batch_metrics]),
            'val_loss': np.mean([m['loss'] for m in val_metrics]),
            'val_acc': np.mean([m['acc'] for m in val_metrics])
        }
        
        monitor.print_metrics(current_metrics)
        
        # Log epoch metrics
        logger.log_epoch_progress(epoch + 1, epochs, current_metrics)
        
        # Save best model if needed
        if current_metrics['val_acc'] > monitor.best_metrics['val_acc']:
            monitor.best_metrics['val_acc'] = current_metrics['val_acc']
            logger.log("\nNew best accuracy achieved! Saving model...")
            print(f"\n{Fore.GREEN}New best accuracy! Saving model...{Style.RESET_ALL}")
            save_path = os.path.join("models", "saved_models", "best_model")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            model.save(save_path)
    
    training_time = datetime.now() - start_time
    
    print_header("Training Complete")
    print(f"Total training time: {training_time}")
    print(f"Best accuracy achieved: {best_accuracy:.2%}")
    
    # Log training completion
    logger.log_training_complete(best_accuracy)

def load_image(image_path):
    """Load a preprocessed image for prediction"""
    # Read image 
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    # Convert to float and normalize
    img = img.astype(np.float32) / 255.0
    
    # Reshape for CNN (batch_size=1, channels=3, height, width)
    img = np.transpose(img, (2, 0, 1))  # Change from HWC to CHW format
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    return img

def predict_mushroom(image_path, model_path="mushroom_classifier"):
    print_header("Making Prediction")
    """Predict whether a mushroom is edible or poisonous"""
    # Initialize and load model
    model = CNN()
    try:
        model.load(model_path)
    except FileNotFoundError:
        print(f"Error: Could not find saved model at {model_path}")
        return None
    
    # Load preprocessed image
    try:
        img = load_image(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Make prediction
    predictions = model.forward(img)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]
    
    # Map prediction to class name
    class_names = ["edible", "poisonous"]
    result = {
        "class": class_names[class_index],
        "confidence": float(confidence),
        "predictions": {name: float(pred) for name, pred in zip(class_names, predictions[0])}
    }
    
    if result:
        print("\nPrediction Results:")
        print(f"Classification: {result['class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print("\nDetailed Predictions:")
        for class_name, prob in result['predictions'].items():
            print(f"{class_name}: {prob:.2%}")
            
    return result

if __name__ == "__main__":
    configure_gpu()
    train()
