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
from gpu_utils import configure_gpu, clear_gpu_memory, to_gpu, to_cpu, GPU_AVAILABLE, get_gpu_info
from progress_monitor import EnhancedProgressMonitor, EnhancedBatchProgress, print_gpu_status
from colorama import Fore, Style, init
from logger import TrainingLogger
from optimizers import Adam, RMSprop, LearningRateScheduler, AdaptiveMomentum, AdaptiveLearningRate
from exceptions import ValidationError, ShapeError, GPUError, MemoryError, GPUMemoryTracker
init(autoreset=True)

class ValidationContext:
    """Context manager for validation phase"""
    def __enter__(self):
        self.training_state = False
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.training_state = True

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

def calculate_loss(predictions, labels):
    """Calculate cross entropy loss ensuring consistent array types"""
    xp = cp if GPU_AVAILABLE else np
    predictions = to_gpu(predictions)
    labels = to_gpu(labels)
    
    # Add small epsilon to prevent log(0)
    epsilon = 1e-7
    predictions = xp.clip(predictions, epsilon, 1-epsilon)
    
    # Calculate cross entropy loss
    loss = -xp.sum(labels * xp.log(predictions)) / len(labels)
    return float(to_cpu(loss))

def calculate_accuracy(predictions, labels):
    """Calculate accuracy ensuring consistent array types"""
    predictions = to_cpu(predictions)
    labels = to_cpu(labels)
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))

def train():
    # Initialize logger and monitor
    logger = TrainingLogger()
    
    # Configure GPU first and get status
    is_gpu_available, gpu_config = configure_gpu()
    
    # Training parameters with adaptive components
    batch_size = gpu_config['optimal_batch_size'] if gpu_config else 8
    epochs = 10
    initial_lr = 0.01
    
    # Initialize optimizers and schedulers
    optimizer = Adam(learning_rate=initial_lr)
    lr_scheduler = LearningRateScheduler(initial_lr=initial_lr)
    momentum_adapter = AdaptiveMomentum(base_momentum=0.9)
    adaptive_lr = AdaptiveLearningRate(initial_lr=initial_lr)
    
    try:
        # Initialize data loader with validation
        data_loader = DataLoader('data/processed_mushroom_dataset', batch_size=batch_size)
        
        # Get data splits
        train_indices, val_indices, test_indices = data_loader.split_indices()
        
        # Initialize model with optimizers
        model = CNN(optimizer='adam', lr=initial_lr)
        monitor = EnhancedProgressMonitor(total_epochs=epochs)
        model.monitor = monitor
        
        # Initialize batch tracking
        monitor.current_batch = 0
        monitor.total_batches = len(train_indices) // batch_size
        
        # Log training start
        logger.log_training_start(batch_size, epochs, initial_lr)
        
        # Training loop with memory tracking and adaptive components
        best_accuracy = 0
        patience_counter = 0
        
        print_gpu_status(get_gpu_info())  # Add this line
        
        for epoch in range(epochs):
            with GPUMemoryTracker(threshold_mb=1000) as memory_tracker:
                # Update learning rate using scheduler
                current_lr = lr_scheduler.cosine_decay(epoch, epochs)
                optimizer.learning_rate = current_lr
                
                monitor.print_epoch_header(epoch + 1)
                
                # Training phase with gradient accumulation
                train_progress = EnhancedBatchProgress(
                    total_batches=len(train_indices) // batch_size,
                    total_samples=len(train_indices),
                    desc=f"🚂 Training"
                )
                
                train_metrics = []
                batch_counter = 0
                
                for batch_images, batch_labels in data_loader.get_batches():
                    try:
                        # Memory-efficient training step
                        loss, predictions = model.train_step(batch_images, batch_labels)
                        accuracy = calculate_accuracy(predictions, batch_labels)
                        
                        batch_counter += 1
                        train_progress.update(
                            samples_in_batch=len(batch_images),
                            loss=loss,
                            acc=accuracy,
                            batch=batch_counter
                        )
                        
                        train_metrics.append({
                            'loss': loss,
                            'accuracy': accuracy
                        })
                        
                        # Adaptive momentum based on gradients
                        if hasattr(model, 'layers'):
                            for layer in model.layers:
                                if hasattr(layer, 'optimizer'):
                                    layer.momentum = momentum_adapter.get_momentum(
                                        f'layer_{id(layer)}',
                                        getattr(layer, 'last_gradient', None)
                                    )
                        
                    except MemoryError as e:
                        logger.log(f"Memory error in batch: {str(e)}")
                        clear_gpu_memory(0)
                        continue
                    
                train_progress.close()
                
                # Validation phase
                val_progress = EnhancedBatchProgress(
                    total_batches=len(val_indices) // batch_size,
                    total_samples=len(val_indices),
                    desc=f"🔍 Validation"
                )
                
                val_metrics = []
                with ValidationContext():
                    for batch_images, batch_labels in data_loader.get_batches(validation=True):
                        predictions = model.forward(batch_images, training=False)
                        val_loss = calculate_loss(predictions, batch_labels)
                        val_accuracy = calculate_accuracy(predictions, batch_labels)
                        
                        val_progress.update(
                            samples_in_batch=len(batch_images),
                            loss=val_loss,
                            acc=val_accuracy
                        )
                        
                        val_metrics.append({
                            'loss': val_loss,
                            'accuracy': val_accuracy
                        })
                
                val_progress.close()
                
                # Calculate epoch metrics
                epoch_metrics = {
                    'train_loss': np.mean([m['loss'] for m in train_metrics]),
                    'train_acc': np.mean([m['accuracy'] for m in train_metrics]),
                    'val_loss': np.mean([m['loss'] for m in val_metrics]),
                    'val_acc': np.mean([m['accuracy'] for m in val_metrics])
                }
                
                # Adaptive learning rate adjustment
                if adaptive_lr.should_reduce_lr(epoch_metrics['val_loss']):
                    logger.log(f"Reducing learning rate to {adaptive_lr.current_lr}")
                    optimizer.learning_rate = adaptive_lr.current_lr
                
                # Save best model
                if epoch_metrics['val_acc'] > best_accuracy:
                    best_accuracy = epoch_metrics['val_acc']
                    patience_counter = 0
                    save_path = os.path.join("models", "saved_models", "best_model")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    model.save(save_path)
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= 5:  # 5 epochs without improvement
                    logger.log("Early stopping triggered")
                    break
                
                # Log epoch results
                logger.log_epoch_progress(epoch + 1, epochs, epoch_metrics)
                monitor.print_metrics(epoch_metrics)
                
                # Log memory usage
                memory_usage = get_memory_usage()
                logger.log(f"Memory usage - RAM: {memory_usage['RAM']}, GPU: {memory_usage['GPU']}")
                
    except Exception as e:
        logger.log(f"Training failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if is_gpu_available:
            clear_gpu_memory(0)

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
