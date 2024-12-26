import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from src.model import create_model  # Update import path
from src.utils.muhsroom_processor import MushroomDataProcessor
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gpu_config import setup_gpu
from pathlib import Path
import albumentations as A


# Enable eager execution
tf.config.run_functions_eagerly(True)

# GPU and memory configuration
setup_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Updated hyperparameters
IMG_HEIGHT = 299
IMG_WIDTH = 299
BATCH_SIZE = 8
EPOCHS = 150
BASE_LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate gradients


def plot_training_history(history):
    """Plot and save training metrics"""
    # Save plots in results/plots directory
    plots_dir = os.path.join('results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot accuracy metrics
    plt.figure(figsize=(12, 4))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    
    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_history.png'))
    plt.close()
    
    # Plot additional metrics
    plt.figure(figsize=(15, 5))
    
    # F1 Score
    plt.subplot(1, 3, 1)
    plt.plot(history.history['f1_score'], label='Training F1')
    plt.plot(history.history['val_f1_score'], label='Validation F1')
    plt.title('Model F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend(loc='lower right')
    
    # Precision
    plt.subplot(1, 3, 2)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    
    # Recall
    plt.subplot(1, 3, 3)
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Model Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'additional_metrics.png'))
    plt.close()

def save_metrics(history, test_results, model_name='mushroom_classifier'):
    """Save training history and test results to file"""
    # Save metrics in results/metrics directory
    metrics_dir = os.path.join('results', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    with open(os.path.join(metrics_dir, f'{model_name}_metrics.txt'), 'w') as f:
        # Write training history summary
        f.write("Training History Summary:\n")
        f.write("=======================\n\n")
        
        # Final epoch metrics
        final_epoch = len(history.history['accuracy'])
        f.write(f"Total Epochs Trained: {final_epoch}\n\n")
        
        f.write("Final Training Metrics:\n")
        for metric in history.history.keys():
            if not metric.startswith('val_'):
                f.write(f"{metric}: {history.history[metric][-1]:.4f}\n")
        
        f.write("\nFinal Validation Metrics:\n")
        for metric in history.history.keys():
            if metric.startswith('val_'):
                f.write(f"{metric}: {history.history[metric][-1]:.4f}\n")
        
        # Test results
        f.write("\nTest Set Results:\n")
        f.write("================\n\n")
        for metric_name, value in zip(['loss', 'accuracy', 'auc', 'precision', 'recall', 'f1_score'], test_results):
            f.write(f"test_{metric_name}: {value:.4f}\n")
        
        # Best metrics
        f.write("\nBest Metrics Achieved:\n")
        f.write("====================\n\n")
        for metric in history.history.keys():
            if metric.startswith('val_'):
                best_value = max(history.history[metric])
                best_epoch = history.history[metric].index(best_value) + 1
                f.write(f"Best {metric}: {best_value:.4f} (Epoch {best_epoch})\n")


# Add to your GPU configuration
def configure_memory():
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if (physical_devices):
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                
            # More conservative memory limit
            tf.config.set_logical_device_configuration(
                physical_devices[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=3072)]  # Leave some VRAM for system
            )
            
        # Add this to prevent TF from pre-allocating memory
        tf.config.experimental.set_virtual_device_configuration(
            physical_devices[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)]
        )
    except Exception as e:
        print(f"Error in GPU configuration: {e}")

def verify_data_exists():
    """Verify data exists and return paths"""
    base_dir = Path('data/processed')
    
    # Handle symlink for Colab
    if os.path.islink(base_dir):
        base_dir = Path(os.readlink(base_dir))
    
    if not base_dir.exists():
        raise ValueError("Processed data directory not found!")
        
    train_dir = base_dir / 'train'
    if not train_dir.exists():
        raise ValueError("Training data directory not found!")
        
    counts = {
        'edible': len(list((train_dir / 'edible').glob('*.[Jj][Pp][Gg]'))),
        'poisonous': len(list((train_dir / 'poisonous').glob('*.[Jj][Pp][Gg]')))
    }
    
    if sum(counts.values()) == 0:
        raise ValueError("No training images found! Run with preprocess=True first.")
        
    print(f"Found {counts['edible']} edible and {counts['poisonous']} poisonous training images")
    return True

def create_datasets(data_processor, batch_size=BATCH_SIZE):
    """Create datasets with data verification"""
    try:
        verify_data_exists()
    except ValueError as e:
        print(f"Data verification failed: {e}")
        print("Running preprocessing to create required data...")
        data_processor.split_dataset()
        data_processor.verify_dataset()
        
        # Verify again after preprocessing
        verify_data_exists()
    
    # Basic normalization preprocessing
    def preprocess(x, y):
        return tf.cast(x, tf.float32) / 255.0, y
    
    # Data augmentation layer for training
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.GaussianNoise(0.1)
    ])
    
    # Add memory-efficient options
    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    
    # Calculate class weights
    total_samples = sum(len(list(Path(data_processor.output_dir / 'train' / c).glob('*.[Jj][Pp][Gg]'))) 
                       for c in ['poisonous', 'edible'])
    class_weights = {}
    for idx, class_name in enumerate(['edible', 'poisonous']):
        samples = len(list(Path(data_processor.output_dir / 'train' / class_name).glob('*.[Jj][Pp][Gg]')))
        class_weights[idx] = (1 / samples) * (total_samples / 2.0)

    # Calculate steps per epoch
    steps_per_epoch = 20000 // BATCH_SIZE

    # Update paths to handle symlinks
    train_path = Path(data_processor.output_dir) / 'train'
    val_path = Path(data_processor.output_dir) / 'val'
    test_path = Path(data_processor.output_dir) / 'test'
    
    if os.path.islink(train_path):
        train_path = Path(os.readlink(train_path))
        val_path = Path(os.readlink(val_path))
        test_path = Path(os.readlink(test_path))

    # Create training dataset with optimizations
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode='categorical',
        class_names=['edible', 'poisonous'],
        seed=42,
        shuffle=True
    )
    
    # Optimize training pipeline
    train_ds = train_ds.map(
        lambda x, y: (preprocess(data_augmentation(x), y)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.shuffle(buffer_size=1000)  # Reduced buffer size
    train_ds = train_ds.repeat()
    train_ds = train_ds.with_options(options)
    train_ds = train_ds.prefetch(2)  # Fixed small prefetch buffer
    
    # Create validation dataset with optimizations
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_path,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode='categorical',
        class_names=['edible', 'poisonous'],
        seed=42
    )
    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.with_options(options)
    val_ds = val_ds.cache()
    val_ds = val_ds.prefetch(2)
    
    # Create test dataset with optimizations
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode='categorical',
        class_names=['edible', 'poisonous'],
        seed=42
    )
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.with_options(options)
    test_ds = test_ds.cache()
    test_ds = test_ds.prefetch(2)
    
    return train_ds, val_ds, test_ds, class_weights, steps_per_epoch



def train_model(preprocess=False):
    """Train the mushroom classifier with improved error handling"""
    try:
        # Initialize data processor
        data_processor = MushroomDataProcessor()
        
        # Force preprocessing if data doesn't exist
        processed_dir = Path('data/processed/train')
        if os.path.islink(processed_dir):
            processed_dir = Path(os.readlink(processed_dir))
            
        if not processed_dir.exists() or preprocess:
            print("Starting data preprocessing...")
            data_processor.split_dataset()
            data_processor.verify_dataset()
        
        # Create datasets with automatic preprocessing if needed
        train_ds, val_ds, test_ds, class_weights, steps_per_epoch = create_datasets(data_processor)
        
        # Create and compile model
        inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
        model = create_model(inputs)
        
        # Create the learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-4,
            first_decay_steps=1000,
            alpha=1e-6
        )
        
        # Use the schedule in optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,  # Use the scheduler here
            weight_decay=1e-5,
            global_clipnorm=1.0
        )
        
            # Add LR monitoring callback
        class LRLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                lr = self.model.optimizer.learning_rate
                if hasattr(lr, 'value'):
                    lr = lr.value()
                print(f'\nLearning rate for epoch {epoch+1} was {lr:.2e}')
        
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc', num_thresholds=200),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.F1Score(name='f1_score', average='macro', threshold=None)
            ]
        )
        
        # Update callbacks list with new directories
        callbacks = [
            LRLogger(),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join('results', 'logs'),
                update_freq='epoch',
                profile_batch=0
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join('models', 'best_mushroom_model.keras'),
                monitor='val_f1_score',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_f1_score',
                mode='max',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
        ]
        # Verify class weights before training
        print("\nClass weights being used:")
        for class_idx, weight in class_weights.items():
            print(f"Class {class_idx}: {weight:.4f}")
        
        # Training
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1,
            class_weight=class_weights,
            steps_per_epoch=steps_per_epoch
        )
        
        # Evaluate model on test set
        print("\nEvaluating on test set:")
        test_results = model.evaluate(test_ds, verbose=1)
        
        # After training, save final model to models directory
        os.makedirs('models', exist_ok=True)
        model.save(os.path.join('models', 'mushroom_classifier.keras'))
        
        # Save results
        plot_training_history(history)
        save_metrics(history, test_results)
        
        print("\nTest Results:")
        for metric_name, value in zip(model.metrics_names, test_results):
            print(f"{metric_name}: {value:.4f}")
            
        print("\nModel and results saved in:")
        print("- Models: ./models/")
        print("- Plots: ./results/plots/")
        print("- Metrics: ./results/metrics/")
        print("- Logs: ./results/logs/")
        
        return model, history

    except Exception as e:
        print(f"Error during training: {e}")
        print("\nPlease ensure you have:")
        print("1. Uploaded your data correctly")
        print("2. Run with preprocess=True for first time setup")
        print("3. Have the correct directory structure")
        raise

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs(os.path.join('results', 'plots'), exist_ok=True)
    os.makedirs(os.path.join('results', 'metrics'), exist_ok=True)
    os.makedirs(os.path.join('results', 'logs'), exist_ok=True)
    
    model, history = train_model(preprocess=False)