import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from model import create_model
from utils.muhsroom_processor import MushroomDataProcessor
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
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8
EPOCHS = 150
BASE_LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate gradients


def plot_training_history(history):
    """Plot and save training metrics"""
    # Create directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
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
    plt.savefig('plots/training_history.png')
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
    plt.savefig('plots/additional_metrics.png')
    plt.close()

def save_metrics(history, test_results, model_name='mushroom_classifier'):
    """Save training history and test results to file"""
    os.makedirs('metrics', exist_ok=True)
    
    with open(f'metrics/{model_name}_metrics.txt', 'w') as f:
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
        if physical_devices:
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

def create_datasets(data_processor, batch_size=BATCH_SIZE):
    """Create datasets using TensorFlow's built-in functionality with memory optimizations"""
    
    # Basic normalization preprocessing
    def preprocess(x, y):
        return tf.cast(x, tf.float32) / 255.0, y
    
    # Data augmentation layer for training
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
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

    # Create training dataset with optimizations
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_processor.output_dir / 'train',
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
        data_processor.output_dir / 'val',
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
        data_processor.output_dir / 'test',
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
    """Train the mushroom classifier"""
    
    # Initialize data processor
    data_processor = MushroomDataProcessor()
    if os.path.exists('data/processed') and os.path.exists('data/processed/train') and os.path.exists('data/processed/test') and os.path.exists('data/processed/val'):
        print("\nProcessed data already exists. Skipping preprocessing...")
        train_files = list(Path('data/processed/train').rglob('*.[Jj][Pp][Gg]'))
        test_files = list(Path('data/processed/test').rglob('*.[Jj][Pp][Gg]'))
        val_files = list(Path('data/processed/val').rglob('*.[Jj][Pp][Gg]'))
        if train_files and test_files:
            print(f"Found {len(train_files)} training images, {len(val_files)} validation images, {len(test_files)} test images")
            preprocess = False
        else:
            print("Existing processed directories are empty. Will preprocess data...")
    
    if preprocess:
        data_processor.create_balanced_dataset(target_count=5000)
        data_processor.split_dataset()
        data_processor.verify_dataset()
    
    # Create datasets
    train_ds, val_ds, test_ds, class_weights, steps_per_epoch = create_datasets(data_processor)
    
    # Create and compile model
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model = create_model(inputs)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=BASE_LEARNING_RATE,
        global_clipnorm=1.0
    )
    
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
    
    # Fix callbacks configuration
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_mushroom_model.keras',
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,  # Fixed from save_best_weights
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            update_freq='epoch'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights,
        steps_per_epoch=steps_per_epoch
    )
    
    # Evaluate and save results
    print("\nEvaluating on test set:")
    test_results = model.evaluate(test_ds, verbose=1)
    
    plot_training_history(history)
    save_metrics(history, test_results)
    
    print("\nTest Results:")
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"{metric_name}: {value:.4f}")
    
    return model, history

if __name__ == "__main__":
    model, history = train_model(preprocess=False)