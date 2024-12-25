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
BATCH_SIZE = 16  # Reduced batch size
EPOCHS = 50
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


def configure_memory():
    try:
        # Memory growth must be set before GPUs are initialized
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                
            # Limit memory allocation
            tf.config.set_logical_device_configuration(
                physical_devices[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
            )
        else:
            print("Warning: No GPU found")
    except Exception as e:
        print(f"Error in GPU configuration: {e}")

def create_datasets(data_processor, batch_size=BATCH_SIZE):
    """Create datasets using tf.data pipeline"""
    
    # Convert images to float32 and normalize
    def preprocess(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_processor.output_dir / 'train',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode='categorical',  # Keep categorical for 2 classes
        class_names=['edible', 'poisonous'],
        seed=42,
        shuffle=True
    ).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.with_options(options)
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_processor.output_dir / 'val',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode='categorical',
        class_names=['edible', 'poisonous'],
        seed=42
    ).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_processor.output_dir / 'test',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size,
        label_mode='categorical',
        class_names=['edible', 'poisonous'],
        seed=42
    ).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Configure datasets for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, test_ds

def train_model(preprocess=False):
    """Train the mushroom classifier with the new data pipeline"""
    configure_memory()
    
    # Initialize data processor
    data_processor = MushroomDataProcessor()
    if os.path.exists('data/processed') and os.path.exists('data/processed/train') and os.path.exists('data/processed/test') and os.path.exists('data/processed/val'):
        print("\nProcessed data already exists. Skipping preprocessing...")
        # Verify data exists in both directories
        train_files = list(Path('data/processed/train').rglob('*.[Jj][Pp][Gg]'))
        test_files = list(Path('data/processed/test').rglob('*.[Jj][Pp][Gg]'))
        val_files = list(Path('data/processed/val').rglob('*.[Jj][Pp][Gg]'))
        if train_files and test_files:
            print(f"Found {len(train_files)} training images, {len(val_files)} validation images, {len(test_files)} test images")
            preprocess = False
        else:
            print("Existing processed directories are empty. Will preprocess data...")
    
    # Run preprocessing if needed
    if preprocess:
        data_processor.create_balanced_dataset(target_count=5000)
        data_processor.split_dataset()
        data_processor.verify_dataset()
    
    # Create datasets
    train_ds, val_ds, test_ds = create_datasets(data_processor)
    
    # Create and compile model
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model = create_model(inputs)
    
    # Add cosine decay learning rate schedule
    initial_learning_rate = BASE_LEARNING_RATE
    # decay_steps = EPOCHS * len(train_ds)
    # lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate,
    #     decay_steps,
    #     alpha=1e-6
    # )
    
    # Update optimizer with schedule
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=BASE_LEARNING_RATE,
        global_clipnorm=1.0
    )
        
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', num_thresholds=200),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.F1Score(name='f1_score', average='macro', threshold=None)  # Changed parameters
        ]
    )
    
    # Enhanced callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_f1_score',
            mode='max',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_f1_score',
            mode='max',
            save_best_only=True,
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
        
        
        # Add CSV logger
        tf.keras.callbacks.CSVLogger('training_log.csv')
    ]
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set:")
    test_results = model.evaluate(test_ds, verbose=1)
    
    # Plot and save metrics
    plot_training_history(history)
    save_metrics(history, test_results)
    
    # Print final results
    print("\nTest Results:")
    for metric_name, value in zip(model.metrics_names, test_results):
        print(f"{metric_name}: {value:.4f}")
    
    return model, history

if __name__ == "__main__":
    model, history = train_model(preprocess=True)