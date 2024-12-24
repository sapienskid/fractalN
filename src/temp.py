import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Add this before importing tensorflow
import tensorflow as tf
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import gc
from temp_model import create_model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gpu_config import setup_gpu
from utils.data_pipeline import DataPipeline
import logging
tf.get_logger().setLevel(logging.ERROR)
# Configure GPU and floating point precision
setup_gpu()
tf.keras.backend.set_floatx('float32')

# Training parameters
IMG_HEIGHT = 256         # Reduced image size
IMG_WIDTH = 256          # Reduced image size
BATCH_SIZE = 32          # Reduced batch size
EPOCHS = 50            # Increased epochs
LEARNING_RATE = 1e-4    # Reduced learning rate
warmup_epochs = 10      # Increased warmup epochs
decay_epochs = EPOCHS - warmup_epochs  # Unused now

def configure_memory():
    """Configure GPU memory settings"""
    try:
        if tf.test.is_gpu_available():
            print("GPU is configured and ready")
            tf.debugging.set_log_device_placement(True)
        else:
            print("Warning: GPU is not available")
            print("CUDA Path:", os.environ.get('CUDA_HOME'))
            print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))
    except Exception as e:
        print(f"Error in GPU configuration: {e}")

def plot_training_history(history):
    """Plot and save training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def save_metrics(history, test_metrics):
    """Save training and test metrics to file"""
    with open('training_metrics.txt', 'w') as f:
        f.write("Training History:\n")
        f.write(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}\n")
        f.write(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}\n")
        f.write(f"Final training loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final validation loss: {history.history['val_loss'][-1]:.4f}\n\n")
        
        f.write("Test Metrics:\n")
        f.write(f"Test loss: {test_metrics[0]:.4f}\n")
        f.write(f"Test accuracy: {test_metrics[1]:.4f}\n")
        f.write(f"Test AUC: {test_metrics[2]:.4f}\n")

def create_data_augmentation():
    """Create data augmentation layer"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

def train_model(preprocess=False):
    """Main training function"""
    # Clear memory
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Configure memory
    configure_memory()
    
    # Initialize data pipeline
    pipeline = DataPipeline(
        img_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )
    
    # Get datasets
    datasets = pipeline.setup_data(preprocess=preprocess)
    train_ds = datasets['train']
    val_ds = datasets['validation']
    test_ds = datasets['test']
    
    # Calculate steps
    steps_per_epoch = len(train_ds)
    validation_steps = len(val_ds)

    print(f"\nTraining configuration:")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Add data augmentation to model
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    data_augmentation = create_data_augmentation()
    x = data_augmentation(inputs)
    model = create_model(x)

    # Update learning rate and optimizer
    initial_learning_rate = 1e-3
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps=5 * steps_per_epoch,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-5
    )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-5,
        beta_1=0.9,
        beta_2=0.999
    )

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # Increased patience
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001  # Minimum improvement required
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_mushroom_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
    ]

    # Get class weights from datasets
    class_weights = datasets['class_weights']
    
    # Train model with class weights
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
        class_weight=class_weights  # Add class weights here
    )

    # Evaluate model
    print("\nEvaluating model on test set...")
    test_metrics = model.evaluate(
        test_ds,
        steps=len(test_ds)
    )

    print(f"\nTest loss: {test_metrics[0]:.4f}")
    print(f"Test accuracy: {test_metrics[1]:.4f}")
    print(f"Test AUC: {test_metrics[2]:.4f}")

    # Save results
    plot_training_history(history)
    save_metrics(history, test_metrics)

    # Save final model
    model.save('mushroom_classifier.keras')

if __name__ == "__main__":
    train_model(preprocess=True)