import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import gc
from model import create_model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gpu_config import setup_gpu

# Configure GPU before importing other dependencies
setup_gpu()

# Add CUDA configuration at the top of the file
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Update image parameters
IMG_HEIGHT = 224  # Reduced from 224
IMG_WIDTH = 224   # Reduced from 224
BATCH_SIZE = 16    # Further reduced batch size
EPOCHS = 10

def configure_memory():
    try:
        # Use simpler GPU configuration since we handled it in setup_gpu()
        if tf.test.is_gpu_available():
            print("GPU is configured and ready")
            # Print device placement for operations
            tf.debugging.set_log_device_placement(True)
        else:
            print("Warning: GPU is not available")
            print("CUDA Path:", os.environ.get('CUDA_HOME'))
            print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))
    except Exception as e:
        print(f"Error in GPU configuration: {e}")

def create_data_generators():
    # Simple preprocessing only since images are already augmented
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    # Update paths to use processed data
    train_generator = train_datagen.flow_from_directory(
        'data/processed/train',  # Updated path
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        seed=42,
        color_mode='rgb'
    )

    validation_generator = test_datagen.flow_from_directory(
        'data/processed/test',  # Updated path
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,
        color_mode='rgb'
    )

    # Print class distribution
    print("\nClass distribution in generators:")
    print(f"Train generator: {train_generator.samples} total samples")
    print(f"Train class indices: {train_generator.class_indices}")
    print(f"Train class counts: {np.bincount(train_generator.classes)}")
    print(f"Validation generator: {validation_generator.samples} total samples")
    print(f"Validation class counts: {np.bincount(validation_generator.classes)}")

    return train_generator, validation_generator

def plot_training_history(history):
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
    # Save training history to file
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

def train_model():
    # Clear memory
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Configure memory
    configure_memory()
    
    train_generator, validation_generator = create_data_generators()
    
    # Calculate correct steps
    steps_per_epoch = train_generator.n // BATCH_SIZE
    validation_steps = validation_generator.n // BATCH_SIZE

    # Ensure we don't run out of data
    if train_generator.n % BATCH_SIZE != 0:
        steps_per_epoch += 1
    if validation_generator.n % BATCH_SIZE != 0:
        validation_steps += 1

    print(f"\nTraining configuration:")
    print(f"Training samples: {train_generator.n}")
    print(f"Validation samples: {validation_generator.n}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Create model
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model = create_model(inputs)
    
    # Modified optimizer with lower learning rate
    initial_learning_rate = 0.0001  # Reduced from 0.001
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=initial_learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        clipnorm=1.0  # Added gradient clipping
    )
    
    # Compile model with added gradient clipping
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

    # Update callbacks with more patience
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased from 10
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,  # Changed from 0.2
            patience=7,   # Increased from 5
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Add progress logging callback
        tf.keras.callbacks.CSVLogger('training_log.csv'),
        # Add memory cleanup callback
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect()
        )
    ]

    # Train the model with simplified parameters
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
        shuffle=True
    )

    # Reset generators before evaluation
    validation_generator.reset()
    
    # Evaluate model
    val_metrics = model.evaluate(
        validation_generator,
        steps=validation_steps
    )

    print(f"\nValidation loss: {val_metrics[0]:.4f}")
    print(f"Validation accuracy: {val_metrics[1]:.4f}")
    print(f"Validation AUC: {val_metrics[2]:.4f}")

    # Plot and save training history
    plot_training_history(history)
    save_metrics(history, val_metrics)

    # Save the final model
    model.save('mushroom_classifier.keras')  # Changed from .h5 to .keras

if __name__ == "__main__":
    train_model()
