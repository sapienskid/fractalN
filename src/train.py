import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import gc
from model import create_model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gpu_config import setup_gpu
tf.keras.backend.set_floatx('float32')  # Ensure we're using FP32


# Configure GPU before importing other dependencies
setup_gpu()

# Add CUDA configuration at the top of the file
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Update image parameters
IMG_HEIGHT = 224  # Reduced from 224
IMG_WIDTH = 224  # Reduced from 224
BATCH_SIZE = 32   # Reduced from 32
EPOCHS = 50  # Increased to allow more training time
LEARNING_RATE = 5e-5  # Reduced learning rate for better stability

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




def create_data_generators(img_height=224, img_width=224, batch_size=32):
    """Create enhanced data generators with strong augmentation"""
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect',
        brightness_range=[0.7, 1.3],
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    # Validation generator with minimal processing
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        'data/processed/train',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        seed=42
    )

    validation_generator = val_datagen.flow_from_directory(
        'data/processed/test',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

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

def train_model(preprocess=False):
    """
    Train the mushroom classifier model
    
    Args:
        preprocess (bool): Whether to run complete preprocessing pipeline (default: False)
    """
    # Clear memory
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Configure memory
    configure_memory()
    
    # Optional complete preprocessing pipeline
    if preprocess:
        print("Running complete preprocessing pipeline...")
        # Step 1: Organize
        from utils.reorganize_data import reorganize_mushroom_data
        reorganize_mushroom_data()
        
        # Step 2: Augment
        from utils.augment_mushroom_data import augment_mushroom_data
        augment_mushroom_data(target_count=5000)
        
        # Step 3: Preprocess
        from utils.preprocess_data import preprocess_dataset
        preprocess_dataset(
            data_dir='data/mushroom_data',
            output_dir='data/processed',
            test_size=0.2,
            img_size=(224, 224)
        )
    else:
        # Verify processed data exists
        if not os.path.exists('data/processed'):
            raise FileNotFoundError(
                "Processed data not found in data/processed. "
                "Run with preprocess=True or process data first"
            )
    
    # Create data generators
    train_generator, validation_generator = create_data_generators()
    
    # Calculate correct steps
    steps_per_epoch = len(train_generator)  # Changed to use len()
    validation_steps = len(validation_generator)  # Changed to use len()

    print(f"\nTraining configuration:")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Create model
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model = create_model(inputs)
    
    # Modified optimizer with lower learning rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        clipnorm=1.0
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

    # Update model training configuration
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Changed back to val_loss
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',  # Changed back to val_loss
            factor=0.5,  # Less aggressive reduction
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',  # Changed back to val_accuracy
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect()
        )
    ]

    # Train the model with updated parameters
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
    train_model(preprocess=True)
