import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import gc
from model import create_model
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_config import setup_gpu

# Configure GPU before importing other dependencies
setup_gpu()

# Add CUDA configuration at the top of the file
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Update image parameters
IMG_HEIGHT = 160  # Reduced from 224
IMG_WIDTH = 160   # Reduced from 224
BATCH_SIZE = 8    # Further reduced batch size
EPOCHS = 50

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
    # Enable memory efficient mode
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

    # Print directory contents before creating generators
    print("\nChecking data directory contents:")
    for split in ['train', 'test']:
        for category in ['poisonous', 'edible']:
            path = os.path.join('data', split, category)
            files = os.listdir(path)
            print(f"{split}/{category}: {len(files)} files")

    train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        seed=42,
        color_mode='rgb',
        interpolation='bilinear'
    )

    validation_generator = test_datagen.flow_from_directory(
        'data/test',  # Using test data for validation
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False,  # No need to shuffle validation data
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
    
    # Calculate steps
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = validation_generator.samples // BATCH_SIZE
    
    # Create model
    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model = create_model(inputs)
    
    # Modified learning rate and optimizer setup with proper type casting
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, initial_learning_rate, decay_steps, decay_rate):
            self.initial_learning_rate = float(initial_learning_rate)
            self.decay_steps = int(decay_steps)
            self.decay_rate = float(decay_rate)
        
        def __call__(self, step):
            step_float = tf.cast(step, tf.float32)
            decay_steps_float = tf.cast(self.decay_steps, tf.float32)
            decay_factor = tf.pow(self.decay_rate, tf.floor(step_float / decay_steps_float))
            return self.initial_learning_rate * decay_factor

        def get_config(self):
            return {
                "initial_learning_rate": self.initial_learning_rate,
                "decay_steps": self.decay_steps,
                "decay_rate": self.decay_rate
            }
    
    # Create optimizer with fixed learning rate first
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    
    # Compile model
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

    # Add learning rate scheduler callback instead of custom schedule
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        ),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect()
        )
    ]

    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    # Final evaluation
    val_metrics = model.evaluate(validation_generator)
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
