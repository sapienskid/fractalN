import tensorflow as tf

def create_model(inputs):
    # Use mixed precision to save memory
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    # Reduce regularization strength
    reg = tf.keras.regularizers.l2(1e-5)  # Reduced from 1e-4
    
    # First conv block with reduced dropout
    x = tf.keras.layers.Conv2D(16, 3, padding='same', kernel_regularizer=reg)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)  # Reduced from 0.25

    # Second conv block
    x = tf.keras.layers.Conv2D(32, 3, padding='same', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)  # Reduced from 0.25

    # Third conv block
    x = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)  # Reduced from 0.25

    # Dense layers with reduced dropout
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)  # Reduced from 0.5
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
