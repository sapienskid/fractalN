import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def create_model(inputs, num_filters_start=32):
    """Simplified CNN model for mushroom classification"""
    reg = tf.keras.regularizers.l2(1e-6)
    
    def conv_block(x, filters):
        x = tf.keras.layers.Conv2D(
            filters, 3,
            padding='same',
            kernel_regularizer=reg
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        return x
    
    # Simpler CNN structure
    x = conv_block(inputs, num_filters_start)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, num_filters_start * 2)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, num_filters_start * 4)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Single dense layer classifier
    x = tf.keras.layers.Dense(128, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)