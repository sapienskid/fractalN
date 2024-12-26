import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def create_model(inputs, num_filters_start=16):  # Reduced from 32
    """Memory-optimized CNN model"""
    reg = tf.keras.regularizers.l2(1e-6)
    
    def conv_block(x, filters, kernel_size=3):
        x = tf.keras.layers.Conv2D(
            filters, kernel_size,
            padding='same',
            kernel_regularizer=reg,
            use_bias=False  # Reduce parameters
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        return x
    
    # Reduced feature maps progression
    x = conv_block(inputs, num_filters_start)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, num_filters_start * 2)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, num_filters_start * 3)  # Changed from *4
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    x = tf.keras.layers.Dense(64, kernel_regularizer=reg)(x)  # Reduced from 128
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)