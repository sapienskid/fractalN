#model.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Add this before importing tensorflow

import tensorflow as tf

def create_model(inputs, num_filters_start=48):
    """Enhanced CNN model with stronger regularization"""
    reg = tf.keras.regularizers.l2(1e-4)  # Increased regularization
    
    def conv_block(x, filters, kernel_size=3, strides=1):
        # Improved conv block with residual connection
        skip = x
        
        # Add spatial dropout
        x = tf.keras.layers.SpatialDropout2D(0.1)(x)
        
        x = tf.keras.layers.Conv2D(
            filters, kernel_size,
            strides=strides,
            padding='same',
            kernel_regularizer=reg,
            kernel_initializer='he_uniform'
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        x = tf.keras.layers.Activation('swish')(x)  # Using swish activation
        
        x = tf.keras.layers.Conv2D(
            filters, kernel_size,
            padding='same',
            kernel_regularizer=reg,
            kernel_initializer='he_uniform'
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
        
        # Add residual connection if shapes match
        if strides == 1 and x.shape[-1] == skip.shape[-1]:
            x = tf.keras.layers.Add()([x, skip])
        
        x = tf.keras.layers.Activation('swish')(x)
        
        # Increase dropout rates
        x = tf.keras.layers.Dropout(0.3)(x)
        return x
    
    # Initial feature extraction
    x = conv_block(inputs, num_filters_start)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    # Deeper network with gradual filter increase
    x = conv_block(x, num_filters_start * 2)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = conv_block(x, num_filters_start * 4)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = conv_block(x, num_filters_start * 8)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Multi-layer classifier head
    x = tf.keras.layers.Dense(
        256,
        kernel_regularizer=reg,
        kernel_initializer='he_uniform'
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(
        128,
        kernel_regularizer=reg,
        kernel_initializer='he_uniform'
    )(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9)(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_initializer='glorot_uniform',
        bias_initializer=tf.keras.initializers.Constant(-0.2)  # Slight bias toward negative class
    )(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)
