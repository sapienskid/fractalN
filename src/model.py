import tensorflow as tf

def create_model(inputs, num_filters_start=32):  # Increased from 16
    """Enhanced CNN model with residual connections and better regularization"""
    reg = tf.keras.regularizers.l2(1e-5)  # Increased regularization
    
    def conv_block(x, filters, kernel_size=3):
        # Store input for residual connection
        shortcut = x
        
        # First conv layer
        x = tf.keras.layers.Conv2D(
            filters, kernel_size,
            padding='same',
            kernel_regularizer=reg,
            kernel_initializer='he_normal'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        # Second conv layer
        x = tf.keras.layers.Conv2D(
            filters, kernel_size,
            padding='same',
            kernel_regularizer=reg,
            kernel_initializer='he_normal'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Add residual connection if shapes match
        if shortcut.shape[-1] == filters:
            x = tf.keras.layers.Add()([shortcut, x])
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Dropout(0.3)(x)  # Increased dropout
        return x
    
    # Input preprocessing
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = tf.keras.layers.RandomBrightness(0.2)(x)
    x = tf.keras.layers.RandomContrast(0.2)(x)
    
    # Deeper architecture with residual connections
    x = conv_block(x, num_filters_start)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, num_filters_start * 2)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, num_filters_start * 4)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = conv_block(x, num_filters_start * 8)
    
    # Global pooling with attention
    attention = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([x, attention])
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Improved classifier head
    x = tf.keras.layers.Dense(256, kernel_regularizer=reg)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)  # Increased dropout
    
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)