import tensorflow as tf

def create_model(inputs, num_filters_start=16):  # Reduced from 32
    """Lightweight CNN model with memory optimizations"""
    reg = tf.keras.regularizers.l2(1e-3)
    
    def conv_block(x, filters, kernel_size=3, strides=1):
        skip = x
        
        # First conv layer with SE block
        x = tf.keras.layers.Conv2D(
            filters, kernel_size,
            strides=strides,
            padding='same',
            kernel_regularizer=reg,
            kernel_initializer='he_normal'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)  # Using swish activation
        
        # Second conv layer
        x = tf.keras.layers.Conv2D(
            filters, kernel_size,
            padding='same',
            kernel_regularizer=reg,
            kernel_initializer='he_normal'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Squeeze-and-Excitation block
        se = tf.keras.layers.GlobalAveragePooling2D()(x)
        se = tf.keras.layers.Dense(max(filters // 8, 4), activation='swish')(se)  # Reduced size
        se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
        se = tf.keras.layers.Reshape((1, 1, filters))(se)
        x = tf.keras.layers.Multiply()([x, se])
        
        # Residual connection
        if strides == 1 and x.shape[-1] == skip.shape[-1]:
            x = tf.keras.layers.Add()([x, skip])
        elif strides > 1 or x.shape[-1] != skip.shape[-1]:
            skip = tf.keras.layers.Conv2D(
                filters, 1, 
                strides=strides,
                padding='same',
                kernel_regularizer=reg
            )(skip)
            x = tf.keras.layers.Add()([x, skip])
            
        x = tf.keras.layers.Activation('swish')(x)
        # Add spatial dropout after SE block
        x = tf.keras.layers.SpatialDropout2D(0.2)(x)
        return x
    
    # Simpler architecture
    x = conv_block(inputs, num_filters_start)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.SpatialDropout2D(0.2)(x)
    
    x = conv_block(x, num_filters_start * 2)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Simpler classifier
    x = tf.keras.layers.Dense(32, kernel_regularizer=reg)(x)  # Reduced from 64
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('swish')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)