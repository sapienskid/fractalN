import tensorflow as tf
def create_model(inputs, num_filters_start=64):
    """Enhanced CNN model with modern architecture principles"""
    reg = tf.keras.regularizers.l2(1e-4)
    
    def residual_block(x, filters, kernel_size=3):
        skip = x
        
        # First conv path
        x = tf.keras.layers.Conv2D(
            filters, kernel_size, 
            padding='same',
            kernel_regularizer=reg,
            kernel_initializer='he_normal'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        # Second conv path
        x = tf.keras.layers.Conv2D(
            filters, kernel_size, 
            padding='same',
            kernel_regularizer=reg,
            kernel_initializer='he_normal'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Handle different input dimensions
        if skip.shape[-1] != filters:
            skip = tf.keras.layers.Conv2D(
                filters, 1,
                kernel_regularizer=reg,
                kernel_initializer='he_normal'
            )(skip)
        
        # Add skip connection
        x = tf.keras.layers.Add()([x, skip])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    # Enhanced feature extraction
    x = tf.keras.layers.Conv2D(
        num_filters_start, 7,
        strides=2,
        padding='same',
        kernel_regularizer=reg,
        kernel_initializer='he_normal'
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Progressive residual blocks
    filter_sizes = [num_filters_start, num_filters_start*2, num_filters_start*4]
    for filters in filter_sizes:
        x = residual_block(x, filters)
        x = residual_block(x, filters)  # Double residual blocks
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.SpatialDropout2D(0.3)(x)
    
    # Advanced pooling strategy
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Enhanced dense layers
    x = tf.keras.layers.Dense(
        512,
        kernel_regularizer=reg,
        kernel_initializer='he_normal'
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    outputs = tf.keras.layers.Dense(
        1,
        activation='sigmoid',
        kernel_regularizer=reg,
        kernel_initializer='glorot_normal'
    )(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)