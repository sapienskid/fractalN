import tensorflow as tf
import os

def verify_gpu_setup():
    print("TensorFlow version:", tf.__version__)
    print("GPU is built with CUDA:", tf.test.is_built_with_cuda())
    print("GPU devices:", tf.config.list_physical_devices('GPU'))
    
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    # Try to create a simple operation on GPU
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("Matrix multiplication result:", c)
            print("Operation was performed on GPU successfully!")
    except RuntimeError as e:
        print("GPU operation failed:", e)

if __name__ == "__main__":
    verify_gpu_setup()
