import tensorflow as tf
import os
from gpu_config import setup_gpu

def verify_gpu_setup():
    # Setup GPU configuration
    setup_gpu()
    
    # Additional verification
    print("\nDetailed GPU Information:")
    print("CUDA toolkit path:", os.environ.get('CUDA_HOME', 'Not set'))
    print("CUDA library path:", os.environ.get('LD_LIBRARY_PATH', 'Not set'))
    
    # Try GPU operation with error handling
    try:
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
            print("\nGPU Operation Result:")
            print(c.numpy())
            print("Device:", c.device)
    except Exception as e:
        print("\nGPU operation failed:", e)
        print("Falling back to CPU")

if __name__ == "__main__":
    verify_gpu_setup()
