import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Add this before importing tensorflow
import tensorflow as tf
import subprocess
import re

def get_nvidia_smi_output():
    try:
        output = subprocess.check_output(['nvidia-smi']).decode()
        return output
    except:
        return "nvidia-smi command failed"

def get_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode()
        version = re.search(r'release (\d+\.\d+)', output)
        return version.group(1) if version else None
    except:
        return None

def setup_gpu():
    # Check CUDA version
    cuda_version = get_cuda_version()
    print(f"Detected CUDA version: {cuda_version}")
    
    # Set environment variables with explicit versions
    cuda_paths = {
        'CUDA_HOME': '/opt/cuda',
        'CUDA_ROOT': '/opt/cuda',
        'LD_LIBRARY_PATH': '/opt/cuda/lib64:/opt/cuda/extras/CUPTI/lib64:/usr/lib',
        'XLA_FLAGS': '--xla_gpu_cuda_data_dir=/opt/cuda',
        'TF_CUDA_PATHS': '/opt/cuda',
        'TF_CUDA_VERSION': '11.8',
        'TF_CUDNN_VERSION': '8',
        'CUDA_CACHE_PATH': os.path.expanduser('~/.cache/cuda')
    }
    
    # Update environment variables
    for key, value in cuda_paths.items():
        if key in os.environ:
            if value not in os.environ[key]:
                os.environ[key] = f"{value}:{os.environ[key]}"
        else:
            os.environ[key] = value

    # Print NVIDIA GPU information
    print("\nNVIDIA GPU Information:")
    print(get_nvidia_smi_output())
    
    try:
        # Verify CUDA libraries
        cudart_path = "/opt/cuda/lib64/libcudart.so"
        if os.path.exists(cudart_path):
            real_path = os.path.realpath(cudart_path)
            print(f"Using CUDA RT library: {real_path}")
        else:
            print("Warning: libcudart.so not found")

        # Configure GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"\nFound {len(physical_devices)} GPU(s)")
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                print(f"Enabled memory growth for {device.name}")
        
        # Test GPU operation
        if physical_devices:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                c = tf.matmul(a, b)
                print("\nGPU test result:", c.numpy())
                print("Using device:", c.device)
    
    except Exception as e:
        print(f"\nError during GPU setup: {e}")
        print("\nDebug information:")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"CUDA available: {tf.test.is_built_with_cuda()}")
        print(f"GPU available: {tf.test.is_gpu_available()}")
        for key, value in cuda_paths.items():
            print(f"{key}: {os.environ.get(key, 'Not set')}")
