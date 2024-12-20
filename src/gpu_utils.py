import numpy as np
from colorama import Fore, Style
try:
    import cupy as cp
    GPU_AVAILABLE = hasattr(cp, 'cuda') and cp.cuda.is_available()
except ImportError:
    cp = None
    GPU_AVAILABLE = False

def get_gpu_info():
    """Get GPU device information"""
    if not GPU_AVAILABLE:
        return None
    try:
        device = cp.cuda.Device(0)
        props = device.attributes
        name = props.get('DEVICE_NAME', b'Unknown GPU')
        # Handle both string and bytes types for device name
        if isinstance(name, bytes):
            name = name.decode('utf-8')
            
        return {
            'name': name,
            'compute_capability': f"{props.get('COMPUTE_CAPABILITY_MAJOR', 0)}.{props.get('COMPUTE_CAPABILITY_MINOR', 0)}",
            'total_memory': device.mem_info[1],
            'cuda_version': cp.cuda.runtime.runtimeGetVersion()
        }
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None

def is_high_memory_gpu():
    """Detect if running on a high-memory GPU (>=12GB)"""
    if not GPU_AVAILABLE:
        return False
    try:
        device = cp.cuda.Device(0)
        total_mem_gb = device.mem_info[1] / (1024**3)
        return total_mem_gb >= 12
    except:
        return False

def get_gpu_config():
    """Get GPU configuration without setting up memory pool"""
    if not GPU_AVAILABLE:
        return None
        
    try:
        device = cp.cuda.Device(0)
        total_mem_gb = device.mem_info[1] / (1024**3)
        
        return {
            'is_high_memory': total_mem_gb >= 12,
            'total_memory': total_mem_gb,
            'optimal_batch_size': 64 if total_mem_gb >= 12 else 4,
            'pool_size': int(0.8 * total_mem_gb * 1024**3),
            'enable_parallel': total_mem_gb >= 12
        }
    except:
        return None

def check_cudnn_installation():
    """Detailed check of cuDNN installation"""
    if not GPU_AVAILABLE:
        print("CUDA is not available")
        return False
        
    try:
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        print(f"CUDA Version: {cuda_version}")
        
        # Check cuDNN availability
        if hasattr(cp.cuda, 'cudnn') and cp.cuda.cudnn.available:
            cudnn_version = cp.cuda.cudnn.getVersion()
            print(f"cuDNN Version: {cudnn_version}")
            
            # Test cuDNN functionality with simpler test
            try:
                # Create a test tensor
                x = cp.random.random((1, 1, 3, 3)).astype(cp.float32)
                
                # Test basic cuDNN operation (convolution)
                w = cp.random.random((1, 1, 2, 2)).astype(cp.float32)
                y = cp.cuda.cudnn.convolution_forward(
                    x, w,
                    pad=((0, 0), (0, 0)),
                    stride=(1, 1),
                    dilation=(1, 1),
                    groups=1
                )
                print("cuDNN is working properly")
                return True
            except Exception as e:
                print(f"cuDNN functionality test failed: {e}")
                return False
        else:
            print("cuDNN is not available")
            return False
            
    except Exception as e:
        print(f"Error checking CUDA/cuDNN: {e}")
        return False

def configure_gpu():
    """Configure GPU and return status and config"""
    if not GPU_AVAILABLE:
        return False, None

    try:
        print("\nChecking CUDA and cuDNN installation:")
        cudnn_available = check_cudnn_installation()
        
        device = cp.cuda.Device(0)
        total_mem_gb = device.mem_info[1] / (1024**3)
        
        # Configure based on available memory
        is_high_memory = total_mem_gb >= 8  # Lowered threshold
        optimal_batch_size = min(32 if total_mem_gb >= 8 else 4, 16)
        pool_size = int(0.7 * total_mem_gb * 1024**3)  # Use 70% of memory
        
        gpu_config = {
            'is_high_memory': is_high_memory,
            'total_memory': total_mem_gb,
            'optimal_batch_size': optimal_batch_size,
            'pool_size': pool_size,
            'enable_parallel': is_high_memory,
            'cudnn_available': cudnn_available,
            'cudnn_version': cp.cuda.cudnn.getVersion() if cudnn_available else None
        }
        
        # Configure memory pool
        mempool = cp.cuda.MemoryPool()
        cp.cuda.set_allocator(mempool.malloc)
        
        # Set cudnn default config if available
        if cudnn_available:
            cp.cuda.cudnn.set_default_handle()
        
        print(f"\nGPU Configuration:")
        print(f"Total Memory: {total_mem_gb:.1f}GB")
        print(f"Mode: {'High Memory' if is_high_memory else 'Low Memory'}")
        print(f"Optimal Batch Size: {optimal_batch_size}")
        print(f"cuDNN Available: {cudnn_available}")
        if cudnn_available:
            print(f"cuDNN Version: {gpu_config['cudnn_version']}")
        
        return True, gpu_config

    except Exception as e:
        print(f"Error configuring GPU: {e}")
        return False, None

def to_gpu(x):
    """Transfer numpy array to GPU if available"""
    if GPU_AVAILABLE and isinstance(x, np.ndarray):
        return cp.asarray(x)
    return x

def to_cpu(x):
    """Transfer CuPy array to CPU if needed"""
    if GPU_AVAILABLE and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x

def clear_gpu_memory(threshold_mb=100):
    """More aggressive memory cleanup"""
    if GPU_AVAILABLE:
        pool = cp.get_default_memory_pool()
        used_mb = pool.used_bytes() / (1024 * 1024)
        if used_mb > threshold_mb:
            pool.free_all_blocks()
            cp.cuda.Device().synchronize()
