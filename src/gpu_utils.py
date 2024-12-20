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

def configure_gpu():
    """Configure GPU for high performance with 15GB memory"""
    if not GPU_AVAILABLE:
        return False

    try:
        device = cp.cuda.Device(0)
        total_mem_gb = device.mem_info[1] / (1024**3)
        
        if total_mem_gb >= 12:  # High memory GPU
            # Use 80% of available memory for pool
            pool_size = int(0.8 * total_mem_gb * 1024**3)
            # Enable high performance settings
            cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
            cp.cuda.cudnn.set_enabled(True)  # Enable cuDNN if available
            
            # Set stream preferences for parallel execution
            cp.cuda.Stream.null.use()
            cp.random.seed(None)
        else:
            # Configure smaller memory pool for 4GB GPU
            if total_mem_gb < 6:
                pool_size = 3 * 1024 * 1024 * 1024  # 3GB pool
            else:
                pool_size = 12 * 1024 * 1024 * 1024  # 12GB pool
            
            cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
            cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
        
        device.use()
        print(f"\nGPU Configuration (High Performance):")
        print(f"Total Memory: {total_mem_gb:.1f}GB")
        print(f"Pool Size: {pool_size/1024**3:.1f}GB")
        return True

    except Exception as e:
        print(f"Error configuring GPU: {e}")
        return False

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
