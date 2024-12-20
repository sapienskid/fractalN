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
    """Configure GPU for optimal performance in Colab"""
    if not GPU_AVAILABLE:
        return False

    try:
        # Set larger memory pool for Colab's 15GB GPU
        pool_size = 12 * 1024 * 1024 * 1024  # 12GB pool
        cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
        cp.cuda.set_pinned_memory_allocator(cp.cuda.PinnedMemoryPool().malloc)
        
        # Enable unified memory access
        device = cp.cuda.Device(0)
        device.use()
        
        # Set compute mode to maximize throughput
        cp.cuda.runtime.setDeviceFlags(cp.cuda.runtime.deviceScheduleAuto)
        
        # Print GPU configuration
        mem_info = device.mem_info
        print(f"\nGPU Memory Configuration:")
        print(f"Total: {mem_info[1]/1024**3:.1f}GB")
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

def clear_gpu_memory(threshold_mb=1000):
    """Smart GPU memory cleanup"""
    if GPU_AVAILABLE:
        pool = cp.get_default_memory_pool()
        used_mb = pool.used_bytes() / (1024 * 1024)
        if used_mb > threshold_mb:
            pool.free_all_blocks()
