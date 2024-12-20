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
    """Configure CuPy GPU settings"""
    if not GPU_AVAILABLE:
        print(f"{Fore.YELLOW}No GPU available. Running on CPU{Style.RESET_ALL}")
        return False

    try:
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        device = cp.cuda.Device(0)
        device.use()
        
        gpu_info = get_gpu_info()
        if gpu_info:
            print(f"\n{Fore.GREEN}GPU Configuration:")
            print(f"├── Device: {gpu_info['name']}")
            print(f"├── Compute Capability: {gpu_info['compute_capability']}")
            print(f"├── Total Memory: {gpu_info['total_memory'] / (1024**2):.1f} MB")
            print(f"└── CUDA Version: {gpu_info['cuda_version']}{Style.RESET_ALL}")
            return True
        return False
        
    except Exception as e:
        print(f"{Fore.RED}Error configuring GPU: {e}{Style.RESET_ALL}")
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

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
