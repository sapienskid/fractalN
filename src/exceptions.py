import numpy as np
import traceback

class NNError(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details
        self.traceback = traceback.format_exc()
    
    def __str__(self):
        error_msg = super().__str__()
        if self.details:
            error_msg += f"\nDetails: {self.details}"
        return error_msg

class ValidationError(NNError):
    def __init__(self, message, invalid_value=None):
        details = f"Invalid value: {invalid_value}" if invalid_value is not None else None
        super().__init__(message, details)

class MemoryError(NNError):
    def __init__(self, message, required_memory=None, available_memory=None):
        details = {}
        if required_memory is not None:
            details['required_memory'] = f"{required_memory/1e9:.2f}GB"
        if available_memory is not None:
            details['available_memory'] = f"{available_memory/1e9:.2f}GB"
        super().__init__(message, details)

class ShapeError(NNError):
    def __init__(self, message, expected_shape=None, actual_shape=None):
        details = {
            'expected_shape': expected_shape,
            'actual_shape': actual_shape
        } if expected_shape and actual_shape else None
        super().__init__(message, details)

class GPUError(NNError):
    def __init__(self, message, gpu_info=None):
        details = f"GPU Info: {gpu_info}" if gpu_info else None
        super().__init__(message, details)

class OptimizationError(NNError):
    def __init__(self, message, optimizer_params=None):
        details = f"Optimizer parameters: {optimizer_params}" if optimizer_params else None
        super().__init__(message, details)

class LayerError(NNError):
    def __init__(self, message, layer_name=None, layer_type=None):
        details = {
            'layer_name': layer_name,
            'layer_type': layer_type
        } if layer_name or layer_type else None
        super().__init__(message, details)

class DataError(NNError):
    def __init__(self, message, data_shape=None, expected_shape=None):
        details = {
            'data_shape': data_shape,
            'expected_shape': expected_shape
        } if data_shape or expected_shape else None
        super().__init__(message, details)

def validate_input(x, expected_dims=4, min_size=1, dtype=None):
    """Validate input tensor dimensions, size, and type"""
    if x is None:
        raise ValidationError("Input tensor cannot be None")
        
    if not isinstance(x, (np.ndarray, np.generic)):
        raise ValidationError(f"Expected numpy array, got {type(x)}")
        
    if len(x.shape) != expected_dims:
        raise ShapeError(
            f"Expected {expected_dims}D input tensor, got shape {x.shape}"
        )
    
    if any(s < min_size for s in x.shape):
        raise ValidationError(
            f"All dimensions must be at least {min_size}, got shape {x.shape}"
        )
        
    if dtype and x.dtype != dtype:
        raise ValidationError(
            f"Expected dtype {dtype}, got {x.dtype}"
        )

def validate_conv_params(kernel_size, stride, padding):
    """Validate convolution layer parameters"""
    if not isinstance(kernel_size, (int, tuple)):
        raise ValidationError(
            f"Kernel size must be int or tuple, got {type(kernel_size)}"
        )
    
    if not isinstance(stride, (int, tuple)):
        raise ValidationError(
            f"Stride must be int or tuple, got {type(stride)}"
        )
        
    if padding not in ['valid', 'same']:
        raise ValidationError(
            f"Padding must be 'valid' or 'same', got {padding}"
        )

def validate_shapes_for_matmul(shape1, shape2):
    """Validate shapes for matrix multiplication"""
    if len(shape1) < 2 or len(shape2) < 2:
        raise ShapeError(
            f"Invalid shapes for matrix multiplication: {shape1}, {shape2}"
        )
    
    if shape1[-1] != shape2[-2]:
        raise ShapeError(
            f"Incompatible shapes for matrix multiplication: {shape1}, {shape2}"
        )

def check_gpu_memory(required_bytes):
    """Check if enough GPU memory is available"""
    try:
        import cupy as cp
        if hasattr(cp, 'cuda'):
            free_memory = cp.cuda.runtime.memGetInfo()[0]
            if free_memory < required_bytes:
                raise MemoryError(
                    f"Not enough GPU memory. Required: {required_bytes/1e9:.2f}GB, "
                    f"Available: {free_memory/1e9:.2f}GB",
                    required_memory=required_bytes,
                    available_memory=free_memory
                )
    except ImportError:
        raise GPUError("CUDA/CuPy not available")

def validate_optimizer_params(params):
    """Validate optimizer parameters"""
    required_params = ['learning_rate']
    for param in required_params:
        if param not in params:
            raise OptimizationError(f"Missing required parameter: {param}")
        
    if params['learning_rate'] <= 0:
        raise OptimizationError(
            f"Learning rate must be positive, got {params['learning_rate']}",
            optimizer_params=params
        )

def validate_batch_size(batch_size, dataset_size):
    """Validate batch size against dataset size"""
    if batch_size <= 0:
        raise ValidationError(f"Batch size must be positive, got {batch_size}")
        
    if batch_size > dataset_size:
        raise ValidationError(
            f"Batch size ({batch_size}) cannot be larger than dataset size ({dataset_size})"
        )

def validate_layer_input(layer_name, input_shape, expected_shape):
    """Validate layer input shape"""
    if input_shape != expected_shape:
        raise LayerError(
            f"Invalid input shape for {layer_name}. "
            f"Expected {expected_shape}, got {input_shape}",
            layer_name=layer_name
        )

def validate_data_format(data, labels=None):
    """Validate input data and labels"""
    if not isinstance(data, np.ndarray):
        raise DataError(f"Expected numpy array for data, got {type(data)}")
        
    if labels is not None:
        if not isinstance(labels, np.ndarray):
            raise DataError(f"Expected numpy array for labels, got {type(labels)}")
            
        if len(data) != len(labels):
            raise DataError(
                f"Data and labels must have same length. "
                f"Got {len(data)} and {len(labels)}"
            )

def check_nan_inf(tensor, tensor_name="tensor"):
    """Check for NaN or Inf values in tensor"""
    if np.isnan(tensor).any():
        raise ValidationError(f"NaN values detected in {tensor_name}")
    if np.isinf(tensor).any():
        raise ValidationError(f"Inf values detected in {tensor_name}")

class GPUMemoryTracker:
    """Enhanced context manager for tracking GPU memory usage"""
    def __init__(self, threshold_mb=100, raise_error=True):
        self.threshold_mb = threshold_mb
        self.raise_error = raise_error
        self.initial_memory = None
        self.peak_memory = 0
        self.memory_log = []
        
    def _log_memory(self, event):
        try:
            import cupy as cp
            if hasattr(cp, 'cuda'):
                current = cp.cuda.runtime.memGetInfo()[1]
                used = (self.initial_memory - current) / 1024**2
                self.peak_memory = max(self.peak_memory, used)
                self.memory_log.append({
                    'event': event,
                    'used_mb': used,
                    'peak_mb': self.peak_memory
                })
        except ImportError:
            pass
        
    def __enter__(self):
        try:
            import cupy as cp
            if hasattr(cp, 'cuda'):
                self.initial_memory = cp.cuda.runtime.memGetInfo()[1]
                self._log_memory('start')
        except ImportError:
            if self.raise_error:
                raise GPUError("CUDA/CuPy not available")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            import cupy as cp
            if hasattr(cp, 'cuda'):
                self._log_memory('end')
                memory_diff = self.peak_memory
                if memory_diff > self.threshold_mb:
                    message = f"Large memory usage detected: {memory_diff:.2f}MB"
                    cp.cuda.memory.gc()
                    if self.raise_error:
                        raise MemoryError(message, 
                                        required_memory=memory_diff * 1024**2,
                                        available_memory=self.initial_memory)
                    else:
                        print(f"Warning: {message}")
                        print("Memory log:", self.memory_log)
        except ImportError:
            pass
