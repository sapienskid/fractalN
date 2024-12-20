import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = hasattr(cp, 'cuda') and cp.cuda.is_available()
except ImportError:
    cp = None
    GPU_AVAILABLE = False
from layers import ConvLayer, MaxPoolLayer, FCLayer, relu, softmax, BatchNormLayer
import os
from gpu_utils import (
    GPU_AVAILABLE, 
    to_gpu, 
    to_cpu, 
    clear_gpu_memory, 
    configure_gpu,
    is_high_memory_gpu
)

class ReLU:
    def __init__(self):
        self.use_gpu = hasattr(cp, 'cuda') and cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np
        
    def to_device(self, x):
        if self.use_gpu and isinstance(x, np.ndarray):
            return cp.asarray(x)
        elif not self.use_gpu and isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return x
        
    def forward(self, x):
        x = self.to_device(x)
        self.input = x
        return self.xp.maximum(0, x)
    
    def backward(self, grad):
        grad = self.to_device(grad)
        return grad * (self.input > 0)
    
    def save_params(self, path, layer_name):
        pass
    
    def load_params(self, path, layer_name):
        pass

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.use_gpu = hasattr(cp, 'cuda') and cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np
        
    def to_device(self, x):
        if self.use_gpu and isinstance(x, np.ndarray):
            return cp.asarray(x)
        elif not self.use_gpu and isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return x
        
    def forward(self, x, training=False):
        x = self.to_device(x)
        if training:
            self.mask = self.xp.random.binomial(1, 1-self.p, size=x.shape) / (1-self.p)
            return x * self.mask
        return x
    
    def backward(self, grad):
        grad = self.to_device(grad)
        return grad * self.mask
    
    def save_params(self, path, layer_name):
        pass
    
    def load_params(self, path, layer_name):
        pass

def cross_entropy_loss(predictions, targets):
    """Calculate cross entropy loss ensuring consistent array types and shapes"""
    xp = cp if GPU_AVAILABLE else np
    predictions = to_gpu(predictions)
    targets = to_gpu(targets)
    
    # Ensure predictions have correct shape (batch_size, num_classes)
    if len(predictions.shape) > 2:
        predictions = predictions.reshape(predictions.shape[0], -1)
    
    # Ensure targets have correct shape (batch_size, num_classes)
    if len(targets.shape) > 2:
        targets = targets.reshape(targets.shape[0], -1)
        
    epsilon = 1e-7
    predictions = xp.clip(predictions, epsilon, 1-epsilon)
    return float(to_cpu(-xp.mean(xp.sum(targets * xp.log(predictions), axis=1))))

def cross_entropy_gradient(predictions, targets):
    """Calculate gradient ensuring consistent array types"""
    xp = cp if GPU_AVAILABLE else np
    predictions = to_gpu(predictions)
    targets = to_gpu(targets)
    
    epsilon = 1e-7
    predictions = xp.clip(predictions, epsilon, 1-epsilon)
    return (predictions - targets) / predictions.shape[0]

class CNN:
    def __init__(self):
        self.use_gpu = GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        # Initialize GPU configuration
        self.gpu_config = self._init_gpu_config()
        
        # Initialize basic parameters
        input_size = 224  # Input image size
        c1_out = ((input_size - 2) // 2)
        c2_out = ((c1_out - 2) // 2)
        c3_out = ((c2_out - 2) // 2)
        flattened_size = 128 * c3_out * c3_out

        # Use GPU configuration for optimization settings
        if self.gpu_config and self.use_gpu:
            self.batch_size = self.gpu_config['optimal_batch_size']
            self.enable_parallel = self.gpu_config['is_high_memory']
            self.enable_chunk_processing = not self.gpu_config['is_high_memory']  # Added this line
            self.memory_threshold = 2000 if self.gpu_config['is_high_memory'] else 500
            self.cleanup_frequency = 10 if self.gpu_config['is_high_memory'] else 2
        else:
            # Default settings for CPU or unknown GPU
            self.batch_size = 4
            self.enable_parallel = False
            self.enable_chunk_processing = True
            self.memory_threshold = 500
            self.cleanup_frequency = 2

        # Initialize memory counter
        self.memory_cleanup_counter = 0
        
        # Initialize layers
        self.layers = [
            ConvLayer(num_filters=32, filter_size=3),
            BatchNormLayer(32),
            ReLU(),
            MaxPoolLayer(2),
            
            ConvLayer(num_filters=64, filter_size=3),
            BatchNormLayer(64),
            ReLU(),
            MaxPoolLayer(2),
            
            ConvLayer(num_filters=128, filter_size=3),
            BatchNormLayer(128),
            ReLU(),
            MaxPoolLayer(2),
            
            Flatten(),  # This is crucial for reshaping
            FCLayer(flattened_size, 512),
            BatchNormLayer(512),
            ReLU(),
            Dropout(0.5),
            FCLayer(512, 2)  # Output layer with 2 classes
        ]

    def _init_gpu_config(self):
        """Initialize GPU configuration"""
        if not self.use_gpu:
            return {
                'is_high_memory': False,
                'optimal_batch_size': 4,
                'enable_parallel': False
            }
            
        try:
            _, gpu_config = configure_gpu()
            if gpu_config is None:
                raise ValueError("GPU configuration failed")
            return gpu_config
        except Exception as e:
            print(f"Warning: Could not configure GPU optimally: {e}")
            return {
                'is_high_memory': False,
                'optimal_batch_size': 4,
                'enable_parallel': False
            }

    def forward(self, x, training=False):
        try:
            x = to_gpu(x)
            
            # Ensure input has correct shape (batch_size, channels, height, width)
            if len(x.shape) != 4:
                if len(x.shape) == 3:
                    x = x.reshape(1, *x.shape)  # Add batch dimension
                else:
                    raise ValueError(f"Expected 4D input, got shape {x.shape}")
            
            # Use GPU-specific optimizations
            if self.gpu_config and self.gpu_config['is_high_memory']:
                return self._forward_parallel(x, training)
            elif self.use_gpu:
                return self._forward_in_chunks(x, training) # Changed from _forward_chunked
            return self._forward_cpu(x, training)
        except Exception as e:
            print(f"Error in forward pass: {e}")
            raise e

    def _forward_single_chunk(self, x, training=False):
        """Process a single chunk of data with shape tracking"""
        out = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (Dropout, BatchNormLayer)):
                out = layer.forward(out, training)
            else:
                out = layer.forward(out)
                
            # Debug shape after each layer
            print(f"Layer {i} ({layer.__class__.__name__}) output shape: {out.shape}")
        
        # Ensure final output has correct shape
        if len(out.shape) > 2:
            out = out.reshape(out.shape[0], -1)
        return out

    def _forward_in_chunks(self, x, training=False):
        """Memory-efficient forward pass with parallel processing"""
        if self.enable_parallel and self.use_gpu:
            return self._forward_parallel(x, training)
        outputs = []
        chunk_size = min(self.batch_size, 4)  # Use smaller chunks
        
        for i in range(0, len(x), chunk_size):
            if GPU_AVAILABLE:
                clear_gpu_memory(100)  # More aggressive cleanup
            
            chunk = x[i:i + chunk_size]
            out = self._forward_single_chunk(chunk, training)
            outputs.append(to_cpu(out))  # Store on CPU to save GPU memory
            
        return to_gpu(np.concatenate(outputs))
    
    def _forward_cpu(self, x, training=False):
        """CPU fallback forward pass"""
        return self._forward_single_chunk(x, training)

    def train_step(self, x, y):
        try:
            if self.enable_chunk_processing and x.shape[0] > self.batch_size:
                return self._train_step_chunked(x, y)
            
            # Clear GPU memory more frequently
            if self.memory_cleanup_counter % 2 == 0:
                clear_gpu_memory(100)
            self.memory_cleanup_counter += 1
            
            # Ensure x has correct shape
            if len(x.shape) != 4:
                raise ValueError(f"Expected input shape (batch_size, channels, height, width), got {x.shape}")
            
            # Ensure y has correct shape
            if len(y.shape) != 2:
                raise ValueError(f"Expected target shape (batch_size, num_classes), got {y.shape}")
            
            x = to_gpu(x)
            y = to_gpu(y)
            
            predictions = self.forward(x, training=True)
            
            # Ensure predictions have correct shape before loss calculation
            if predictions.shape[-1] != y.shape[-1]:
                predictions = predictions.reshape(y.shape[0], -1)
                if predictions.shape[-1] != y.shape[-1]:
                    raise ValueError(
                        f"Predictions shape mismatch. "
                        f"Got {predictions.shape}, expected shape with {y.shape[-1]} outputs"
                    )
            
            predictions = to_gpu(predictions)
            
            loss = float(to_cpu(cross_entropy_loss(predictions, y)))
            
            # Clear intermediate results
            if GPU_AVAILABLE:
                clear_gpu_memory(100)
            
            grad = cross_entropy_gradient(predictions, y)
            for layer in reversed(self.layers):
                grad = layer.backward(grad)
                if GPU_AVAILABLE and isinstance(layer, ConvLayer):
                    clear_gpu_memory(100)  # Clear after each conv layer
            
            return loss, to_cpu(predictions)
            
        except Exception as e:
            print(f"\nError in train_step: {str(e)}")
            print(f"Input shape: {x.shape}")
            print(f"Target shape: {y.shape}")
            print(f"Predictions shape: {predictions.shape if 'predictions' in locals() else 'N/A'}")
            raise e
    
    def _train_step_chunked(self, x, y):
        """Memory-efficient training step"""
        chunk_size = 4
        total_loss = 0
        all_predictions = []
        
        for i in range(0, len(x), chunk_size):
            if GPU_AVAILABLE:
                clear_gpu_memory(100)
            
            chunk_x = x[i:i + chunk_size]
            chunk_y = y[i:i + chunk_size]
            
            loss, predictions = self.train_step(chunk_x, chunk_y)
            total_loss += loss * len(chunk_x)
            all_predictions.append(predictions)
        
        avg_loss = total_loss / len(x)
        combined_predictions = np.concatenate(all_predictions)
        return avg_loss, combined_predictions
    
    def to_device(self, x):
        if self.use_gpu and isinstance(x, np.ndarray):
            return cp.asarray(x)
        elif not self.use_gpu and isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return x

    def save(self, path):
        # Creates directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Save parameters of each layer
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'save_params'):
                layer.save_params(path, f"layer_{i}")
    
    def load(self, path):
        # Load parameters for each layer
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'save_params'):
                layer.load_params(path, f"layer_{i}")

def softmax(x):
    """Apply softmax ensuring consistent array types"""
    xp = cp if GPU_AVAILABLE else np
    x = to_gpu(x)
    exp_x = xp.exp(x - xp.max(x, axis=1, keepdims=True))
    out = exp_x / xp.sum(exp_x, axis=1, keepdims=True)
    return to_cpu(out)

class Flatten:
    def __init__(self):
        self.use_gpu = hasattr(cp, 'cuda') and cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np
        
    def to_device(self, x):
        if self.use_gpu and isinstance(x, np.ndarray):
            return cp.asarray(x)
        elif not self.use_gpu and isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
        return x
        
    def forward(self, x):
        x = self.to_device(x)
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, grad):
        grad = self.to_device(grad)
        return grad.reshape(self.input_shape)

