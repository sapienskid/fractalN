import numpy as np
try:
    import cupy as cp
    GPU_AVAILABLE = hasattr(cp, 'cuda') and cp.cuda.is_available()
except ImportError:
    cp = None
    GPU_AVAILABLE = False
from layers import ConvLayer, MaxPoolLayer, FCLayer, relu, softmax, BatchNormLayer
import os
from gpu_utils import GPU_AVAILABLE, to_gpu, to_cpu, clear_gpu_memory

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
    """Calculate cross entropy loss ensuring consistent array types"""
    xp = cp if GPU_AVAILABLE else np
    predictions = to_gpu(predictions)
    targets = to_gpu(targets)
    
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
        # Calculate sizes for proper layer connections
        input_size = 224  # Input image size
        c1_out = ((input_size - 2) // 2)  # After first conv and pool
        c2_out = ((c1_out - 2) // 2)      # After second conv and pool
        c3_out = ((c2_out - 2) // 2)      # After third conv and pool
        flattened_size = 128 * c3_out * c3_out  # For fully connected layer
        
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
            
            Flatten(),
            FCLayer(flattened_size, 512),  # Adjust size based on your input dimensions
            BatchNormLayer(512),
            ReLU(),
            Dropout(0.5),
            FCLayer(512, 2)
        ]
        
        # Initialize other parameters
        self.learning_rate = 0.001
        self.lr_decay = 0.95
        self.epoch_count = 0
        self.train_mode = True
        self.memory_cleanup_counter = 0
        self.enable_gpu_optimizations = True
        
        # Adjust batch size based on available GPU memory
        if GPU_AVAILABLE:
            device = cp.cuda.Device(0)
            total_mem_gb = device.mem_info[1] / (1024**3)
            
            if total_mem_gb >= 12:  # High memory GPU optimizations
                self.batch_size = 64  # Larger batch size
                self.enable_parallel = True
                self.memory_threshold = 2000  # Higher threshold (2GB)
                self.cleanup_frequency = 10   # Less frequent cleanup
                
                # Enable performance optimizations
                for layer in self.layers:
                    if isinstance(layer, ConvLayer):
                        layer.use_fft = True
                        layer.parallel_channels = True
                        layer.chunk_size = 32
            else:
                # Use smaller batch size for limited GPU memory
                self.batch_size = 4 if total_mem_gb < 6 else 32
                self.enable_chunk_processing = total_mem_gb < 6
        else:
            self.batch_size = 32
            self.enable_chunk_processing = False
        
        self.memory_threshold = 500  # MB (lower threshold for 4GB GPU)
        self.cleanup_frequency = 2    # More frequent cleanup

    def forward(self, x, training=False):
        try:
            x = to_gpu(x)
            if self.enable_gpu_optimizations:
                # Process in smaller chunks if needed
                if x.shape[0] > self.batch_size:
                    return self._forward_in_chunks(x, training)
            
            out = x
            total_layers = len(self.layers)
            for i, layer in enumerate(self.layers):
                # Get layer info for progress monitoring
                layer_name = layer.__class__.__name__
                layer_info = {
                    'name': layer_name,
                    'index': i + 1,
                    'total': total_layers,
                    'progress': (i + 1) / total_layers * 100
                }
                
                # Update monitor if available
                if hasattr(self, 'monitor'):
                    self.monitor.update_layer_info(layer_info)
                
                # Process layer
                if isinstance(layer, (Dropout, BatchNormLayer)):
                    out = layer.forward(out, training)
                else:
                    out = layer.forward(out)
            
            out = to_cpu(out)
            predictions = softmax(out)
            
            if self.memory_cleanup_counter % self.cleanup_frequency == 0:
                clear_gpu_memory(self.memory_threshold)
            
            return to_cpu(predictions)
            
        except Exception as e:
            print(f"\nError in forward pass: {str(e)}")
            raise e

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

    def _forward_parallel(self, x, training=False):
        """High performance forward pass for 15GB GPU"""
        with cp.cuda.Stream():
            outputs = []
            chunk_size = self.batch_size
            
            for i in range(0, len(x), chunk_size):
                chunk = x[i:i + chunk_size]
                out = self._forward_single_chunk(chunk, training)
                outputs.append(out)  # Keep on GPU
            
            return cp.concatenate(outputs)

    def train_step(self, x, y):
        try:
            if self.enable_chunk_processing and x.shape[0] > 4:
                return self._train_step_chunked(x, y)
            
            # Clear GPU memory more frequently
            if self.memory_cleanup_counter % 2 == 0:
                clear_gpu_memory(100)
            self.memory_cleanup_counter += 1
            
            x = to_gpu(x)
            y = to_gpu(y)
            
            predictions = self.forward(x, training=True)
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

