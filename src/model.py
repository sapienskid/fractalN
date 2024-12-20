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
        self.batch_size = 8
        self.memory_cleanup_counter = 0

    def forward(self, x, training=False):
        try:
            x = to_gpu(x)
            
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
            return to_cpu(predictions)
            
        except Exception as e:
            print(f"\nError in forward pass: {str(e)}")
            raise e

    def train_step(self, x, y):
        try:
            # Clear GPU memory periodically
            if self.memory_cleanup_counter % 10 == 0:
                clear_gpu_memory()
            self.memory_cleanup_counter += 1
            
            # Ensure inputs are on correct device
            x = to_gpu(x)
            y = to_gpu(y)
            
            # Forward pass with proper type conversion
            predictions = self.forward(x, training=True)
            predictions = to_gpu(predictions)  # Move back to GPU for loss calculation
            
            # Calculate loss and convert to CPU float
            loss = float(to_cpu(cross_entropy_loss(predictions, y)))
            
            # Backward pass
            grad = cross_entropy_gradient(predictions, y)
            for layer in reversed(self.layers):
                grad = layer.backward(grad)
            
            # Return results ensuring they're on CPU
            return loss, to_cpu(predictions)
            
        except Exception as e:
            print(f"\nError in train_step: {str(e)}")
            raise e
    
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

