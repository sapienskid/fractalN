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
from optimizers import Adam, RMSprop, LearningRateScheduler

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
    def __init__(self, optimizer='adam', lr=0.001):
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
        
        # Initialize optimizer
        if optimizer == 'adam':
            self.optimizer = Adam(learning_rate=lr)
        elif optimizer == 'rmsprop':
            self.optimizer = RMSprop(learning_rate=lr)
        
        # Initialize learning rate scheduler
        self.lr_scheduler = LearningRateScheduler(initial_lr=lr)
        
        # Initialize layers with correct shapes
        self.layers = [
            ConvLayer(num_filters=32, filter_size=3, padding='same', stride=1),  # 224x224 -> 224x224
            BatchNormLayer(32),
            ReLU(),
            MaxPoolLayer(2),  # 224x224 -> 112x112
            
            ConvLayer(num_filters=64, filter_size=3, padding='same', stride=1),  # 112x112 -> 112x112
            BatchNormLayer(64),
            ReLU(),
            MaxPoolLayer(2),  # 112x112 -> 56x56
            
            ConvLayer(num_filters=128, filter_size=3, padding='same', stride=1),  # 56x56 -> 56x56
            BatchNormLayer(128),
            ReLU(),
            MaxPoolLayer(2),  # 56x56 -> 28x28
            
            Flatten(),  # (batch_size, 128, 28, 28) -> (batch_size, 128*28*28)
            FCLayer(128 * 28 * 28, 512),  # Update input size to match flattened shape
            BatchNormLayer(512),
            ReLU(),
            Dropout(0.5),
            FCLayer(512, 2)
        ]
        
        # Assign optimizer to layers
        for layer in self.layers:
            if hasattr(layer, 'optimizer'):
                layer.optimizer = self.optimizer

        # Add gradient accumulation settings
        self.grad_accumulation_steps = 4  # Accumulate gradients over 4 steps
        self.accumulated_gradients = {}
        self.current_accumulation_step = 0

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
        """Main forward pass with error handling"""
        try:
            x = to_gpu(x)
            
            # Input shape validation
            if len(x.shape) != 4:
                raise ValueError(f"Expected 4D input (batch, channels, height, width), got shape {x.shape}")
            
            # Track shapes for debugging
            current_shape = x.shape
            for i, layer in enumerate(self.layers):
                try:
                    if isinstance(layer, (Dropout, BatchNormLayer)):
                        x = layer.forward(x, training)
                    else:
                        x = layer.forward(x)
                    
                    if x is None:
                        raise ValueError(f"Layer {i} ({layer.__class__.__name__}) produced None output")
                    
                    current_shape = x.shape
                    
                except Exception as e:
                    print(f"Error in layer {i} ({layer.__class__.__name__})")
                    print(f"Input shape: {current_shape}")
                    print(f"Error: {str(e)}")
                    raise
            
            return x
            
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            raise

    def _forward_parallel(self, x, training=False):
        """Parallel forward pass for high memory GPUs"""
        try:
            out = x
            for layer in self.layers:
                if isinstance(layer, (Dropout, BatchNormLayer)):
                    out = layer.forward(out, training)
                else:
                    out = layer.forward(out)
                
                # Clear intermediate results to save memory
                if GPU_AVAILABLE and isinstance(layer, ConvLayer):
                    clear_gpu_memory(100)
                    
            return out
            
        except Exception as e:
            print(f"Error in parallel forward pass: {e}")
            # Fallback to regular forward pass
            return self._forward_single_chunk(x, training)

    def _forward_in_chunks(self, x, training=False):
        """Process input in chunks to save memory"""
        chunk_size = self.batch_size
        outputs = []
        
        for i in range(0, len(x), chunk_size):
            chunk = x[i:i + chunk_size]
            out = self._forward_single_chunk(chunk, training)
            outputs.append(to_cpu(out))  # Store on CPU
            
            if GPU_AVAILABLE:
                clear_gpu_memory(100)
                
        return to_gpu(np.concatenate(outputs))

    def _forward_single_chunk(self, x, training=False):
        """Process a single chunk of data"""
        out = x
        for layer in self.layers:
            try:
                if isinstance(layer, (Dropout, BatchNormLayer)):
                    out = layer.forward(out, training)
                else:
                    out = layer.forward(out)
                    
                if out is None:
                    raise ValueError(f"Layer {layer.__class__.__name__} produced None output")
                    
            except Exception as e:
                print(f"Error in layer {layer.__class__.__name__}: {str(e)}")
                raise
                
        return out

    def train_step(self, x, y):
        try:
            # Split batch if too large
            if x.shape[0] > self.batch_size:
                return self._train_step_with_accumulation(x, y)
            
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

    def _train_step_with_accumulation(self, x, y):
        """Training step with gradient accumulation"""
        total_loss = 0
        predictions = []
        batch_size = len(x)
        mini_batch_size = batch_size // self.grad_accumulation_steps
        
        for i in range(0, batch_size, mini_batch_size):
            end_idx = min(i + mini_batch_size, batch_size)
            mini_batch_x = x[i:end_idx]
            mini_batch_y = y[i:end_idx]
            
            # Forward pass
            mini_batch_pred = self.forward(mini_batch_x, training=True)
            loss = float(to_cpu(cross_entropy_loss(mini_batch_pred, mini_batch_y)))
            total_loss += loss * len(mini_batch_x)
            predictions.append(to_cpu(mini_batch_pred))
            
            # Backward pass with scaled gradients
            grad = cross_entropy_gradient(mini_batch_pred, mini_batch_y)
            grad = grad / self.grad_accumulation_steps
            
            # Accumulate gradients
            if i == 0:
                self._init_gradients(grad)
            else:
                self._accumulate_gradients(grad)
            
            self.current_accumulation_step += 1
            
            # Update weights when accumulation is complete
            if self.current_accumulation_step >= self.grad_accumulation_steps:
                self._update_weights_with_accumulated_gradients()
                self.current_accumulation_step = 0
        
        # Return average loss and combined predictions
        avg_loss = total_loss / batch_size
        combined_predictions = np.concatenate(predictions)
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

