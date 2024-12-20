import numpy as np
import cupy as cp
import os
from gpu_utils import GPU_AVAILABLE, to_gpu, to_cpu, clear_gpu_memory
from exceptions import (
    ValidationError, ShapeError, GPUError, 
    LayerError, MemoryError, GPUMemoryTracker,
    validate_conv_params,
    validate_shapes_for_matmul,
    check_gpu_memory,
    validate_input
)

class BatchNormLayer:
    def __init__(self, num_features):
        self.use_gpu = GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.num_features = num_features
        self.gamma = to_gpu(np.ones(num_features))
        self.beta = to_gpu(np.zeros(num_features))
        self.eps = 1e-5
        self.running_mean = None
        self.running_var = None
        
    def forward(self, x, training=True):
        x = to_gpu(x)
        self.input_shape = x.shape
        
        if len(x.shape) == 4:
            N, C, H, W = x.shape
            x_reshaped = x.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            x_reshaped = x

        if self.running_mean is None:
            self.running_mean = to_gpu(np.zeros(x_reshaped.shape[1]))
            self.running_var = to_gpu(np.ones(x_reshaped.shape[1]))
            
        if training:
            mean = self.xp.mean(x_reshaped, axis=0)
            var = self.xp.var(x_reshaped, axis=0)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x_reshaped - mean) / self.xp.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        
        if len(self.input_shape) == 4:
            out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        
        if training:
            self.cache = (x_reshaped, x_norm, mean, var)
        
        return out

    def backward(self, dout):
        dout = to_gpu(dout)
        x, x_norm, mean, var = self.cache
        N = x.shape[0]
        
        if len(self.input_shape) == 4:
            N, C, H, W = self.input_shape
            dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)
        
        dx_norm = dout * self.gamma
        dvar = -0.5 * self.xp.sum(dx_norm * (x - mean) * (var + self.eps)**(-1.5), axis=0)
        dmean = -self.xp.sum(dx_norm / self.xp.sqrt(var + self.eps), axis=0) + dvar * -2 * self.xp.mean(x - mean, axis=0)
        dx = dx_norm / self.xp.sqrt(var + self.eps) + dvar * 2 * (x - mean) / N + dmean / N
        
        if len(self.input_shape) == 4:
            dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        
        return dx

    def save_params(self, path, layer_name):
        """Save batch normalization parameters"""
        try:
            np.save(os.path.join(path, f"{layer_name}_gamma.npy"), to_cpu(self.gamma))
            np.save(os.path.join(path, f"{layer_name}_beta.npy"), to_cpu(self.beta))
            if self.running_mean is not None:
                np.save(os.path.join(path, f"{layer_name}_running_mean.npy"), to_cpu(self.running_mean))
                np.save(os.path.join(path, f"{layer_name}_running_var.npy"), to_cpu(self.running_var))
        except Exception as e:
            print(f"Error saving BatchNormLayer parameters: {e}")
    
    def load_params(self, path, layer_name):
        """Load batch normalization parameters"""
        try:
            self.gamma = to_gpu(np.load(os.path.join(path, f"{layer_name}_gamma.npy")))
            self.beta = to_gpu(np.load(os.path.join(path, f"{layer_name}_beta.npy")))
            mean_path = os.path.join(path, f"{layer_name}_running_mean.npy")
            var_path = os.path.join(path, f"{layer_name}_running_var.npy")
            if os.path.exists(mean_path) and os.path.exists(var_path):
                self.running_mean = to_gpu(np.load(mean_path))
                self.running_var = to_gpu(np.load(var_path))
        except Exception as e:
            print(f"Error loading BatchNormLayer parameters: {e}")

class ConvLayer:
    def __init__(self, num_filters, filter_size, padding='valid', stride=1):
        # Validate parameters
        try:
            validate_conv_params(filter_size, stride, padding)
        except ValidationError as e:
            raise LayerError(
                f"Invalid ConvLayer parameters: {str(e)}", 
                layer_type='ConvLayer'
            )
            
        self.use_gpu = GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = None
        self.momentum = 0.9
        self.learning_rate = 0.01
        self.v = None
        self.initialized = False

        # Configure based on available GPU memory
        if self.use_gpu:
            try:
                device = cp.cuda.Device(0)
                total_mem_gb = device.mem_info[1] / (1024**3)
                self.is_high_memory = total_mem_gb >= 12
                self.chunk_size = 32 if total_mem_gb >= 12 else 4
            except:
                self.is_high_memory = False
                self.chunk_size = 4

        self.padding = padding
        self.stride_int = stride
        
        if padding == 'same':
            self.pad_h = (filter_size - 1) // 2
            self.pad_w = (filter_size - 1) // 2
        else:
            self.pad_h = 0
            self.pad_w = 0
        self.pad = ((0, 0), (0, 0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w))
        
        # Initialize dimensions
        self.in_channels = None
        self.input_height = None
        self.input_width = None

    def _initialize_params(self, input_shape):
        """Initialize filters based on input shape"""
        batch_size, in_channels, height, width = input_shape
        self.in_channels = in_channels
        self.input_height = height
        self.input_width = width
        
        if not self.initialized:
            scale = np.sqrt(2.0 / (in_channels * self.filter_size * self.filter_size))
            self.filters = to_gpu(np.random.randn(
                self.num_filters, 
                in_channels,
                self.filter_size, 
                self.filter_size
            ) * scale)
            self.v = to_gpu(np.zeros_like(self.filters))
            self.initialized = True

    def forward(self, inputs):
        """Forward pass using direct convolution"""
        try:
            inputs = to_gpu(inputs)
            if not self.initialized:
                self._initialize_params(inputs.shape)
            
            if self.filters is None:
                raise ValueError("Filters not properly initialized")
            
            self.inputs = inputs
            
            # Choose between chunked or direct processing based on batch size
            if inputs.shape[0] > self.chunk_size:
                return self._forward_chunked(inputs)
            return self._forward_direct(inputs)
            
        except Exception as e:
            print(f"Error in ConvLayer forward pass: {e}")
            raise

    def _forward_direct(self, inputs):
        """Direct convolution implementation"""
        batch_size, channels, height, width = inputs.shape
        output_height = ((height + 2 * self.pad_h - self.filter_size) // self.stride_int) + 1
        output_width = ((width + 2 * self.pad_w - self.filter_size) // self.stride_int) + 1
        
        # Apply padding
        if self.padding == 'same':
            padded_inputs = self.xp.pad(inputs, self.pad, 'constant')
        else:
            padded_inputs = inputs
            
        output = self.xp.zeros((batch_size, self.num_filters, output_height, output_width))
        
        # Optimized direct convolution
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.stride_int
                h_end = h_start + self.filter_size
                w_start = j * self.stride_int
                w_end = w_start + self.filter_size
                
                input_slice = padded_inputs[:, :, h_start:h_end, w_start:w_end]
                for k in range(self.num_filters):
                    output[:, k, i, j] = self.xp.sum(
                        input_slice * self.filters[k, :, :, :], 
                        axis=(1,2,3)
                    )
        
        return output

    def backward(self, dout):
        """Backward pass with direct computation"""
        try:
            dout = to_gpu(dout)
            dx = self.xp.zeros_like(self.inputs)
            dw = self.xp.zeros_like(self.filters)
            
            # Apply padding if needed
            if self.padding == 'same':
                padded_inputs = self.xp.pad(self.inputs, self.pad, 'constant')
            else:
                padded_inputs = self.inputs
            
            # Calculate gradients
            batch_size, _, out_h, out_w = dout.shape
            
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride_int
                    h_end = h_start + self.filter_size
                    w_start = j * self.stride_int
                    w_end = w_start + self.filter_size
                    
                    input_slice = padded_inputs[:, :, h_start:h_end, w_start:w_end]
                    
                    for k in range(self.num_filters):
                        dout_k = dout[:, k, i, j].reshape(-1, 1, 1, 1)
                        dw[k] += self.xp.sum(input_slice * dout_k, axis=0)
                        
                        if self.padding == 'same':
                            dx_slice = dx[:, :, h_start:h_end, w_start:w_end]
                            dx_slice += self.filters[k] * dout_k
                        else:
                            dx[:, :, h_start:h_end, w_start:w_end] += \
                                self.filters[k] * dout_k
            
            # Update weights
            if hasattr(self, 'optimizer'):
                self.filters = self.optimizer.update(self.filters, dw, f'conv_layer_{id(self)}')
            else:
                self.v = self.momentum * self.v - self.learning_rate * dw
                self.filters += self.v
            
            return dx
            
        except Exception as e:
            print(f"Error in backward pass: {e}")
            raise

    def _forward_chunked(self, inputs):
        """Process large batches in memory-efficient chunks"""
        outputs = []
        for i in range(0, inputs.shape[0], self.chunk_size):
            chunk = inputs[i:i + self.chunk_size]
            out = self._forward_direct(chunk)
            outputs.append(to_cpu(out))  # Store on CPU
            if self.use_gpu:
                clear_gpu_memory(100)
        return to_gpu(np.concatenate(outputs))

    def save_params(self, path, layer_name):
        """Save convolutional layer parameters"""
        try:
            if self.initialized:
                np.save(os.path.join(path, f"{layer_name}_filters.npy"), to_cpu(self.filters))
                np.save(os.path.join(path, f"{layer_name}_v.npy"), to_cpu(self.v))
        except Exception as e:
            print(f"Error saving ConvLayer parameters: {e}")
    
    def load_params(self, path, layer_name):
        """Load convolutional layer parameters"""
        try:
            self.filters = to_gpu(np.load(os.path.join(path, f"{layer_name}_filters.npy")))
            self.v = to_gpu(np.load(os.path.join(path, f"{layer_name}_v.npy")))
            self.initialized = True
        except Exception as e:
            print(f"Error loading ConvLayer parameters: {e}")

    def __del__(self):
        """Cleanup resources"""
        pass

class MaxPoolLayer:
    def __init__(self, pool_size=2):
        self.use_gpu = GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.pool_size = pool_size
        
    def forward(self, inputs):
        inputs = to_gpu(inputs)
        self.inputs = inputs
        self.batch_size, self.channels, self.height, self.width = inputs.shape
        output_height = self.height // self.pool_size
        output_width = self.width // self.pool_size
        
        output = self.xp.zeros((self.batch_size, self.channels, output_height, output_width))
        self.max_positions = self.xp.zeros_like(inputs)  # Store positions of max values
        
        for i in range(output_height):
            for j in range(output_width):
                h_start = i * self.pool_size
                h_end = h_start + self.pool_size
                w_start = j * self.pool_size
                w_end = w_start + self.pool_size
                input_slice = inputs[:, :, h_start:h_end, w_start:w_end]
                
                # Find max positions and values
                slice_shape = input_slice.shape
                reshaped_slice = input_slice.reshape(slice_shape[0], slice_shape[1], -1)
                max_indices = self.xp.argmax(reshaped_slice, axis=2)
                
                # Convert flat indices to 2D positions
                max_h = max_indices // self.pool_size
                max_w = max_indices % self.pool_size
                
                # Store max positions for backward pass
                for b in range(self.batch_size):
                    for c in range(self.channels):
                        h_max = h_start + max_h[b, c]
                        w_max = w_start + max_w[b, c]
                        self.max_positions[b, c, h_max, w_max] = 1
                
                output[:, :, i, j] = self.xp.max(input_slice, axis=(2, 3))
        
        return output
    
    def backward(self, dout):
        dout = to_gpu(dout)
        dx = self.xp.zeros_like(self.inputs)
        
        out_height = self.height // self.pool_size
        out_width = self.width // self.pool_size
        
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.pool_size
                h_end = h_start + self.pool_size
                w_start = j * self.pool_size
                w_end = w_start + self.pool_size
                
                # Get the region where max was found
                max_positions = self.max_positions[:, :, h_start:h_end, w_start:w_end]
                
                # Distribute gradient to max positions
                for b in range(self.batch_size):
                    for c in range(self.channels):
                        dx[b, c, h_start:h_end, w_start:w_end] += \
                            dout[b, c, i, j] * max_positions[b, c]
        
        return dx

    def save_params(self, path, layer_name):
        """Save MaxPool parameters"""
        try:
            config = {
                'pool_size': self.pool_size
            }
            np.save(os.path.join(path, f"{layer_name}_config.npy"), np.array([self.pool_size]))
            if hasattr(self, 'max_positions'):
                np.save(os.path.join(path, f"{layer_name}_last_positions.npy"), to_cpu(self.max_positions))
        except Exception as e:
            print(f"Error saving MaxPool parameters: {e}")
    
    def load_params(self, path, layer_name):
        """Load MaxPool parameters"""
        try:
            config_path = os.path.join(path, f"{layer_name}_config.npy")
            positions_path = os.path.join(path, f"{layer_name}_last_positions.npy")
            if os.path.exists(config_path):
                self.pool_size = int(np.load(config_path)[0])
            if os.path.exists(positions_path):
                self.max_positions = to_gpu(np.load(positions_path))
        except Exception as e:
            print(f"Error loading MaxPool parameters: {e}")

class FCLayer:
    def __init__(self, input_size, output_size):
        self.use_gpu = GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        scale = np.sqrt(2.0 / (input_size + output_size))
        self.weights = to_gpu(np.random.randn(input_size, output_size) * scale)
        self.bias = to_gpu(np.zeros(output_size))
        self.momentum = 0.9
        self.learning_rate = 0.01
        self.v_w = to_gpu(np.zeros_like(self.weights))
        self.v_b = to_gpu(np.zeros_like(self.bias))
    
    def forward(self, inputs):
        inputs = to_gpu(inputs)
        self.inputs = inputs
        return self.xp.dot(inputs, self.weights) + self.bias
        
    def backward(self, dout):
        dout = to_gpu(dout)
        dx = self.xp.dot(dout, self.weights.T)
        dw = self.xp.dot(self.inputs.T, dout)
        db = self.xp.sum(dout, axis=0)
        
        # Update with momentum
        self.v_w = self.momentum * self.v_w - self.learning_rate * dw
        self.v_b = self.momentum * self.v_b - self.learning_rate * db
        
        self.weights += self.v_w
        self.bias += self.v_b
        
        return dx

    def save_params(self, path, layer_name):
        """Save layer parameters to disk"""
        try:
            np.save(os.path.join(path, f"{layer_name}_weights.npy"), to_cpu(self.weights))
            np.save(os.path.join(path, f"{layer_name}_bias.npy"), to_cpu(self.bias))
            np.save(os.path.join(path, f"{layer_name}_v_w.npy"), to_cpu(self.v_w))
            np.save(os.path.join(path, f"{layer_name}_v_b.npy"), to_cpu(self.v_b))
        except Exception as e:
            print(f"Error saving FCLayer parameters: {e}")
    
    def load_params(self, path, layer_name):
        """Load layer parameters from disk"""
        try:
            self.weights = to_gpu(np.load(os.path.join(path, f"{layer_name}_weights.npy")))
            self.bias = to_gpu(np.load(os.path.join(path, f"{layer_name}_bias.npy")))
            self.v_w = to_gpu(np.load(os.path.join(path, f"{layer_name}_v_w.npy")))
            self.v_b = to_gpu(np.load(os.path.join(path, f"{layer_name}_v_b.npy")))
        except Exception as e:
            print(f"Error loading FCLayer parameters: {e}")

class ReLU:
    def __init__(self):
        self.use_gpu = GPU_AVAILABLE
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
        """Save ReLU parameters"""
        # No parameters to save for ReLU
        try:
            # Save activation state for debugging
            if hasattr(self, 'input'):
                np.save(os.path.join(path, f"{layer_name}_last_input.npy"), to_cpu(self.input))
        except Exception as e:
            print(f"Error saving ReLU state: {e}")
    
    def load_params(self, path, layer_name):
        """Load ReLU parameters"""
        # No parameters to load for ReLU
        try:
            input_path = os.path.join(path, f"{layer_name}_last_input.npy")
            if os.path.exists(input_path):
                self.input = to_gpu(np.load(input_path))
        except Exception as e:
            print(f"Error loading ReLU state: {e}")

class Dropout:
    def __init__(self, p=0.5):
        self.use_gpu = GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np
        self.p = p
        
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
        """Save Dropout parameters"""
        try:
            config = {
                'p': self.p,
                'use_gpu': self.use_gpu
            }
            if hasattr(self, 'mask'):
                np.save(os.path.join(path, f"{layer_name}_last_mask.npy"), to_cpu(self.mask))
            np.save(os.path.join(path, f"{layer_name}_config.npy"), np.array([self.p]))
        except Exception as e:
            print(f"Error saving Dropout parameters: {e}")
    
    def load_params(self, path, layer_name):
        """Load Dropout parameters"""
        try:
            config_path = os.path.join(path, f"{layer_name}_config.npy")
            mask_path = os.path.join(path, f"{layer_name}_last_mask.npy")
            if os.path.exists(config_path):
                self.p = float(np.load(config_path)[0])
            if os.path.exists(mask_path):
                self.mask = to_gpu(np.load(mask_path))
        except Exception as e:
            print(f"Error loading Dropout parameters: {e}")

class Flatten:
    def __init__(self):
        self.use_gpu = hasattr(cp, 'cuda') and cp.cuda.is_available()
        self.xp = cp if self.use_gpu else np
        
    def forward(self, x):
        x = self.to_device(x)
        self.input_shape = x.shape
        # Ensure proper flattening of NCHW format
        return x.reshape(x.shape[0], -1)  # Flatten all dimensions after batch
    
    def backward(self, grad):
        return grad.reshape(self.input_shape)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    """Apply softmax ensuring consistent array types"""
    xp = cp if GPU_AVAILABLE else np
    x = to_gpu(x)
    x = x - xp.max(x, axis=1, keepdims=True)  # For numerical stability
    exp_x = xp.exp(x)
    out = exp_x / xp.sum(exp_x, axis=1, keepdims=True)
    return to_cpu(out)

# Add utility function for array type checking
def get_array_module(x):
    """Get appropriate array module (numpy or cupy) for input array"""
    if isinstance(x, cp.ndarray):
        return cp
    return np

def save_model(model, path="model_weights"):
    """Save model parameters to disk"""
    if not os.path.exists(path):
        os.makedirs(path)
    
    for i, layer in enumerate(model.layers):
        layer.save_params(path, f"layer_{i}")

def load_model(model, path="model_weights"):
    """Load model parameters from disk"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model path {path} not found")
    
    for i, layer in enumerate(model.layers):
        layer.load_params(path, f"layer_{i}")
