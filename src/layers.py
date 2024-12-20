import numpy as np
import cupy as cp
import os
from gpu_utils import GPU_AVAILABLE, to_gpu, to_cpu, clear_gpu_memory
import cupy.cuda.cudnn as cudnn
from exceptions import (
    ValidationError, ShapeError, GPUError, 
    LayerError, MemoryError, GPUMemoryTracker,
    validate_conv_params,  # Add this import
    validate_shapes_for_matmul,  # Add this if you need matrix multiplication validation
    check_gpu_memory,  # Add this if you need memory checks
    validate_input  # Add this if you need input validation
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
        self.use_fft = True  # Enable FFT by default for 15GB GPU
        self.parallel_channels = True  # Enable parallel channel processing
        self.chunk_size = 32  # Larger chunks for more memory
        self.min_fft_size = 32  # Minimum size for using FFT
        self.memory_cleanup_counter = 0

        self.padding = padding  # Store the padding type
        self.stride_int = stride  # Keep original integer stride
        self.stride = (stride, stride)  # Tuple for operations
        
        if padding == 'same':
            self.pad_h = (filter_size - 1) // 2
            self.pad_w = (filter_size - 1) // 2
        else:
            self.pad_h = 0
            self.pad_w = 0
        self.pad = ((0, 0), (0, 0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w))

        # Check GPU memory capacity
        self.is_high_memory = False
        if self.use_gpu:
            try:
                device = cp.cuda.Device(0)
                total_mem_gb = device.mem_info[1] / (1024**3)
                self.is_high_memory = total_mem_gb >= 12
                
                # Configure based on memory
                self.use_fft = self.is_high_memory
                self.parallel_channels = self.is_high_memory
                self.chunk_size = 32 if self.is_high_memory else 2
            except:
                pass

        self.use_cudnn = False  # Disable cuDNN by default
        
        # Initialize filters with zeros to prevent None
        self.filters = None
        self.initialized = False
        
        # Add input dimensions
        self.in_channels = None
        self.input_height = None
        self.input_width = None

        # Initialize cuDNN support
        self.use_cudnn = False
        if GPU_AVAILABLE and hasattr(cp.cuda, 'cudnn') and cp.cuda.cudnn.available:
            try:
                self.use_cudnn = True
                # Remove the get_default_handle call as it's not needed
                self.pad = (self.pad_h, self.pad_h, self.pad_w, self.pad_w)  # HWHW padding
                self.stride = (self.stride, self.stride)
                self.dilation = (1, 1)
            except Exception as e:
                print(f"cuDNN initialization warning: {e}")
                self.use_cudnn = False
    
    def _initialize_params(self, input_shape):
        """Initialize filters based on input shape"""
        batch_size, in_channels, height, width = input_shape
        self.in_channels = in_channels
        self.input_height = height
        self.input_width = width
        
        # Only initialize if not already done
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

    def _init_cudnn_descriptors(self, x_shape):
        """Initialize cuDNN descriptors"""
        if not self.use_cudnn:
            return
            
        try:
            N, C, H, W = x_shape
            
            # Create tensor descriptors
            self.x_desc = cp.cuda.cudnn.create_tensor_descriptor()
            self.w_desc = cp.cuda.cudnn.create_filter_descriptor()
            self.conv_desc = cp.cuda.cudnn.create_convolution_descriptor()
            
            # Set descriptor configurations
            cp.cuda.cudnn.set_tensor4d_descriptor(
                self.x_desc,
                cp.cuda.cudnn.CUDNN_DATA_FLOAT,
                N, C, H, W
            )
            
            cp.cuda.cudnn.set_filter4d_descriptor(
                self.w_desc,
                cp.cuda.cudnn.CUDNN_DATA_FLOAT,
                self.num_filters, C,
                self.filter_size, self.filter_size
            )
            
            cp.cuda.cudnn.set_convolution2d_descriptor(
                self.conv_desc,
                self.pad_h, self.pad_w,
                self.stride, self.stride,
                1, 1,  # dilation
                cp.cuda.cudnn.CUDNN_CROSS_CORRELATION,
                cp.cuda.cudnn.CUDNN_DATA_FLOAT
            )
            
        except Exception as e:
            print(f"Error initializing cuDNN descriptors: {e}")
            self.use_cudnn = False

    def forward(self, inputs):
        try:
            inputs = to_gpu(inputs)
            # Initialize parameters if needed
            if not self.initialized:
                self._initialize_params(inputs.shape)
            
            # Verify filter initialization
            if self.filters is None:
                raise ValueError("Filters not properly initialized")
                
            # Try cuDNN first if available
            if self.use_cudnn:
                try:
                    return self._forward_cudnn(inputs)
                except Exception as e:
                    print(f"cuDNN forward failed: {e}, falling back to direct")
            
            # Use direct convolution instead of cuDNN
            self.inputs = inputs
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

    def _forward_implementation(self, inputs):
        # Validate input shape
        if len(inputs.shape) != 4:
            raise ShapeError(
                "Invalid input shape",
                expected_shape=(None, self.in_channels, None, None),
                actual_shape=inputs.shape
            )
            
        inputs = to_gpu(inputs)
        self.inputs = inputs
        
        if not self.initialized:
            self._initialize_params(inputs.shape)
            
        # Validate input shape
        if len(inputs.shape) != 4:
            raise ValueError(f"Expected 4D input tensor, got shape {inputs.shape}")
            
        # Choose convolution method
        try:
            if self.is_high_memory and self.use_gpu:
                return self._forward_parallel(inputs)
            elif inputs.shape[0] > self.chunk_size and self.use_gpu:
                return self._forward_chunked(inputs)
            return self._forward_direct(inputs)
        except Exception as e:
            print(f"Forward pass failed: {e}")
            return self._forward_direct(inputs)

    def _forward_fft(self, inputs):
        """FFT-based convolution with fixed shape handling"""
        xp = self.xp
        batch_size, in_channels, height, width = inputs.shape
        
        # Calculate padding to ensure output shape is correct
        pad_h = self.filter_size - 1
        pad_w = self.filter_size - 1
        
        # Calculate FFT size to be power of 2 for efficiency
        fft_height = 2 ** np.ceil(np.log2(height + pad_h)).astype(int)
        fft_width = 2 ** np.ceil(np.log2(width + pad_w)).astype(int)
        
        # Pad input for FFT
        padded_inputs = xp.pad(
            inputs, 
            ((0,0), (0,0), 
             (0, fft_height - height), 
             (0, fft_width - width)),
            mode='constant'
        )
        
        # Pad filters to match FFT size
        padded_filters = xp.pad(
            self.filters,
            ((0,0), (0,0),
             (0, fft_height - self.filter_size),
             (0, fft_width - self.filter_size)),
            mode='constant'
        )
        
        # Calculate output dimensions
        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1
        output = xp.zeros((batch_size, self.num_filters, out_height, out_width), dtype=inputs.dtype)
        
        try:
            for i in range(batch_size):
                for c in range(in_channels):
                    # Compute FFT of input channel
                    fft_input = xp.fft.rfft2(padded_inputs[i, c])
                    
                    for f in range(self.num_filters):
                        # Compute FFT of filter
                        fft_filter = xp.fft.rfft2(padded_filters[f, c])
                        
                        # Multiply in frequency domain
                        fft_product = fft_input * fft_filter
                        
                        # Inverse FFT and accumulate
                        if c == 0:
                            output[i, f] = xp.fft.irfft2(fft_product)[:out_height, :out_width]
                        else:
                            output[i, f] += xp.fft.irfft2(fft_product)[:out_height, :out_width]
            
            return output
            
        except Exception as e:
            print(f"FFT convolution failed: {e}, falling back to direct convolution")
            # Clear GPU memory before fallback
            if self.use_gpu:
                cp.get_default_memory_pool().free_all_blocks()
            return self._forward_direct(inputs)

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

    def _forward_parallel(self, inputs):
        """Optimized parallel forward pass for high memory GPUs"""
        batch_size, in_channels, height, width = inputs.shape
        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1
        
        # Pre-allocate output array
        output = self.xp.zeros((batch_size, self.num_filters, out_height, out_width))
        
        # Process channels in parallel using CuPy's optimized operations
        for k in range(self.num_filters):
            # Use vectorized operations for entire batch
            output[:, k] = self.xp.sum([
                self.xp.correlate2d(
                    inputs[:, c],
                    self.filters[k, c],
                    mode='valid'
                ) for c in range(in_channels)
            ], axis=0)
            
        return output

    def _forward_cudnn(self, inputs):
        """Forward pass using cuDNN"""
        try:
            # Use raw cuDNN convolution
            output = cp.zeros((
                inputs.shape[0],
                self.num_filters,
                ((inputs.shape[2] + 2 * self.pad_h - self.filter_size) // self.stride_int) + 1,
                ((inputs.shape[3] + 2 * self.pad_w - self.filter_size) // self.stride_int) + 1
            ), dtype=inputs.dtype)
            
            cp.cuda.cudnn.convolution_forward(
                inputs, self.filters, output,
                pad=(self.pad_h, self.pad_w),
                stride=self.stride,
                dilation=(1, 1),
                groups=1,
                auto_tune=True
            )
            return output
        except Exception as e:
            print(f"cuDNN forward failed: {e}, falling back to direct")
            return self._forward_direct(inputs)

    def backward(self, dout):
        dout = to_gpu(dout)
        
        try:
            dx, dw = self._backward_direct(dout)
                
            # Gradient clipping
            clip_threshold = 5.0
            dw = self.xp.clip(dw, -clip_threshold, clip_threshold)
            
            # Update with chosen optimizer
            if hasattr(self, 'optimizer'):
                self.filters = self.optimizer.update(self.filters, dw, f'conv_layer_{id(self)}')
            else:
                # Default momentum update
                self.v = self.momentum * self.v - self.learning_rate * dw
                self.filters += self.v
            
            # Memory cleanup
            if self.use_gpu and self.memory_cleanup_counter % 2 == 0:
                clear_gpu_memory(100)
            self.memory_cleanup_counter += 1
            
            return dx
            
        except Exception as e:
            print(f"Error in backward pass: {e}")
            raise

    def backward(self, dout):
        dout = to_gpu(dout)
        dx = self.xp.zeros_like(self.inputs)
        dw = self.xp.zeros_like(self.filters)
        
        for i in range(dout.shape[2]):
            for j in range(dout.shape[3]):
                input_slice = self.inputs[:, :, i:i+self.filter_size, j:j+self.filter_size]
                for k in range(self.num_filters):
                    dw[k] += self.xp.sum(input_slice * dout[:, k, i, j][:, None, None, None], axis=0)
                    dx[:, :, i:i+self.filter_size, j:j+self.filter_size] += \
                        self.filters[k] * dout[:, k, i, j][:, None, None, None]
        
        # Update with momentum
        self.v = self.momentum * self.v - self.learning_rate * dw
        self.filters += self.v
        
        # Add memory cleanup
        if self.use_gpu and self.memory_cleanup_counter % 2 == 0:
            clear_gpu_memory(100)
        self.memory_cleanup_counter += 1
        
        return dx

    def _backward_direct(self, dout):
        """Direct backward pass implementation"""
        dx = self.xp.zeros_like(self.inputs)
        dw = self.xp.zeros_like(self.filters)
        
        # Add gradient checking in debug mode
        if hasattr(self, 'debug_mode') and self.debug_mode:
            self._check_gradients(dout)
        
        # Implement gradient accumulation for large batches
        if dout.shape[0] > self.chunk_size:
            return self._backward_chunked(dout)
            
        # ... rest of backward implementation ...

    def _backward_cudnn(self, dout):
        """Backward pass using cuDNN"""
        try:
            dx = cp.zeros_like(self.inputs)
            dw = cp.zeros_like(self.filters)
            
            # Compute gradients using raw cuDNN
            cp.cuda.cudnn.convolution_backward_data(
                self.filters, dout, dx,
                pad=(self.pad_h, self.pad_w),
                stride=self.stride,
                dilation=(1, 1),
                groups=1,
                auto_tune=True
            )
            
            cp.cuda.cudnn.convolution_backward_filter(
                self.inputs, dout, dw,
                pad=(self.pad_h, self.pad_w),
                stride=self.stride,
                dilation=(1, 1),
                groups=1,
                auto_tune=True
            )
            
            return dx, dw
        except Exception as e:
            print(f"cuDNN backward failed: {e}, falling back to direct")
            return self._backward_direct(dout)

    def _check_gradients(self, dout):
        """Gradient checking using finite differences"""
        epsilon = 1e-7
        numerical_gradients = self.xp.zeros_like(self.filters)
        
        # Compute numerical gradients for a subset of weights
        for i in range(min(self.num_filters, 2)):
            for j in range(min(self.in_channels, 2)):
                original = self.filters[i,j,0,0].copy()
                
                # f(x + h)
                self.filters[i,j,0,0] = original + epsilon
                out_plus = self._forward_direct(self.inputs)
                cost_plus = self.xp.sum(out_plus * dout)
                
                # f(x - h)
                self.filters[i,j,0,0] = original - epsilon
                out_minus = self._forward_direct(self.inputs)
                cost_minus = self.xp.sum(out_minus * dout)
                
                # (f(x + h) - f(x - h)) / 2h
                numerical_gradients[i,j,0,0] = (cost_plus - cost_minus) / (2 * epsilon)
                self.filters[i,j,0,0] = original
        
        # Compare with analytical gradients
        analytical_gradients = self._compute_weight_gradients(dout)
        diff = self.xp.abs(numerical_gradients - analytical_gradients)
        
        if self.xp.max(diff) > 1e-5:
            print(f"Gradient check failed! Max difference: {self.xp.max(diff)}")

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
        """Cleanup cuDNN resources"""
        if hasattr(self, 'use_cudnn') and self.use_cudnn:
            for desc in [self.x_desc, self.w_desc, self.y_desc, self.conv_desc]:
                if desc is not None:
                    cudnn.destroy_tensor_descriptor(desc)
            if hasattr(self, 'cudnn_handle'):
                cudnn.destroy(self.cudnn_handle)

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
        # Ensure proper flattening
        return x.reshape(x.shape[0], -1)
    
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
