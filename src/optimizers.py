import numpy as np
from gpu_utils import to_gpu, to_cpu, GPU_AVAILABLE

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.current_step = 0
    
    def step(self):
        self.current_step += 1

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0
        
    def update(self, params, grads, key):
        if key not in self.m:
            self.m[key] = np.zeros_like(params)
            self.v[key] = np.zeros_like(params)
            
        self.t += 1
        
        # Update biased first moment estimate
        self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads
        
        # Update biased second moment estimate
        self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads ** 2)
        
        # Bias correction
        m_hat = self.m[key] / (1 - self.beta1 ** self.t)
        v_hat = self.v[key] / (1 - self.beta2 ** self.t)
        
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class RMSprop:
    def __init__(self, learning_rate=0.01, decay_rate=0.99, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}
        
    def update(self, params, grads, key):
        if key not in self.cache:
            self.cache[key] = np.zeros_like(params)
            
        self.cache[key] = self.decay_rate * self.cache[key] + (1 - self.decay_rate) * (grads ** 2)
        return params - self.learning_rate * grads / (np.sqrt(self.cache[key]) + self.epsilon)

class LearningRateScheduler:
    def __init__(self, initial_lr):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        
    def step_decay(self, epoch, drop_rate=0.5, epochs_drop=10.0):
        """Step decay schedule"""
        self.current_lr = self.initial_lr * np.power(drop_rate, np.floor((1 + epoch) / epochs_drop))
        return self.current_lr
        
    def exp_decay(self, epoch, decay_rate=0.95):
        """Exponential decay schedule"""
        self.current_lr = self.initial_lr * np.exp(-decay_rate * epoch)
        return self.current_lr
        
    def cosine_decay(self, epoch, total_epochs):
        """Cosine annealing schedule"""
        self.current_lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        return self.current_lr

class AdaptiveMomentum:
    def __init__(self, base_momentum=0.9, adapt_factor=0.01):
        self.base_momentum = base_momentum
        self.adapt_factor = adapt_factor
        self.previous_grads = {}
        self.momentum_values = {}
        
    def get_momentum(self, key, current_grad):
        if key not in self.previous_grads:
            self.previous_grads[key] = current_grad
            self.momentum_values[key] = self.base_momentum
            return self.base_momentum
            
        # Compute angle between current and previous gradient
        cos_theta = self.xp.sum(current_grad * self.previous_grads[key]) / \
                   (self.xp.linalg.norm(current_grad) * self.xp.linalg.norm(self.previous_grads[key]) + 1e-7)
        
        # Adapt momentum based on gradient similarity
        self.momentum_values[key] = self.base_momentum * (1 + self.adapt_factor * cos_theta)
        self.previous_grads[key] = current_grad
        
        return self.momentum_values[key]

class AdaptiveLearningRate:
    def __init__(self, initial_lr=0.01, min_lr=1e-6, patience=3):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.patience = patience
        self.best_loss = float('inf')
        self.waiting = 0
        self.current_lr = initial_lr
        
    def should_reduce_lr(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.waiting = 0
            return False
        
        self.waiting += 1
        if self.waiting >= self.patience:
            self.current_lr = max(self.current_lr * 0.5, self.min_lr)
            self.waiting = 0
            return True
        return False
