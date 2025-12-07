import numpy as np

class Optimizer:
    """
    Base class for optimizers.
    """
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
    
    def update(self, layer):
        """
        Update the parameters of a layer.
        
        Parameters:
        layer (Layer): The layer whose parameters need to be updated.
        """
        raise NotImplementedError("Update method must be implemented in subclass")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer with momentum option.
    """
    def __init__(self, learning_rate=0.001, momentum=0.0):
        """
        Initialize SGD optimizer.
        
        Parameters:
        learning_rate (float): Learning rate.
        momentum (float): Momentum coefficient, between 0 and 1.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, layer):
        """
        Update the parameters of a dense layer using SGD with momentum.
        
        Parameters:
        layer (Dense): The dense layer whose parameters need to be updated.
        """
        # Skip if layer has no weights (e.g., activation layers)
        if not hasattr(layer, 'weights'):
            return
        
        # Initialize velocities if not exists
        if id(layer) not in self.velocities:
            self.velocities[id(layer)] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
        
        # Update weights with momentum
        weights_delta = self.momentum * self.velocities[id(layer)]['weights'] + self.learning_rate * layer.weights_gradient
        self.velocities[id(layer)]['weights'] = weights_delta
        layer.weights -= weights_delta
        
        # Update biases with momentum
        biases_delta = self.momentum * self.velocities[id(layer)]['biases'] + self.learning_rate * layer.biases_gradient
        self.velocities[id(layer)]['biases'] = biases_delta
        layer.biases -= biases_delta


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer.
        
        Parameters:
        learning_rate (float): Learning rate.
        beta1 (float): Exponential decay rate for the first moment estimates.
        beta2 (float): Exponential decay rate for the second moment estimates.
        epsilon (float): Small constant to prevent division by zero.
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    
    def update(self, layer):
        """
        Update the parameters of a dense layer using Adam optimization.
        
        Parameters:
        layer (Dense): The dense layer whose parameters need to be updated.
        """
        # Skip if layer has no weights (e.g., activation layers)
        if not hasattr(layer, 'weights'):
            return
        
        # Increment time step
        self.t += 1
        
        # Initialize m and v if not exists
        if id(layer) not in self.m:
            self.m[id(layer)] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
            self.v[id(layer)] = {
                'weights': np.zeros_like(layer.weights),
                'biases': np.zeros_like(layer.biases)
            }
        
        # Update biased first moment estimate for weights
        self.m[id(layer)]['weights'] = self.beta1 * self.m[id(layer)]['weights'] + (1 - self.beta1) * layer.weights_gradient
        # Update biased second raw moment estimate for weights
        self.v[id(layer)]['weights'] = self.beta2 * self.v[id(layer)]['weights'] + (1 - self.beta2) * np.square(layer.weights_gradient)
        
        # Compute bias-corrected first moment estimate for weights
        m_hat_weights = self.m[id(layer)]['weights'] / (1 - np.power(self.beta1, self.t))
        # Compute bias-corrected second raw moment estimate for weights
        v_hat_weights = self.v[id(layer)]['weights'] / (1 - np.power(self.beta2, self.t))
        
        # Update weights
        layer.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        
        # Update biased first moment estimate for biases
        self.m[id(layer)]['biases'] = self.beta1 * self.m[id(layer)]['biases'] + (1 - self.beta1) * layer.biases_gradient
        # Update biased second raw moment estimate for biases
        self.v[id(layer)]['biases'] = self.beta2 * self.v[id(layer)]['biases'] + (1 - self.beta2) * np.square(layer.biases_gradient)
        
        # Compute bias-corrected first moment estimate for biases
        m_hat_biases = self.m[id(layer)]['biases'] / (1 - np.power(self.beta1, self.t))
        # Compute bias-corrected second raw moment estimate for biases
        v_hat_biases = self.v[id(layer)]['biases'] / (1 - np.power(self.beta2, self.t))
        
        # Update biases
        layer.biases -= self.learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)
