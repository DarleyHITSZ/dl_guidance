import numpy as np
from .layers import Layer

class Activation(Layer):
    """
    Base class for activation functions.
    Activation layers are special layers that apply a non-linear transformation to their input.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, input_data):
        """
        Apply activation function to input.
        
        Parameters:
        input_data (numpy.ndarray): Input data to the activation function.
        
        Returns:
        numpy.ndarray: Activated output.
        """
        self.input = input_data
        return self.activation(input_data)
    
    def backward(self, output_gradient, learning_rate):
        """
        Compute gradient of loss with respect to input.
        
        Parameters:
        output_gradient (numpy.ndarray): Gradient of loss with respect to layer output.
        learning_rate (float): Learning rate (unused for activation layers).
        
        Returns:
        numpy.ndarray: Gradient of loss with respect to layer input.
        """
        return np.multiply(output_gradient, self.derivative(self.input))
    
    def activation(self, input_data):
        """
        The activation function.
        
        Parameters:
        input_data (numpy.ndarray): Input data.
        
        Returns:
        numpy.ndarray: Activated output.
        """
        raise NotImplementedError("Activation method must be implemented in subclass")
    
    def derivative(self, input_data):
        """
        The derivative of the activation function.
        
        Parameters:
        input_data (numpy.ndarray): Input data.
        
        Returns:
        numpy.ndarray: Derivative of the activation function at input_data.
        """
        raise NotImplementedError("Derivative method must be implemented in subclass")


class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU) activation function.
    """
    def activation(self, input_data):
        return np.maximum(0, input_data)
    
    def derivative(self, input_data):
        return (input_data > 0).astype(float)


class Sigmoid(Activation):
    """
    Sigmoid activation function.
    """
    def activation(self, input_data):
        return 1 / (1 + np.exp(-input_data))
    
    def derivative(self, input_data):
        sigmoid = self.activation(input_data)
        return sigmoid * (1 - sigmoid)


class Tanh(Activation):
    """
    Hyperbolic Tangent (tanh) activation function.
    """
    def activation(self, input_data):
        return np.tanh(input_data)
    
    def derivative(self, input_data):
        return 1 - np.tanh(input_data) ** 2


class Linear(Activation):
    """
    Linear (identity) activation function.
    """
    def activation(self, input_data):
        return input_data
    
    def derivative(self, input_data):
        return np.ones_like(input_data)
