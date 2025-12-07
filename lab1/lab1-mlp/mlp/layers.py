import numpy as np

class Layer:
    """
    Base class for all neural network layers.
    Each layer should implement forward(), backward(), and update_params() methods.
    """
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input_data):
        """
        Forward pass through the layer.
        
        Parameters:
        input_data (numpy.ndarray): Input data to the layer.
        
        Returns:
        numpy.ndarray: Output from the layer.
        """
        raise NotImplementedError("Forward method must be implemented in subclass")
    
    def backward(self, output_gradient, learning_rate):
        """
        Backward pass through the layer.
        
        Parameters:
        output_gradient (numpy.ndarray): Gradient of the loss with respect to the layer's output.
        learning_rate (float): Learning rate for parameter updates.
        
        Returns:
        numpy.ndarray: Gradient of the loss with respect to the layer's input.
        """
        raise NotImplementedError("Backward method must be implemented in subclass")
    
    def update_params(self, learning_rate):
        """
        Update the layer's parameters (if any).
        
        Parameters:
        learning_rate (float): Learning rate for parameter updates.
        """
        pass


class Dense(Layer):
    """
    Fully connected neural network layer.
    """
    def __init__(self, input_size, output_size, weight_init="he"):
        """
        Initialize the dense layer.
        
        Parameters:
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        weight_init (str): Weight initialization method, "he" or "xavier".
        """
        super().__init__()
        
        # Initialize weights and biases
        if weight_init.lower() == "he":
            # He initialization for ReLU activation
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        elif weight_init.lower() == "xavier":
            # Xavier initialization for sigmoid/tanh activation
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        else:
            # Default random initialization
            self.weights = np.random.randn(input_size, output_size) * 0.01
        
        self.biases = np.zeros((1, output_size))
        
    def forward(self, input_data):
        """
        Forward pass through the dense layer.
        
        Parameters:
        input_data (numpy.ndarray): Input data of shape (batch_size, input_size).
        
        Returns:
        numpy.ndarray: Output of shape (batch_size, output_size).
        """
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        """
        Backward pass through the dense layer.
        
        Parameters:
        output_gradient (numpy.ndarray): Gradient of loss with respect to layer output, shape (batch_size, output_size).
        learning_rate (float): Learning rate for parameter updates.
        
        Returns:
        numpy.ndarray: Gradient of loss with respect to layer input, shape (batch_size, input_size).
        """
        # Calculate gradients
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        self.biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # For backward compatibility with direct update
        if learning_rate > 0:
            # Update parameters
            self.weights -= learning_rate * self.weights_gradient
            self.biases -= learning_rate * self.biases_gradient
        
        return input_gradient
    
    def update_params(self, learning_rate):
        """
        Update the layer's parameters using the gradients computed in backward().
        (Note: For Dense layer, parameters are updated directly in backward() method.)
        """
        pass
