import numpy as np

class Loss:
    """
    Base class for loss functions.
    """
    def __init__(self):
        pass
    
    def forward(self, y_pred, y_true):
        """
        Compute the loss value.
        
        Parameters:
        y_pred (numpy.ndarray): Predicted values.
        y_true (numpy.ndarray): True values.
        
        Returns:
        float or numpy.ndarray: Loss value(s).
        """
        raise NotImplementedError("Forward method must be implemented in subclass")
    
    def backward(self, y_pred, y_true):
        """
        Compute the gradient of the loss with respect to the predictions.
        
        Parameters:
        y_pred (numpy.ndarray): Predicted values.
        y_true (numpy.ndarray): True values.
        
        Returns:
        numpy.ndarray: Gradient of loss with respect to predictions.
        """
        raise NotImplementedError("Backward method must be implemented in subclass")


class MSE(Loss):
    """
    Mean Squared Error (MSE) loss function, used for regression tasks.
    """
    def forward(self, y_pred, y_true):
        """
        Compute the MSE loss.
        
        Parameters:
        y_pred (numpy.ndarray): Predicted values of shape (batch_size, output_size).
        y_true (numpy.ndarray): True values of shape (batch_size, output_size).
        
        Returns:
        float: Mean squared error loss.
        """
        return np.mean(np.power(y_pred - y_true, 2))
    
    def backward(self, y_pred, y_true):
        """
        Compute the gradient of MSE loss with respect to predictions.
        
        Parameters:
        y_pred (numpy.ndarray): Predicted values of shape (batch_size, output_size).
        y_true (numpy.ndarray): True values of shape (batch_size, output_size).
        
        Returns:
        numpy.ndarray: Gradient of loss with respect to predictions, shape (batch_size, output_size).
        """
        batch_size = y_pred.shape[0]
        return 2 * (y_pred - y_true) / batch_size
