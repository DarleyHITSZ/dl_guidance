import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class MLP:
    """
    Multi-Layer Perceptron (MLP) neural network model.
    """
    def __init__(self, loss=None, optimizer=None):
        """
        Initialize the MLP model.
        
        Parameters:
        loss (Loss): Loss function object.
        optimizer (Optimizer): Optimizer object.
        """
        self.layers = []
        self.loss = loss
        self.optimizer = optimizer
    
    def add_layer(self, layer):
        """
        Add a layer to the model.
        
        Parameters:
        layer (Layer): Layer object to add.
        """
        self.layers.append(layer)
    
    def add(self, layer):
        """
        Alias for add_layer method.
        
        Parameters:
        layer (Layer): Layer object to add.
        """
        self.add_layer(layer)
    
    def set_loss(self, loss):
        """
        Set the loss function for the model.
        
        Parameters:
        loss (Loss): Loss function object.
        """
        self.loss = loss
    
    def set_optimizer(self, optimizer):
        """
        Set the optimizer for the model.
        
        Parameters:
        optimizer (Optimizer): Optimizer object.
        """
        self.optimizer = optimizer
    
    def forward(self, input_data):
        """
        Perform forward pass through all layers.
        
        Parameters:
        input_data (numpy.ndarray): Input data of shape (batch_size, input_size).
        
        Returns:
        numpy.ndarray: Model output of shape (batch_size, output_size).
        """
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, output_gradient):
        """
        Perform backward pass through all layers.
        
        Parameters:
        output_gradient (numpy.ndarray): Gradient of loss with respect to model output.
        """
        gradient = output_gradient
        # Backward pass through layers in reverse order
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, 0.0)  # Learning rate 0 since we use optimizer for updates
    
    def update(self):
        """
        Update all layers' parameters using the optimizer.
        """
        for layer in self.layers:
            self.optimizer.update(layer)
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, X_val=None, y_val=None, verbose=True):
        """
        Train the model on the given dataset.
        
        Parameters:
        X_train (numpy.ndarray): Training features of shape (num_samples, input_size).
        y_train (numpy.ndarray): Training targets of shape (num_samples, output_size).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        X_val (numpy.ndarray): Validation features (optional).
        y_val (numpy.ndarray): Validation targets (optional).
        verbose (bool): Whether to print training progress.
        
        Returns:
        dict: Training history containing loss and validation metrics.
        """
        history = {
            'loss': [],
            'val_loss': []
        }
        
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle training data for each epoch
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Training loop with mini-batches
            for i in range(0, num_samples, batch_size):
                # Get batch data
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Calculate loss
                batch_loss = self.loss.forward(y_pred, y_batch)
                epoch_loss += batch_loss
                num_batches += 1
                
                # Backward pass
                output_gradient = self.loss.backward(y_pred, y_batch)
                self.backward(output_gradient)
                
                # Update parameters
                self.update()
            
            # Calculate average loss for epoch
            avg_loss = epoch_loss / num_batches
            history['loss'].append(avg_loss)
            
            # Calculate validation metrics if validation data is provided
            if X_val is not None and y_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self.loss.forward(val_pred, y_val)
                history['val_loss'].append(val_loss)
            
            # Print progress
            if verbose:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test dataset.
        
        Parameters:
        X_test (numpy.ndarray): Test features of shape (num_samples, input_size).
        y_test (numpy.ndarray): Test targets of shape (num_samples, output_size).
        
        Returns:
        dict: Evaluation metrics including MSE, MAE, and R².
        """
        y_pred = self.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2
        }
    
    def predict(self, X):
        """
        Predict outputs for given input data.
        
        Parameters:
        X (numpy.ndarray): Input data of shape (num_samples, input_size).
        
        Returns:
        numpy.ndarray: Predicted outputs of shape (num_samples, output_size).
        """
        return self.forward(X)
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot the training history.
        
        Parameters:
        history (dict): Training history returned by train method.
        save_path (str): Path to save the plot (optional).
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        return plt
    
    def plot_predictions(self, y_true, y_pred, save_path=None):
        """
        Plot actual vs predicted values.
        
        Parameters:
        y_true (numpy.ndarray): Actual values.
        y_pred (numpy.ndarray): Predicted values.
        save_path (str): Path to save the plot (optional).
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.7, color='blue', label='Predictions')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        return plt
