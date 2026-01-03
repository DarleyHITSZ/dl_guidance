# Custom Deep Learning Framework

A lightweight, NumPy-based deep learning framework implementing core neural network components, including convolutional layers, fully connected layers, activation functions, and loss functions. Designed for educational purposes to demonstrate the fundamentals of deep learning architectures and backpropagation.

## Overview

This framework provides modular implementations of key deep learning building blocks, allowing users to construct and train neural networks from scratch. It includes:
- Convolutional layers (`Conv2D`) with efficient `im2col`/`col2im` operations
- Pooling layers (`MaxPool2D`)
- Fully connected layers (`Dense`) with various weight initializations
- Activation functions (ReLU, Softmax)
- Loss functions (Cross-Entropy)
- Utility layers (Flatten)

## Installation

### Prerequisites
- Python 3.6+
- NumPy
- Matplotlib
- TensorFlow (for data loading, optional)

### Setup
Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd <repository-directory>
pip install numpy matplotlib tensorflow
```

## Usage Example

### Training a CNN on MNIST
```python
import numpy as np
from dl_guidance.lab2.cnn_mnist import (
    Conv2D, MaxPool2D, Flatten, Dense,
    ReLU, Softmax, CrossEntropyLoss
)

# Load and preprocess data (example using TensorFlow)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 1, 28, 28) / 255.0
x_test = x_test.reshape(-1, 1, 28, 28) / 255.0

# Define model architecture
model = [
    Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
    ReLU(),
    MaxPool2D(pool_size=2, stride=2),
    Flatten(),
    Dense(in_features=14*14*16, out_features=100),
    ReLU(),
    Dense(in_features=100, out_features=10),
    Softmax()
]

# Loss function
criterion = CrossEntropyLoss()

# Training loop
epochs = 10
batch_size = 32
learning_rate = 0.01

for epoch in range(epochs):
    total_loss = 0
    for i in range(0, len(x_train), batch_size):
        # Get batch
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        # Forward pass
        out = x_batch
        for layer in model:
            out = layer.forward(out)
        
        # Calculate loss
        loss = criterion.forward(out, y_batch)
        total_loss += loss
        
        # Backward pass
        dout = criterion.backward()
        for layer in reversed(model):
            dout = layer.backward(dout)
        
        # Update parameters (implement optimizer or update in layers)
        for layer in model:
            if hasattr(layer, 'W'):  # Check if layer has weights
                layer.W -= learning_rate * layer.dW
                layer.b -= learning_rate * layer.db
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(x_train)*batch_size:.4f}")
```

## Key Components

### Layers
- `Conv2D`: Convolutional layer with configurable kernel size, stride, and padding
- `MaxPool2D`: Max pooling layer for spatial downsampling
- `Dense`: Fully connected layer with He/Xavier initialization options
- `Flatten`: Utility layer to flatten multi-dimensional tensors

### Activations
- `ReLU`: Rectified Linear Unit activation with backward pass
- `Softmax`: Stable softmax implementation for classification tasks

### Loss Functions
- `CrossEntropyLoss`: Cross-entropy loss with support for one-hot encoded labels

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Inspired by fundamental deep learning concepts from "Deep Learning" by Ian Goodfellow et al.
- MNIST dataset handling uses TensorFlow's utility functions for simplicity

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests for any improvements or additional features.
