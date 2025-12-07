# MLP Neural Network Implementation based on NumPy

A comprehensive implementation of a Multi-Layer Perceptron (MLP) neural network library using only NumPy, designed for solving regression problems like the Boston Housing price prediction task.

## Project Overview

This project implements a full-featured MLP neural network from scratch using NumPy, without relying on any deep learning frameworks like TensorFlow or PyTorch. The library includes all core components of a neural network, including:

- Dense (fully connected) layers
- Various activation functions (ReLU, Sigmoid, Tanh, Linear)
- Loss functions (MSE for regression)
- Optimizers (SGD with momentum, Adam)
- Model training and evaluation utilities
- Dataset loading and preprocessing

## Features

- **Pure NumPy Implementation**: All neural network operations are implemented using NumPy matrix operations for high performance.
- **Modular Design**: Easy to extend with new layers, activation functions, loss functions, and optimizers.
- **Comprehensive API**: Simple and intuitive interface for building, training, and evaluating MLP models.
- **Built-in Dataset Support**: Includes loader for the Boston Housing dataset.
- **Visualization Tools**: Built-in functions for plotting training history and prediction results.
- **Detailed Documentation**: Complete API reference and experiment guide.

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy 1.21 or higher

### Setup

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd lab1-mlp
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Usage

```python
from mlp.datasets import BostonHousingLoader
from mlp.layers import Dense
from mlp.activations import ReLU, Linear
from mlp.losses import MSE
from mlp.optimizers import Adam
from mlp.model import MLP

# Load and preprocess dataset
loader = BostonHousingLoader(test_size=0.2)
X_train, y_train, X_test, y_test = loader.load_data()

# Build MLP model
model = MLP(loss=MSE(), optimizer=Adam(learning_rate=0.001))
model.add(Dense(13, 64))
model.add(ReLU())
model.add(Dense(64, 32))
model.add(ReLU())
model.add(Dense(32, 1))
model.add(Linear())

# Train model
history = model.train(X_train, y_train, epochs=100, batch_size=32)

# Evaluate model
metrics = model.evaluate(X_test, y_test)
print(f"Test MSE: {metrics['mse']:.4f}")
print(f"Test R²: {metrics['r2']:.4f}")

# Make predictions
y_pred = model.predict(X_test)
```

### Run the Demo

To run the complete demo with the Boston Housing dataset:

```bash
python demo.py
```

This will:
- Load and preprocess the dataset
- Build and train an MLP model
- Evaluate model performance
- Show example predictions
- Generate and save visualization plots

## Model Building and Training Guide

### Creating a Model

There are two ways to create an MLP model:

**Method 1: Initialize with loss and optimizer**
```python
model = MLP(loss=MSE(), optimizer=Adam(learning_rate=0.001))
```

**Method 2: Set components separately**
```python
model = MLP()
model.set_loss(MSE())
model.set_optimizer(Adam(learning_rate=0.001))
```

### Adding Layers

Add layers to the model using the `add()` method:

```python
# Input layer (13 features) to hidden layer (64 neurons) with ReLU activation
model.add(Dense(13, 64, weight_init="he"))
model.add(ReLU())

# Hidden layer (64 neurons) to hidden layer (32 neurons) with ReLU activation
model.add(Dense(64, 32, weight_init="he"))
model.add(ReLU())

# Output layer (1 neuron) with Linear activation (for regression)
model.add(Dense(32, 1, weight_init="he"))
model.add(Linear())
```

### Training the Model

Train the model using the `train()` method:

```python
history = model.train(
    X_train, y_train,          # Training data
    epochs=100,                # Number of epochs
    batch_size=32,             # Batch size
    X_val=X_test, y_val=y_test,# Validation data
    verbose=True               # Print progress
)
```

### Evaluating the Model

Evaluate model performance using the `evaluate()` method:

```python
metrics = model.evaluate(X_test, y_test)
# Metrics include: mse, mae, r2
```

### Making Predictions

Predict new data using the `predict()` method:

```python
# Preprocess new data (if necessary)
X_new = loader.preprocess_new_data(new_data)

# Make predictions
y_pred = model.predict(X_new)
```

## API Reference

### Core Components

#### Layers
- `Layer`: Base class for all layers
- `Dense`: Fully connected neural network layer

#### Activation Functions
- `Activation`: Base class for activation functions
- `ReLU`: Rectified Linear Unit activation
- `Sigmoid`: Sigmoid activation
- `Tanh`: Hyperbolic tangent activation
- `Linear`: Linear (identity) activation

#### Loss Functions
- `Loss`: Base class for loss functions
- `MSE`: Mean Squared Error loss (for regression)

#### Optimizers
- `Optimizer`: Base class for optimizers
- `SGD`: Stochastic Gradient Descent with momentum
- `Adam`: Adaptive Moment Estimation

#### Model
- `MLP`: Main class for building and training neural network models

#### Datasets
- `BostonHousingLoader`: Loader for the Boston Housing dataset

## Project Structure

```
lab1-mlp/
├── mlp/                      # Core MLP library
│   ├── __init__.py          # Module exports
│   ├── activations.py       # Activation functions
│   ├── datasets.py          # Dataset loaders
│   ├── layers.py            # Neural network layers
│   ├── losses.py            # Loss functions
│   ├── model.py             # MLP model implementation
│   └── optimizers.py        # Optimization algorithms
├── demo.py                  # Basic demo script
├── demo_advanced.py         # Advanced API demo
├── requirements.txt         # Dependencies
├── README.md                # This file
└── EXPERIMENT_GUIDE.md      # Detailed experiment guide
```

## Examples

### Basic Demo

The `demo.py` script provides a complete example of using the MLP library to solve the Boston Housing price prediction problem:

```bash
python demo.py
```

### Advanced API Demo

The `demo_advanced.py` script demonstrates two different ways to build MLP models:

```bash
python demo_advanced.py
```

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgments

- The Boston Housing dataset used in this project is obtained from OpenML.
- This implementation was inspired by various online resources and tutorials on neural network implementation.
