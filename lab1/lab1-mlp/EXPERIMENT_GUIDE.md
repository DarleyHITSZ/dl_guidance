# MLP Neural Network Experiment Guide

## 1. Experiment Overview

### 1.1 Experiment Objectives

The primary objective of this experiment is to implement a complete Multi-Layer Perceptron (MLP) neural network using only NumPy and apply it to the Boston Housing price prediction problem. Through this experiment, you will:

- Understand the fundamental principles of neural networks and backpropagation
- Learn how to implement neural network components from scratch
- Gain hands-on experience with model building, training, and evaluation
- Explore hyperparameter tuning and model optimization techniques
- Learn how to analyze and interpret experimental results

### 1.2 Experiment Content

This experiment involves the following key components:

1. **Neural Network Implementation**: Building core components (layers, activation functions, loss functions, optimizers)
2. **Dataset Handling**: Loading, preprocessing, and splitting the Boston Housing dataset
3. **Model Development**: Constructing an MLP architecture for regression tasks
4. **Model Training**: Training the MLP using backpropagation and gradient descent
5. **Performance Evaluation**: Assessing model performance using appropriate metrics
6. **Visualization**: Plotting training history and prediction results
7. **Hyperparameter Tuning**: Exploring different configurations for better performance

### 1.3 Technical Stack

- **Programming Language**: Python 3.8+
- **Core Library**: NumPy 1.21+ (for matrix operations)
- **Data Processing**: Pandas, Scikit-learn
- **Visualization**: Matplotlib
- **No Deep Learning Frameworks**: Implementation uses only NumPy

## 2. Environment Setup

### 2.1 Prerequisites

- Python 3.8 or higher installed on your system
- Basic knowledge of Python programming
- Understanding of linear algebra and calculus fundamentals
- Familiarity with neural network concepts (optional but recommended)

### 2.2 Installation Steps

1. **Clone or download the project repository**
   ```bash
   # If cloning from a repository
   git clone <repository-url>
   cd lab1-mlp
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate
   
   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import numpy, pandas, sklearn, matplotlib; print('All dependencies installed successfully!')"
   ```

## 3. Code Structure Analysis

### 3.1 Core Library Structure

The MLP library is organized in the `mlp/` directory with the following components:

```
mlp/
├── __init__.py          # Library exports and version info
├── activations.py       # Activation functions implementation
├── datasets.py          # Dataset loading and preprocessing
├── layers.py            # Neural network layer implementations
├── losses.py            # Loss function implementations
├── model.py             # MLP model class
└── optimizers.py        # Optimization algorithm implementations
```

### 3.2 Key Component Analysis

#### 3.2.1 Layers (`layers.py`)

- **Layer Base Class**: Defines the interface for all neural network layers
- **Dense Layer**: Implements a fully connected layer with weight initialization options (He/Xavier)

#### 3.2.2 Activation Functions (`activations.py`)

- **ReLU**: Most commonly used activation function for hidden layers
- **Sigmoid**: Useful for binary classification tasks
- **Tanh**: Similar to sigmoid but centered at zero
- **Linear**: Used for regression outputs

#### 3.2.3 Loss Functions (`losses.py`)

- **MSE**: Mean Squared Error, suitable for regression tasks

#### 3.2.4 Optimizers (`optimizers.py`)

- **SGD**: Stochastic Gradient Descent with momentum option
- **Adam**: Adaptive Moment Estimation, combines AdaGrad and RMSProp features

#### 3.2.5 Model (`model.py`)

- **MLP Class**: Main class for building, training, and evaluating neural networks
- **Key Methods**:
  - `add()`: Add layers to the model
  - `train()`: Complete training loop with mini-batches
  - `evaluate()`: Calculate performance metrics
  - `predict()`: Make predictions on new data
  - `plot_training_history()`: Visualize training progress
  - `plot_predictions()`: Compare actual vs predicted values

#### 3.2.6 Datasets (`datasets.py`)

- **BostonHousingLoader**: Handles loading, splitting, and preprocessing the Boston Housing dataset

## 4. Detailed Experiment Steps

### 4.1 Step 1: Understanding the Dataset

The Boston Housing dataset contains 506 samples with 13 features each, predicting median housing values in Boston neighborhoods. Features include:

- CRIM: Per capita crime rate
- ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.
- INDUS: Proportion of non-retail business acres per town
- CHAS: Charles River dummy variable
- NOX: Nitric oxides concentration
- RM: Average number of rooms per dwelling
- AGE: Proportion of owner-occupied units built prior to 1940
- DIS: Weighted distances to five Boston employment centers
- RAD: Index of accessibility to radial highways
- TAX: Full-value property-tax rate per $10,000
- PTRATIO: Pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents
- LSTAT: % lower status of the population

**Target**: MEDV - Median value of owner-occupied homes in $1000's

### 4.2 Step 2: Building the Neural Network

1. **Define Model Architecture**:
   - Input layer: 13 neurons (one for each feature)
   - Hidden layer 1: 64 neurons with ReLU activation
   - Hidden layer 2: 32 neurons with ReLU activation
   - Output layer: 1 neuron with Linear activation (for regression)

2. **Initialize Model Components**:
   - Loss function: MSE (Mean Squared Error)
   - Optimizer: Adam with learning rate 0.001

3. **Implement Model**:
   ```python
   from mlp.model import MLP
   from mlp.layers import Dense
   from mlp.activations import ReLU, Linear
   from mlp.losses import MSE
   from mlp.optimizers import Adam
   
   model = MLP(loss=MSE(), optimizer=Adam(learning_rate=0.001))
   model.add(Dense(13, 64, weight_init="he"))
   model.add(ReLU())
   model.add(Dense(64, 32, weight_init="he"))
   model.add(ReLU())
   model.add(Dense(32, 1, weight_init="he"))
   model.add(Linear())
   ```

### 4.3 Step 3: Loading and Preprocessing Data

1. **Load Dataset**:
   ```python
   from mlp.datasets import BostonHousingLoader
   
   loader = BostonHousingLoader(test_size=0.2, random_state=42)
   X_train, y_train, X_test, y_test = loader.load_data()
   ```

2. **Data Preprocessing Steps**:
   - Standardization: Scaling features to have mean=0 and std=1
   - Splitting: 80% training data, 20% test data

### 4.4 Step 4: Model Training

1. **Configure Training Parameters**:
   - Epochs: 100
   - Batch size: 32
   - Validation data: Use test set for validation

2. **Train Model**:
   ```python
   history = model.train(
       X_train, y_train,
       epochs=100,
       batch_size=32,
       X_val=X_test,
       y_val=y_test,
       verbose=True
   )
   ```

### 4.5 Step 5: Model Evaluation

1. **Evaluate on Test Set**:
   ```python
   metrics = model.evaluate(X_test, y_test)
   print(f"Test MSE: {metrics['mse']:.4f}")
   print(f"Test MAE: {metrics['mae']:.4f}")
   print(f"Test R²: {metrics['r2']:.4f}")
   ```

2. **Key Metrics**:
   - **MSE**: Mean Squared Error - measures average squared difference between predictions and actual values
   - **MAE**: Mean Absolute Error - measures average absolute difference between predictions and actual values
   - **R²**: Coefficient of Determination - measures proportion of variance explained by the model (0-1)

### 4.6 Step 6: Visualization

1. **Plot Training History**:
   ```python
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(10, 6))
   plt.plot(history['loss'], label='Training Loss')
   plt.plot(history['val_loss'], label='Validation Loss')
   plt.title('Training History')
   plt.xlabel('Epochs')
   plt.ylabel('MSE Loss')
   plt.legend()
   plt.grid(True)
   plt.savefig('training_history.png')
   plt.show()
   ```

2. **Plot Predictions**:
   ```python
   y_pred = model.predict(X_test)
   
   plt.figure(figsize=(10, 6))
   plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
   plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
   plt.title('Actual vs Predicted Housing Prices')
   plt.xlabel('Actual Prices ($1000)')
   plt.ylabel('Predicted Prices ($1000)')
   plt.grid(True)
   plt.savefig('predictions.png')
   plt.show()
   ```

## 5. Results Analysis

### 5.1 Interpreting Training History

When analyzing the training history plot:

- **Convergence**: The loss should decrease over epochs and eventually stabilize
- **Overfitting**: If training loss continues to decrease while validation loss increases, the model is overfitting
- **Underfitting**: If both training and validation loss remain high, the model is underfitting
- **Learning Rate**: Adjust based on loss reduction speed (too high = unstable, too low = slow convergence)

### 5.2 Interpreting Prediction Results

The scatter plot of actual vs predicted values shows:

- **Ideal Line**: Perfect predictions would fall exactly on this line
- **Error Distribution**: The spread of points around the ideal line indicates prediction accuracy
- **Outliers**: Points far from the line represent predictions with large errors

### 5.3 Performance Metrics Interpretation

- **MSE**: Lower values indicate better performance (target: below 15 for this dataset)
- **MAE**: Similar to MSE but less sensitive to outliers
- **R²**: Values closer to 1 indicate better model fit (target: above 0.7 for this dataset)

## 6. Extension Experiments

### 6.1 Hyperparameter Tuning

Explore different configurations to improve performance:

1. **Network Architecture**:
   - Number of hidden layers (e.g., 1, 2, 3)
   - Number of neurons per layer (e.g., 32, 64, 128, 256)
   - Activation functions (compare ReLU with other functions)

2. **Optimization Parameters**:
   - Learning rates (0.01, 0.001, 0.0001)
   - Batch sizes (16, 32, 64, 128)
   - Optimizers (compare SGD with Adam)
   - Momentum values (for SGD)

3. **Training Parameters**:
   - Number of epochs (50, 100, 200, 500)
   - Weight initialization methods (He vs Xavier)

### 6.2 Regularization Techniques

Implement and test regularization methods:

1. **Dropout**: Add dropout layers to reduce overfitting
2. **L2 Regularization**: Add weight decay to loss function
3. **Early Stopping**: Stop training when validation loss stops improving

### 6.3 Feature Engineering

Explore feature-related improvements:

1. **Feature Selection**: Identify and use only the most relevant features
2. **Feature Transformation**: Apply non-linear transformations to improve feature representation
3. **Data Augmentation**: Generate synthetic data to improve model generalization

### 6.4 Cross-Validation

Implement k-fold cross-validation for more robust performance assessment:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    # Train and evaluate on each fold
    pass
```

## 7. Common Issues and Solutions

### 7.1 Installation Problems

**Issue**: Missing dependencies or version conflicts
**Solution**: Install exact versions specified in requirements.txt
```bash
pip install -r requirements.txt
```

### 7.2 Dataset Loading Errors

**Issue**: Cannot load Boston Housing dataset
**Solution**: The dataset is loaded from openml. Ensure you have internet access and the latest version of scikit-learn.

### 7.3 Training Issues

**Issue**: Loss does not decrease
**Possible Solutions**:
- Check learning rate (may be too small)
- Verify weight initialization (He initialization works best with ReLU)
- Check data preprocessing (features should be standardized)

**Issue**: Training loss decreases but validation loss increases
**Solution**: Model is overfitting. Try:
- Adding dropout layers
- Using L2 regularization
- Reducing model complexity
- Increasing training data

### 7.4 Performance Issues

**Issue**: High MSE or low R²
**Solutions**:
- Tune hyperparameters (learning rate, batch size)
- Increase model complexity (more layers/neurons)
- Try different optimizers
- Perform feature engineering

## 8. Experiment Report Requirements

### 8.1 Report Structure

Your experiment report should include the following sections:

1. **Abstract**: Brief summary of the experiment and key findings
2. **Introduction**: Experiment objectives and background
3. **Methodology**: Technical approach, model architecture, training parameters
4. **Implementation**: Code structure and key components
5. **Results**: Performance metrics, visualizations, analysis
6. **Discussion**: Interpretation of results, comparison with baseline models
7. **Conclusion**: Summary of findings, limitations, future work
8. **References**: Sources of information and inspiration

### 8.2 Key Content Requirements

- **Model Architecture**: Detailed description of the neural network structure
- **Hyperparameters**: All configuration settings used
- **Results**: Complete set of metrics (MSE, MAE, R²) for both training and testing
- **Visualizations**: Training history plot and actual vs predicted values plot
- **Analysis**: Discussion of model performance, strengths, and limitations
- **Comparison**: Results from different configurations or extension experiments

### 8.3 Evaluation Criteria

Your report will be evaluated based on:
- **Completeness**: Coverage of all experiment components
- **Technical Accuracy**: Correct implementation and interpretation
- **Analysis Depth**: Insights into results and model behavior
- **Creativity**: Novel approaches or extension experiments
- **Clarity**: Well-organized, readable report with proper visualizations

## 9. Additional Resources

- **NumPy Documentation**: https://numpy.org/doc/
- **Neural Networks and Deep Learning**: http://neuralnetworksanddeeplearning.com/
- **Boston Housing Dataset**: https://www.openml.org/d/531
- **MLP Regression Tutorial**: Various online resources for regression with neural networks

## 10. Conclusion

This experiment provides a comprehensive introduction to implementing neural networks from scratch using NumPy. By following the steps outlined in this guide, you will gain practical experience with all aspects of neural network development and learn valuable skills for machine learning model building and optimization.

Remember that successful machine learning involves not just implementation, but also experimentation, analysis, and continuous improvement. Use the extension experiments to further explore and refine your understanding of neural networks.
