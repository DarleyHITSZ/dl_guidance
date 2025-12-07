import numpy as np
from mlp.datasets import BostonHousingLoader
from mlp.layers import Dense
from mlp.activations import ReLU, Linear
from mlp.losses import MSE
from mlp.optimizers import Adam
from mlp.model import MLP

def build_model_method1():
    """
    Build MLP model using Method 1: Initialize with loss and optimizer.
    
    Returns:
    MLP: Configured MLP model.
    """
    print("\n=== Method 1: Initialize with loss and optimizer ===")
    # Initialize model with loss and optimizer
    model = MLP(loss=MSE(), optimizer=Adam(learning_rate=0.001))
    
    # Add layers
    model.add(Dense(13, 64, weight_init="he"))
    model.add(ReLU())
    model.add(Dense(64, 32, weight_init="he"))
    model.add(ReLU())
    model.add(Dense(32, 1, weight_init="he"))
    model.add(Linear())
    
    print("Model built successfully!")
    return model

def build_model_method2():
    """
    Build MLP model using Method 2: Set components separately.
    
    Returns:
    MLP: Configured MLP model.
    """
    print("\n=== Method 2: Set components separately ===")
    # Initialize empty model
    model = MLP()
    
    # Add layers
    model.add(Dense(13, 64, weight_init="he"))
    model.add(ReLU())
    model.add(Dense(64, 32, weight_init="he"))
    model.add(ReLU())
    model.add(Dense(32, 1, weight_init="he"))
    model.add(Linear())
    
    # Set loss function
    model.set_loss(MSE())
    
    # Set optimizer
    model.set_optimizer(Adam(learning_rate=0.001))
    
    print("Model built successfully!")
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test, name):
    """
    Train and evaluate the given model.
    
    Parameters:
    model (MLP): The MLP model to train.
    X_train (numpy.ndarray): Training features.
    y_train (numpy.ndarray): Training targets.
    X_test (numpy.ndarray): Test features.
    y_test (numpy.ndarray): Test targets.
    name (str): Model name for display.
    
    Returns:
    dict: Evaluation metrics.
    """
    print(f"\nTraining {name}...")
    
    # Train model (only 10 epochs for quick demo)
    history = model.train(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        X_val=X_test,
        y_val=y_test,
        verbose=True
    )
    
    # Evaluate model
    print(f"\nEvaluating {name}...")
    metrics = model.evaluate(X_test, y_test)
    print(f"Test R2: {metrics['r2']:.4f}")
    print(f"Test MSE: {metrics['mse']:.4f}")
    print(f"Test MAE: {metrics['mae']:.4f}")
    
    return metrics

def main():
    """
    Main function for advanced API demo.
    """
    print("=== Advanced API Demo for MLP ===")
    print("This demo shows two different ways to build an MLP model.")
    
    # Load and preprocess dataset
    print("\n1. Loading and preprocessing dataset...")
    loader = BostonHousingLoader(test_size=0.2, random_state=42)
    X_train, y_train, X_test, y_test = loader.load_data()
    
    # Build and train model using Method 1
    model1 = build_model_method1()
    metrics1 = train_and_evaluate(model1, X_train, y_train, X_test, y_test, "Model 1")
    
    # Build and train model using Method 2
    model2 = build_model_method2()
    metrics2 = train_and_evaluate(model2, X_train, y_train, X_test, y_test, "Model 2")
    
    # Compare results
    print("\n=== Results Comparison ===")
    print("Model 1 (Initialize with loss and optimizer):")
    print(f"  MSE: {metrics1['mse']:.4f}")
    print(f"  MAE: {metrics1['mae']:.4f}")
    print(f"  R2: {metrics1['r2']:.4f}")
    
    print("Model 2 (Set components separately):")
    print(f"  MSE: {metrics2['mse']:.4f}")
    print(f"  MAE: {metrics2['mae']:.4f}")
    print(f"  R2: {metrics2['r2']:.4f}")
    
    # Make predictions with both models
    print("\n=== Example Predictions ===")
    num_examples = 3
    indices = np.random.choice(X_test.shape[0], num_examples, replace=False)
    X_example = X_test[indices]
    y_example_true = y_test[indices]
    
    y_pred1 = model1.predict(X_example)
    y_pred2 = model2.predict(X_example)
    
    print("True vs Predicted prices:")
    for i in range(num_examples):
        print(f"Example {i+1}:")
        print(f"  True: {y_example_true[i][0]:.2f}")
        print(f"  Model 1 Pred: {y_pred1[i][0]:.2f}")
        print(f"  Model 2 Pred: {y_pred2[i][0]:.2f}")
    
    print("\n=== Advanced API Demo Completed ===")
    print("Both methods achieve similar results but demonstrate different API usage styles.")

if __name__ == "__main__":
    main()
