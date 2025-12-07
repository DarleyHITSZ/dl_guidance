import numpy as np
import matplotlib.pyplot as plt
from mlp.datasets import BostonHousingLoader
from mlp.layers import Dense
from mlp.activations import ReLU, Linear
from mlp.losses import MSE
from mlp.optimizers import Adam
from mlp.model import MLP

def main():
    """
    Main function for the Boston Housing MLP demo.
    """
    print("=== MLP for Boston Housing Price Prediction ===")
    
    # 1. Load and preprocess dataset
    print("\n1. Loading and preprocessing dataset...")
    loader = BostonHousingLoader(test_size=0.2, random_state=42)
    X_train, y_train, X_test, y_test = loader.load_data()
    
    # 2. Build MLP model
    print("\n2. Building MLP model...")
    # Suggested structure: 13→64(ReLU)→32(ReLU)→1(Linear)
    model = MLP(loss=MSE(), optimizer=Adam(learning_rate=0.001))
    model.add(Dense(13, 128, weight_init="he"))
    model.add(ReLU())
    model.add(Dense(128, 32, weight_init="he"))
    model.add(ReLU())
    model.add(Dense(32, 1, weight_init="he"))
    model.add(Linear())
    
    print("Model architecture:")
    print("Input layer: 13 features")
    print("Hidden layer 1: 64 neurons with ReLU activation")
    print("Hidden layer 2: 32 neurons with ReLU activation")
    print("Output layer: 1 neuron with Linear activation")
    print(f"Loss function: {type(model.loss).__name__}")
    print(f"Optimizer: {type(model.optimizer).__name__} (learning rate: {model.optimizer.learning_rate})")
    
    # 3. Train model
    print("\n3. Training model...")
    history = model.train(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        X_val=X_test,
        y_val=y_test,
        verbose=True
    )
    
    # 4. Evaluate model performance
    print("\n4. Evaluating model performance...")
    metrics = model.evaluate(X_test, y_test)
    print(f"Test MSE: {metrics['mse']:.4f}")
    print(f"Test MAE: {metrics['mae']:.4f}")
    print(f"Test R2: {metrics['r2']:.4f}")
    
    # 5. Make example predictions
    print("\n5. Making example predictions...")
    num_examples = 5
    indices = np.random.choice(X_test.shape[0], num_examples, replace=False)
    X_example = X_test[indices]
    y_example_true = y_test[indices]
    y_example_pred = model.predict(X_example)
    
    print("Example predictions:")
    for i in range(num_examples):
        print(f"Example {i+1}: True = {y_example_true[i][0]:.2f}, Predicted = {y_example_pred[i][0]:.2f}")
    
    # 6. Visualize training curves and prediction results
    print("\n6. Visualizing results...")
    
    # Plot training history
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Training history
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Actual vs Predicted
    plt.subplot(1, 2, 2)
    y_pred = model.predict(X_test)
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual House Prices')
    plt.ylabel('Predicted House Prices')
    plt.legend()
    plt.grid(True)
    
    # 7. Save results image
    plt.tight_layout()
    save_path = "boston_housing_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nResults saved to: {save_path}")
    
    # Show plots
    plt.show()
    
    print("\n=== Demo completed successfully! ===")

if __name__ == "__main__":
    main()
