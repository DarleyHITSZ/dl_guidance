import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BostonHousingLoader:
    """
    Boston Housing dataset loader and preprocessor.
    """
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the Boston Housing dataset loader.
        
        Parameters:
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_name = None
    
    def load_data(self):
        """
        Load and preprocess the Boston Housing dataset.
        
        Returns:
        tuple: (X_train, y_train, X_test, y_test)
            X_train (numpy.ndarray): Training features.
            y_train (numpy.ndarray): Training targets.
            X_test (numpy.ndarray): Test features.
            y_test (numpy.ndarray): Test targets.
        """
        print("Loading Boston Housing dataset...")
        
        # Load dataset using fetch_openml (new method since sklearn 1.0)
        # The Boston Housing dataset is available as 'boston'
        # Note: The original Boston dataset was removed from sklearn due to ethical concerns,
        # but it can be loaded from openml.
        data = fetch_openml(name="boston", version=1, as_frame=True)
        
        X = data.data.values
        y = data.target.values.reshape(-1, 1)
        
        self.feature_names = data.feature_names
        self.target_name = data.target.name
        
        # Split the dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Standardize features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"Dataset loaded successfully!")
        print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
        print(f"Features: {X_train.shape[1]}, Target: {y_train.shape[1]}")
        
        return X_train, y_train, X_test, y_test
    
    def preprocess_new_data(self, X):
        """
        Preprocess new data using the same scaler fit on the training data.
        
        Parameters:
        X (numpy.ndarray): New data to preprocess.
        
        Returns:
        numpy.ndarray: Preprocessed data.
        """
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Call load_data() first.")
        
        return self.scaler.transform(X)
    
    def get_feature_names(self):
        """
        Get the feature names of the dataset.
        
        Returns:
        list: List of feature names.
        """
        return self.feature_names
    
    def get_target_name(self):
        """
        Get the target name of the dataset.
        
        Returns:
        str: Target name.
        """
        return self.target_name
