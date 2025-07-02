"""
Base class for data processors.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Any


class BaseProcessor(ABC):
    """
    Abstract base class for data processors.
    
    All data processors should inherit from this class and implement the process method.
    """
    
    def __init__(self, window_size: int = 60):
        """
        Initialize the processor.
        
        Args:
            window_size: Size of the window for sequence data
        """
        self.window_size = window_size
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        """
        Process the data for model training and evaluation.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
        """
        pass
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate the input data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Validated DataFrame
        """
        if data is None or len(data) == 0:
            raise ValueError("Input data is empty")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Input data is missing required columns: {missing_columns}")
        
        # Check for NaN values
        if data.isnull().values.any():
            print("Warning: Input data contains NaN values. Filling with forward fill method.")
            data = data.fillna(method='ffill')
            
            # If there are still NaN values (e.g., at the beginning), fill with backward fill
            if data.isnull().values.any():
                data = data.fillna(method='bfill')
                
            # If there are still NaN values, fill with zeros
            if data.isnull().values.any():
                data = data.fillna(0)
        
        return data
    
    def _split_data(self, X: np.ndarray, y: np.ndarray, train_split: float = 0.7, val_split: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into training, validation, and test sets.
        
        Args:
            X: Input features
            y: Target values
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if train_split + val_split >= 1.0:
            raise ValueError("train_split + val_split must be less than 1.0")
        
        train_size = int(len(X) * train_split)
        val_size = int(len(X) * val_split)
        
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        
        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size]
        y_test = y[train_size + val_size:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test