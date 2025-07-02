"""
LSTM model implementation for market analysis.
"""

import numpy as np
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Handle both package import and direct script execution
try:
    from .base_model import BaseModel
    from ..config import (
        DEFAULT_LSTM_UNITS, DEFAULT_DROPOUT_RATE, DEFAULT_LEARNING_RATE,
        DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_EARLY_STOPPING_PATIENCE,
        DEFAULT_REDUCE_LR_PATIENCE, DEFAULT_REDUCE_LR_FACTOR, DEFAULT_MIN_LR
    )
except ImportError:
    # Add current directory to path for direct script execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(current_dir)
    sys.path.append(parent_dir)
    
    from base_model import BaseModel
    from config import (
        DEFAULT_LSTM_UNITS, DEFAULT_DROPOUT_RATE, DEFAULT_LEARNING_RATE,
        DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_EARLY_STOPPING_PATIENCE,
        DEFAULT_REDUCE_LR_PATIENCE, DEFAULT_REDUCE_LR_FACTOR, DEFAULT_MIN_LR
    )


class LSTMModel(BaseModel):
    """
    LSTM model for time series prediction.
    """
    
    def __init__(self, input_shape: Tuple[int, int], name: str = None, units: int = DEFAULT_LSTM_UNITS, 
                 dropout: float = DEFAULT_DROPOUT_RATE, learning_rate: float = DEFAULT_LEARNING_RATE):
        """
        Initialize the LSTM model.
        
        Args:
            input_shape: Shape of the input data (sequence_length, features)
            name: Optional name for the model
            units: Number of LSTM units in each layer
            dropout: Dropout rate
            learning_rate: Learning rate for the optimizer
        """
        super().__init__(name=name or "LSTM")
        self.input_shape = input_shape
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.build()
    
    def build(self, **kwargs) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            **kwargs: Additional model parameters
        """
        # Override parameters if provided
        units = kwargs.get('units', self.units)
        dropout = kwargs.get('dropout', self.dropout)
        learning_rate = kwargs.get('learning_rate', self.learning_rate)
        
        # Build the model
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=self.input_shape),
            Dropout(dropout),
            LSTM(units, return_sequences=True),
            Dropout(dropout),
            LSTM(units),
            Dropout(dropout),
            Dense(1)
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'
        )
        
        self.model = model
        self.metadata.update({
            'input_shape': self.input_shape,
            'units': units,
            'dropout': dropout,
            'learning_rate': learning_rate
        })
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the LSTM model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Training parameters
                - batch_size: Batch size for training
                - epochs: Number of training epochs
                - verbose: Verbosity level
                - callbacks: Additional callbacks
                
        Returns:
            Dictionary containing training history
        """
        if self.model is None:
            self.build()
        
        # Get training parameters
        batch_size = kwargs.get('batch_size', DEFAULT_BATCH_SIZE)
        epochs = kwargs.get('epochs', DEFAULT_EPOCHS)
        verbose = kwargs.get('verbose', 1)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Prepare callbacks
        callbacks = kwargs.get('callbacks', [])
        
        # Add default callbacks if not provided
        if not callbacks:
            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=DEFAULT_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=verbose
            )
            
            # Learning rate reduction callback
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=DEFAULT_REDUCE_LR_FACTOR,
                patience=DEFAULT_REDUCE_LR_PATIENCE,
                min_lr=DEFAULT_MIN_LR,
                verbose=verbose
            )
            
            # Combine callbacks
            callbacks = [early_stopping, reduce_lr]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_trained = True
        self.history = history.history
        
        # Update metadata
        self.metadata.update({
            'training_parameters': {
                'batch_size': batch_size,
                'epochs': epochs,
                'final_epoch': len(history.history['loss']),
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1] if validation_data else None
            }
        })
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained LSTM model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """
        Save the LSTM model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been built yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        self.model.save(path)
        
        # Save metadata
        self.save_metadata(path)
    
    def load(self, path: str) -> None:
        """
        Load the LSTM model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load the model
        self.model = load_model(path)
        self.is_trained = True
        
        # Load metadata
        self.load_metadata(path)
        
        # Update instance variables from metadata
        if 'input_shape' in self.metadata:
            self.input_shape = tuple(self.metadata['input_shape'])
        if 'units' in self.metadata:
            self.units = self.metadata['units']
        if 'dropout' in self.metadata:
            self.dropout = self.metadata['dropout']
        if 'learning_rate' in self.metadata:
            self.learning_rate = self.metadata['learning_rate']
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the LSTM model parameters.
        
        Returns:
            Dictionary containing model parameters
        """
        return {
            'input_shape': self.input_shape,
            'units': self.units,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate
        }
    
    def set_params(self, **params) -> 'LSTMModel':
        """
        Set the LSTM model parameters.
        
        Args:
            **params: Model parameters
            
        Returns:
            Self for method chaining
        """
        if 'input_shape' in params:
            self.input_shape = params['input_shape']
        if 'units' in params:
            self.units = params['units']
        if 'dropout' in params:
            self.dropout = params['dropout']
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        
        # Rebuild the model with new parameters
        self.build()
        
        return self