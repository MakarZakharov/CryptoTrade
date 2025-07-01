"""
Base class for market analysis models.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional, Union
import os
import json


class BaseModel(ABC):
    """
    Abstract base class for all market analysis models.
    
    All models should inherit from this class and implement the required methods.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the base model.
        
        Args:
            name: Optional name for the model
        """
        self.name = name or self.__class__.__name__
        self.model = None
        self.is_trained = False
        self.history = None
        self.metadata = {}
    
    @abstractmethod
    def build(self, **kwargs) -> None:
        """
        Build the model architecture.
        
        Args:
            **kwargs: Model-specific parameters
        """
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Training parameters
            
        Returns:
            Dictionary containing training history
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from
        """
        pass
    
    def save_metadata(self, path: str) -> None:
        """
        Save model metadata to disk.
        
        Args:
            path: Path to save the metadata
        """
        metadata_path = os.path.join(os.path.dirname(path), f"{self.name}_metadata.json")
        
        # Add basic metadata
        metadata = {
            "model_name": self.name,
            "model_type": self.__class__.__name__,
            "is_trained": self.is_trained,
            "training_history": self.history if isinstance(self.history, dict) else None,
            **self.metadata
        }
        
        # Save to file
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def load_metadata(self, path: str) -> Dict[str, Any]:
        """
        Load model metadata from disk.
        
        Args:
            path: Path to load the metadata from
            
        Returns:
            Dictionary containing model metadata
        """
        metadata_path = os.path.join(os.path.dirname(path), f"{self.name}_metadata.json")
        
        if not os.path.exists(metadata_path):
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.metadata = metadata
        return metadata
    
    def summary(self) -> str:
        """
        Get a summary of the model.
        
        Returns:
            String containing model summary
        """
        if self.model is None:
            return f"Model '{self.name}' has not been built yet."
        
        if hasattr(self.model, 'summary'):
            return self.model.summary()
        else:
            return str(self.model)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the model parameters.
        
        Returns:
            Dictionary containing model parameters
        """
        return {}
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set the model parameters.
        
        Args:
            **params: Model parameters
            
        Returns:
            Self for method chaining
        """
        return self