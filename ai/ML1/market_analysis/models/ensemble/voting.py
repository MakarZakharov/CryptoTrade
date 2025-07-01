"""
Voting ensemble model implementation for market analysis.
"""

import numpy as np
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.metrics import mean_squared_error

# Handle both package import and direct script execution
try:
    from ..base_model import BaseModel
except ImportError:
    # Add current directory to path for direct script execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    sys.path.append(current_dir)
    sys.path.append(parent_dir)
    sys.path.append(grandparent_dir)
    
    try:
        from market_analysis.models.base_model import BaseModel
    except ImportError:
        try:
            sys.path.append(os.path.dirname(grandparent_dir))
            from market_analysis.models.base_model import BaseModel
        except ImportError:
            from models.base_model import BaseModel


class VotingEnsembleModel(BaseModel):
    """
    Voting ensemble model for time series prediction.
    
    Voting combines predictions from multiple base models using a weighted average.
    This implementation supports any models that inherit from BaseModel.
    """
    
    def __init__(self, base_models: List[BaseModel], weights: Optional[List[float]] = None, 
                 name: str = None):
        """
        Initialize the voting ensemble model.
        
        Args:
            base_models: List of base models (must inherit from BaseModel)
            weights: List of weights for each base model (default: equal weights)
            name: Optional name for the model
        """
        super().__init__(name=name or "VotingEnsemble")
        self.base_models = base_models
        
        # Set weights (default to equal weights)
        if weights is None:
            weights = [1.0 / len(base_models)] * len(base_models)
        elif len(weights) != len(base_models):
            raise ValueError(f"Number of weights ({len(weights)}) must match "
                             f"number of base models ({len(base_models)})")
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        
        self.build()
    
    def build(self, **kwargs) -> None:
        """
        Build the ensemble model.
        
        Args:
            **kwargs: Additional model parameters
        """
        # Override parameters if provided
        weights = kwargs.get('weights', self.weights)
        
        # Set weights
        if weights is not None:
            if len(weights) != len(self.base_models):
                raise ValueError(f"Number of weights ({len(weights)}) must match "
                                 f"number of base models ({len(self.base_models)})")
            
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        # Update metadata
        self.metadata.update({
            'base_models': [model.name for model in self.base_models],
            'weights': self.weights
        })
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the ensemble model on the provided data.
        
        This method trains all base models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Training parameters
                - base_model_kwargs: Dictionary of kwargs for each base model (by name)
                - verbose: Verbosity level
                
        Returns:
            Dictionary containing training history
        """
        # Get training parameters
        base_model_kwargs = kwargs.get('base_model_kwargs', {})
        verbose = kwargs.get('verbose', 1)
        
        # Train base models
        if verbose:
            print(f"Training {len(self.base_models)} base models...")
        
        for model in self.base_models:
            model_kwargs = base_model_kwargs.get(model.name, {})
            if verbose:
                print(f"Training base model: {model.name}")
            model.train(X_train, y_train, X_val, y_val, **model_kwargs)
        
        # Evaluate on training set
        train_preds = self.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        
        # Evaluate on validation set if provided
        val_rmse = None
        if X_val is not None and y_val is not None:
            val_preds = self.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        
        self.is_trained = True
        
        # Create history dictionary
        history = {
            'train_rmse': train_rmse,
            'val_rmse': val_rmse
        }
        self.history = history
        
        # Update metadata
        self.metadata.update({
            'training_parameters': {
                'final_train_rmse': train_rmse,
                'final_val_rmse': val_rmse
            }
        })
        
        if verbose:
            print(f"Ensemble training complete. Train RMSE: {train_rmse:.4f}, "
                  f"Val RMSE: {val_rmse:.4f}" if val_rmse else "")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained ensemble model.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        if not all(model.is_trained for model in self.base_models):
            raise ValueError("All base models must be trained before making predictions")
        
        # Get predictions from each base model
        predictions = []
        for model in self.base_models:
            preds = model.predict(X)
            predictions.append(preds)
        
        # Combine predictions using weighted average
        weighted_preds = np.zeros_like(predictions[0])
        for i, preds in enumerate(predictions):
            weighted_preds += preds * self.weights[i]
        
        return weighted_preds
    
    def save(self, path: str) -> None:
        """
        Save the ensemble model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Create ensemble directory
        ensemble_dir = os.path.join(os.path.dirname(path), f"{self.name}_ensemble")
        os.makedirs(ensemble_dir, exist_ok=True)
        
        # Save base models
        base_model_paths = []
        for i, model in enumerate(self.base_models):
            model_path = os.path.join(ensemble_dir, f"base_model_{i}_{model.name}")
            model.save(model_path)
            base_model_paths.append((model.name, model_path))
        
        # Save ensemble info
        ensemble_info = {
            'base_model_paths': base_model_paths,
            'weights': self.weights
        }
        
        with open(path, 'wb') as f:
            pickle.dump(ensemble_info, f)
        
        # Save metadata
        self.save_metadata(path)
    
    def load(self, path: str) -> None:
        """
        Load the ensemble model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load ensemble info
        with open(path, 'rb') as f:
            ensemble_info = pickle.load(f)
        
        # Set weights
        self.weights = ensemble_info['weights']
        
        # Load base models
        self.base_models = []
        for model_name, model_path in ensemble_info['base_model_paths']:
            # We need to determine the model type from the name
            # This is a simplistic approach and might need to be improved
            if 'LSTM' in model_name:
                try:
                    from ..lstm_model import LSTMModel
                except ImportError:
                    try:
                        from market_analysis.models.lstm_model import LSTMModel
                    except ImportError:
                        from models.lstm_model import LSTMModel
                model = LSTMModel(input_shape=(1, 1))  # Placeholder shape
            elif 'GRU' in model_name:
                try:
                    from ..gru_model import GRUModel
                except ImportError:
                    try:
                        from market_analysis.models.gru_model import GRUModel
                    except ImportError:
                        from models.gru_model import GRUModel
                model = GRUModel(input_shape=(1, 1))  # Placeholder shape
            elif 'Transformer' in model_name:
                try:
                    from ..transformer_model import TransformerModel
                except ImportError:
                    try:
                        from market_analysis.models.transformer_model import TransformerModel
                    except ImportError:
                        from models.transformer_model import TransformerModel
                model = TransformerModel(input_shape=(1, 1))  # Placeholder shape
            elif 'XGBoost' in model_name:
                try:
                    from ..xgboost_model import XGBoostModel
                except ImportError:
                    try:
                        from market_analysis.models.xgboost_model import XGBoostModel
                    except ImportError:
                        from models.xgboost_model import XGBoostModel
                model = XGBoostModel()
            else:
                raise ValueError(f"Unknown model type: {model_name}")
            
            # Load the model
            model.load(model_path)
            self.base_models.append(model)
        
        self.is_trained = True
        
        # Load metadata
        self.load_metadata(path)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the ensemble model parameters.
        
        Returns:
            Dictionary containing model parameters
        """
        return {
            'base_models': [model.name for model in self.base_models],
            'weights': self.weights
        }
    
    def set_params(self, **params) -> 'VotingEnsembleModel':
        """
        Set the ensemble model parameters.
        
        Args:
            **params: Model parameters
            
        Returns:
            Self for method chaining
        """
        if 'weights' in params:
            weights = params['weights']
            if len(weights) != len(self.base_models):
                raise ValueError(f"Number of weights ({len(weights)}) must match "
                                 f"number of base models ({len(self.base_models)})")
            
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        # Update metadata
        self.metadata.update({
            'base_models': [model.name for model in self.base_models],
            'weights': self.weights
        })
        
        return self