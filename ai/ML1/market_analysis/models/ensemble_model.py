"""
Ensemble model implementations for market analysis.
"""

import numpy as np
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Handle both package import and direct script execution
try:
    from .base_model import BaseModel
except ImportError:
    # Add current directory to path for direct script execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(current_dir)
    sys.path.append(parent_dir)
    
    from base_model import BaseModel


class StackingEnsembleModel(BaseModel):
    """
    Stacking ensemble model for time series prediction.
    
    Stacking combines multiple base models and uses a meta-model to combine their predictions.
    This implementation supports any models that inherit from BaseModel.
    """
    
    def __init__(self, base_models: List[BaseModel], meta_model: Optional[Any] = None, 
                 name: str = None, use_features_in_meta: bool = False):
        """
        Initialize the stacking ensemble model.
        
        Args:
            base_models: List of base models (must inherit from BaseModel)
            meta_model: Meta-model to combine base model predictions (default: LinearRegression)
            name: Optional name for the model
            use_features_in_meta: Whether to include original features in meta-model input
        """
        super().__init__(name=name or "StackingEnsemble")
        self.base_models = base_models
        self.meta_model = meta_model or LinearRegression()
        self.use_features_in_meta = use_features_in_meta
        self.build()
    
    def build(self, **kwargs) -> None:
        """
        Build the ensemble model.
        
        Args:
            **kwargs: Additional model parameters
        """
        # Override parameters if provided
        meta_model = kwargs.get('meta_model', self.meta_model)
        use_features_in_meta = kwargs.get('use_features_in_meta', self.use_features_in_meta)
        
        # Set meta-model
        self.meta_model = meta_model
        self.use_features_in_meta = use_features_in_meta
        
        # Update metadata
        self.metadata.update({
            'base_models': [model.name for model in self.base_models],
            'meta_model_type': type(self.meta_model).__name__,
            'use_features_in_meta': self.use_features_in_meta
        })
    
    def _get_base_model_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from all base models.
        
        Args:
            X: Input features
            
        Returns:
            Array of base model predictions
        """
        # Get predictions from each base model
        base_predictions = []
        for model in self.base_models:
            if not model.is_trained:
                raise ValueError(f"Base model '{model.name}' has not been trained yet")
            
            # Get predictions and reshape to ensure 2D array
            preds = model.predict(X)
            preds = preds.reshape(-1, 1)
            base_predictions.append(preds)
        
        # Concatenate predictions
        base_predictions = np.hstack(base_predictions)
        
        # Optionally include original features
        if self.use_features_in_meta:
            # For time series data, we need to flatten the sequence dimension
            if len(X.shape) == 3:
                X_flat = X.reshape(X.shape[0], -1)
            else:
                X_flat = X
            base_predictions = np.hstack([base_predictions, X_flat])
        
        return base_predictions
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the ensemble model on the provided data.
        
        This method trains all base models and then trains the meta-model on the base model predictions.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Training parameters
                - train_base_models: Whether to train base models (default: True)
                - base_model_kwargs: Dictionary of kwargs for each base model (by name)
                - verbose: Verbosity level
                
        Returns:
            Dictionary containing training history
        """
        # Get training parameters
        train_base_models = kwargs.get('train_base_models', True)
        base_model_kwargs = kwargs.get('base_model_kwargs', {})
        verbose = kwargs.get('verbose', 1)
        
        # Train base models if requested
        if train_base_models:
            if verbose:
                print(f"Training {len(self.base_models)} base models...")
            
            for model in self.base_models:
                model_kwargs = base_model_kwargs.get(model.name, {})
                if verbose:
                    print(f"Training base model: {model.name}")
                model.train(X_train, y_train, X_val, y_val, **model_kwargs)
        
        # Get base model predictions for meta-model training
        if verbose:
            print("Generating base model predictions for meta-model training...")
        
        base_train_preds = self._get_base_model_predictions(X_train)
        
        # Train meta-model
        if verbose:
            print("Training meta-model...")
        
        self.meta_model.fit(base_train_preds, y_train)
        
        # Evaluate on training set
        train_preds = self.meta_model.predict(base_train_preds)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        
        # Evaluate on validation set if provided
        val_rmse = None
        if X_val is not None and y_val is not None:
            base_val_preds = self._get_base_model_predictions(X_val)
            val_preds = self.meta_model.predict(base_val_preds)
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
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        # Get base model predictions
        base_preds = self._get_base_model_predictions(X)
        
        # Make meta-model predictions
        return self.meta_model.predict(base_preds)
    
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
        
        # Save meta-model
        meta_model_path = os.path.join(ensemble_dir, "meta_model.pkl")
        with open(meta_model_path, 'wb') as f:
            pickle.dump(self.meta_model, f)
        
        # Save ensemble info
        ensemble_info = {
            'base_model_paths': base_model_paths,
            'meta_model_path': meta_model_path,
            'use_features_in_meta': self.use_features_in_meta
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
        
        # Load meta-model
        meta_model_path = ensemble_info['meta_model_path']
        with open(meta_model_path, 'rb') as f:
            self.meta_model = pickle.load(f)
        
        # Set use_features_in_meta
        self.use_features_in_meta = ensemble_info['use_features_in_meta']
        
        # Load base models
        self.base_models = []
        for model_name, model_path in ensemble_info['base_model_paths']:
            # We need to determine the model type from the name
            # This is a simplistic approach and might need to be improved
            if 'LSTM' in model_name:
                try:
                    from .lstm_model import LSTMModel
                except ImportError:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    sys.path.append(current_dir)
                    from lstm_model import LSTMModel
                model = LSTMModel(input_shape=(1, 1))  # Placeholder shape
            elif 'GRU' in model_name:
                try:
                    from .gru_model import GRUModel
                except ImportError:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    sys.path.append(current_dir)
                    from gru_model import GRUModel
                model = GRUModel(input_shape=(1, 1))  # Placeholder shape
            elif 'Transformer' in model_name:
                try:
                    from .transformer_model import TransformerModel
                except ImportError:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    sys.path.append(current_dir)
                    from transformer_model import TransformerModel
                model = TransformerModel(input_shape=(1, 1))  # Placeholder shape
            elif 'XGBoost' in model_name:
                try:
                    from .xgboost_model import XGBoostModel
                except ImportError:
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    sys.path.append(current_dir)
                    from xgboost_model import XGBoostModel
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
            'meta_model_type': type(self.meta_model).__name__,
            'use_features_in_meta': self.use_features_in_meta
        }
    
    def set_params(self, **params) -> 'StackingEnsembleModel':
        """
        Set the ensemble model parameters.
        
        Args:
            **params: Model parameters
            
        Returns:
            Self for method chaining
        """
        if 'meta_model' in params:
            self.meta_model = params['meta_model']
        if 'use_features_in_meta' in params:
            self.use_features_in_meta = params['use_features_in_meta']
        
        # Update metadata
        self.metadata.update({
            'base_models': [model.name for model in self.base_models],
            'meta_model_type': type(self.meta_model).__name__,
            'use_features_in_meta': self.use_features_in_meta
        })
        
        return self


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
                    from .lstm_model import LSTMModel
                except ImportError:
                    from market_analysis.models.lstm_model import LSTMModel
                model = LSTMModel(input_shape=(1, 1))  # Placeholder shape
            elif 'GRU' in model_name:
                try:
                    from .gru_model import GRUModel
                except ImportError:
                    from market_analysis.models.gru_model import GRUModel
                model = GRUModel(input_shape=(1, 1))  # Placeholder shape
            elif 'Transformer' in model_name:
                try:
                    from .transformer_model import TransformerModel
                except ImportError:
                    from market_analysis.models.transformer_model import TransformerModel
                model = TransformerModel(input_shape=(1, 1))  # Placeholder shape
            elif 'XGBoost' in model_name:
                try:
                    from .xgboost_model import XGBoostModel
                except ImportError:
                    from market_analysis.models.xgboost_model import XGBoostModel
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