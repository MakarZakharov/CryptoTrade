"""
XGBoost model implementation for market analysis.
"""

import numpy as np
import os
import pickle
import sys
from typing import Any, Dict, Optional, Tuple, Union, List
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Handle both package import and direct script execution
try:
    from .base_model import BaseModel
    from ..config import (
        DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
    )
except ImportError:
    # Add current directory to path for direct script execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(current_dir)
    sys.path.append(parent_dir)
    
    from base_model import BaseModel
    from config import (
        DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
    )


class XGBoostModel(BaseModel):
    """
    XGBoost model for time series prediction.
    
    XGBoost is a gradient boosting framework that uses decision trees and is known for
    its performance and speed. It's particularly effective for structured/tabular data
    and can capture complex non-linear patterns.
    """
    
    def __init__(self, name: str = None, learning_rate: float = DEFAULT_LEARNING_RATE,
                 max_depth: int = 5, n_estimators: int = 100, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, objective: str = 'reg:squarederror',
                 early_stopping_rounds: int = 10):
        """
        Initialize the XGBoost model.
        
        Args:
            name: Optional name for the model
            learning_rate: Learning rate (eta)
            max_depth: Maximum depth of a tree
            n_estimators: Number of boosting rounds
            subsample: Subsample ratio of the training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            objective: Learning objective
            early_stopping_rounds: Early stopping rounds
        """
        super().__init__(name=name or "XGBoost")
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.early_stopping_rounds = early_stopping_rounds
        self.build()
    
    def build(self, **kwargs) -> None:
        """
        Build the XGBoost model.
        
        Args:
            **kwargs: Additional model parameters
        """
        # Override parameters if provided
        learning_rate = kwargs.get('learning_rate', self.learning_rate)
        max_depth = kwargs.get('max_depth', self.max_depth)
        n_estimators = kwargs.get('n_estimators', self.n_estimators)
        subsample = kwargs.get('subsample', self.subsample)
        colsample_bytree = kwargs.get('colsample_bytree', self.colsample_bytree)
        objective = kwargs.get('objective', self.objective)
        early_stopping_rounds = kwargs.get('early_stopping_rounds', self.early_stopping_rounds)
        
        # Set parameters
        params = {
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'objective': objective,
            'verbosity': 1,
            'n_jobs': -1,  # Use all available cores
            'random_state': 42
        }
        
        # Create model
        self.model = xgb.XGBRegressor(**params)
        
        # Update metadata
        self.metadata.update({
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'objective': objective,
            'early_stopping_rounds': early_stopping_rounds
        })
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the XGBoost model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Training parameters
                - verbose: Verbosity level
                
        Returns:
            Dictionary containing training history
        """
        if self.model is None:
            self.build()
        
        # Get training parameters
        verbose = kwargs.get('verbose', 1)
        
        # Prepare validation data
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Train the model
        try:
            # Try with all parameters (newer XGBoost versions)
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric='rmse',
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=verbose
            )
        except TypeError as e:
            # Handle different parameter compatibility issues
            if 'eval_metric' in str(e):
                try:
                    print("XGBoost version doesn't support eval_metric parameter, using alternative approach")
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        early_stopping_rounds=self.early_stopping_rounds,
                        verbose=verbose
                    )
                except TypeError as e2:
                    if 'early_stopping_rounds' in str(e2):
                        print("XGBoost version doesn't support early_stopping_rounds parameter either, using basic fit")
                        self.model.fit(
                            X_train, y_train,
                            eval_set=eval_set,
                            verbose=verbose
                        )
                    else:
                        # Re-raise if it's a different TypeError
                        raise
            elif 'early_stopping_rounds' in str(e):
                print("XGBoost version doesn't support early_stopping_rounds parameter, using alternative approach")
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    eval_metric='rmse',
                    verbose=verbose
                )
            else:
                # Re-raise if it's a different TypeError
                raise
        
        self.is_trained = True
        
        # Get training history
        if hasattr(self.model, 'evals_result'):
            self.history = self.model.evals_result()
        else:
            # For older versions of XGBoost
            self.history = {}
        
        # Calculate final metrics
        train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        
        val_rmse = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        # Update metadata
        self.metadata.update({
            'training_parameters': {
                'final_train_rmse': train_rmse,
                'final_val_rmse': val_rmse,
                'best_iteration': self.model.best_iteration if hasattr(self.model, 'best_iteration') else None
            }
        })
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained XGBoost model.
        
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
        Save the XGBoost model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been built yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata
        self.save_metadata(path)
    
    def load(self, path: str) -> None:
        """
        Load the XGBoost model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Load the model
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.is_trained = True
        
        # Load metadata
        self.load_metadata(path)
        
        # Update instance variables from metadata
        if 'learning_rate' in self.metadata:
            self.learning_rate = self.metadata['learning_rate']
        if 'max_depth' in self.metadata:
            self.max_depth = self.metadata['max_depth']
        if 'n_estimators' in self.metadata:
            self.n_estimators = self.metadata['n_estimators']
        if 'subsample' in self.metadata:
            self.subsample = self.metadata['subsample']
        if 'colsample_bytree' in self.metadata:
            self.colsample_bytree = self.metadata['colsample_bytree']
        if 'objective' in self.metadata:
            self.objective = self.metadata['objective']
        if 'early_stopping_rounds' in self.metadata:
            self.early_stopping_rounds = self.metadata['early_stopping_rounds']
    
    def feature_importance(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Args:
            feature_names: List of feature names (optional)
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Map to feature names if provided
        if feature_names is not None:
            if len(feature_names) != len(importance):
                raise ValueError(f"Length of feature_names ({len(feature_names)}) does not match "
                                 f"number of features ({len(importance)})")
            return {name: float(score) for name, score in zip(feature_names, importance)}
        else:
            return {f"feature_{i}": float(score) for i, score in enumerate(importance)}
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get the XGBoost model parameters.
        
        Returns:
            Dictionary containing model parameters
        """
        return {
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'n_estimators': self.n_estimators,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'objective': self.objective,
            'early_stopping_rounds': self.early_stopping_rounds
        }
    
    def set_params(self, **params) -> 'XGBoostModel':
        """
        Set the XGBoost model parameters.
        
        Args:
            **params: Model parameters
            
        Returns:
            Self for method chaining
        """
        if 'learning_rate' in params:
            self.learning_rate = params['learning_rate']
        if 'max_depth' in params:
            self.max_depth = params['max_depth']
        if 'n_estimators' in params:
            self.n_estimators = params['n_estimators']
        if 'subsample' in params:
            self.subsample = params['subsample']
        if 'colsample_bytree' in params:
            self.colsample_bytree = params['colsample_bytree']
        if 'objective' in params:
            self.objective = params['objective']
        if 'early_stopping_rounds' in params:
            self.early_stopping_rounds = params['early_stopping_rounds']
        
        # Rebuild the model with new parameters
        self.build()
        
        return self