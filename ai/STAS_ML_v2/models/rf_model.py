"""
Random Forest model implementation for STAS_ML v2
"""

import numpy as np
import joblib
from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..core.config import Config
from ..core.base import BaseModel, Logger, MetricsCalculator


class RandomForestModel(BaseModel):
    """Random Forest model implementation."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.logger = Logger("RandomForestModel")
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train Random Forest model."""
        
        # Create model based on target type
        if self.is_classification():
            self.model = RandomForestClassifier(**self.config.model.rf_params)
        else:
            self.model = RandomForestRegressor(**self.config.model.rf_params)
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training metrics
        train_pred = self.model.predict(X)
        metrics = {'train_samples': len(X)}
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            if self.is_classification():
                metrics.update({
                    'train_accuracy': MetricsCalculator.classification_metrics(y, train_pred)['accuracy'],
                    'val_accuracy': MetricsCalculator.classification_metrics(y_val, val_pred)['accuracy']
                })
            else:
                train_metrics = MetricsCalculator.regression_metrics(y, train_pred)
                val_metrics = MetricsCalculator.regression_metrics(y_val, val_pred)
                metrics.update({
                    'train_rmse': train_metrics['rmse'],
                    'val_rmse': val_metrics['rmse'],
                    'train_r2': train_metrics['r2'],
                    'val_r2': val_metrics['r2']
                })
        
        self.training_metrics = metrics
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for classification."""
        if not self.is_classification():
            raise ValueError("Probabilities only available for classification models")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance."""
        if not self.is_trained:
            return None
        
        importance = self.model.feature_importances_
        return {f"feature_{i}": imp for i, imp in enumerate(importance)}
    
    def save(self, filepath: str):
        """Save model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'config': self.config.to_dict(),
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.is_trained = model_data.get('is_trained', True)
        self.training_metrics = model_data.get('training_metrics', {})
        
        self.logger.info(f"Model loaded from {filepath}")