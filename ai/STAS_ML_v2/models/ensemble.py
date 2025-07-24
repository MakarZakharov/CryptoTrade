"""
Ensemble model implementation for STAS_ML v2
"""

import numpy as np
import joblib
from typing import Dict, Any, Optional, List
from sklearn.ensemble import VotingClassifier, VotingRegressor

from ..core.config import Config
from ..core.base import BaseModel, Logger, MetricsCalculator, ModelFactory


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple base models."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.logger = Logger("EnsembleModel")
        
        self.base_models = []
        self.ensemble_weights = None
    
    def _create_base_models(self) -> List[BaseModel]:
        """Create base models for ensemble."""
        models = []
        model_types = self.config.model.ensemble_params.get('models', ['xgboost', 'random_forest'])
        
        for model_type in model_types:
            # Create temporary config for each model
            temp_config = Config()
            temp_config.__dict__.update(self.config.__dict__)
            temp_config.model.model_type = temp_config.model.model_type.__class__(model_type)
            
            model = ModelFactory.create_model(temp_config)
            models.append((model_type, model))
        
        return models
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train ensemble model."""
        
        self.base_models = self._create_base_models()
        
        # Train each base model
        individual_metrics = {}
        trained_models = []
        
        for model_name, model in self.base_models:
            self.logger.info(f"Training {model_name} model...")
            
            try:
                metrics = model.fit(X, y, X_val, y_val)
                individual_metrics[model_name] = metrics
                trained_models.append((model_name, model))
            except Exception as e:
                self.logger.warning(f"Failed to train {model_name}: {e}")
        
        self.base_models = trained_models
        
        # Create ensemble
        if self.is_classification():
            estimators = [(name, model.model) for name, model in self.base_models]
            voting = self.config.model.ensemble_params.get('voting', 'soft')
            weights = self.config.model.ensemble_params.get('weights')
            
            self.model = VotingClassifier(
                estimators=estimators,
                voting=voting,
                weights=weights
            )
        else:
            estimators = [(name, model.model) for name, model in self.base_models]
            weights = self.config.model.ensemble_params.get('weights')
            
            self.model = VotingRegressor(
                estimators=estimators,
                weights=weights
            )
        
        # Ensemble is already trained since base models are trained
        self.is_trained = True
        
        # Calculate ensemble metrics
        train_pred = self.predict(X)
        metrics = {
            'train_samples': len(X),
            'base_models': list(individual_metrics.keys()),
            'individual_metrics': individual_metrics
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
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
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get predictions from each base model
        predictions = []
        for model_name, model in self.base_models:
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"Failed to get predictions from {model_name}: {e}")
        
        if not predictions:
            raise ValueError("No base models provided valid predictions")
        
        # Ensemble predictions
        predictions = np.array(predictions)
        
        if self.is_classification():
            # Majority voting for classification
            ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
        else:
            # Average for regression
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for ensemble classification."""
        if not self.is_classification():
            raise ValueError("Probabilities only available for classification models")
        
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Get probabilities from each base model
        probabilities = []
        for model_name, model in self.base_models:
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    probabilities.append(proba)
            except Exception as e:
                self.logger.warning(f"Failed to get probabilities from {model_name}: {e}")
        
        if not probabilities:
            raise ValueError("No base models provided valid probabilities")
        
        # Average probabilities
        ensemble_proba = np.mean(probabilities, axis=0)
        return ensemble_proba
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get aggregated feature importance from base models."""
        if not self.is_trained:
            return None
        
        all_importance = []
        for model_name, model in self.base_models:
            importance = model.get_feature_importance()
            if importance:
                all_importance.append(importance)
        
        if not all_importance:
            return None
        
        # Average importance scores
        feature_names = list(all_importance[0].keys())
        aggregated_importance = {}
        
        for feature in feature_names:
            scores = [imp.get(feature, 0) for imp in all_importance]
            aggregated_importance[feature] = np.mean(scores)
        
        return aggregated_importance
    
    def save(self, filepath: str):
        """Save ensemble model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        # Save base models
        base_model_data = []
        for model_name, model in self.base_models:
            # For base models, we save their state
            base_model_data.append({
                'name': model_name,
                'model': model.model,
                'training_metrics': model.training_metrics
            })
        
        model_data = {
            'base_models': base_model_data,
            'config': self.config.to_dict(),
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Ensemble model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load ensemble model."""
        model_data = joblib.load(filepath)
        
        # Recreate base models
        self.base_models = []
        for base_data in model_data['base_models']:
            # Create model wrapper
            temp_config = Config.from_dict(model_data['config'])
            model = ModelFactory.create_model(temp_config)
            model.model = base_data['model']
            model.is_trained = True
            model.training_metrics = base_data.get('training_metrics', {})
            
            self.base_models.append((base_data['name'], model))
        
        self.is_trained = model_data.get('is_trained', True)
        self.training_metrics = model_data.get('training_metrics', {})
        
        self.logger.info(f"Ensemble model loaded from {filepath}")