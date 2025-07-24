"""
Model validation for STAS_ML v2
"""

import numpy as np
from typing import Dict, Any
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, cross_val_score

from ..core.config import Config
from ..core.base import BaseValidator, BaseModel, MetricsCalculator, Logger


class ModelValidator(BaseValidator):
    """Model validator with comprehensive evaluation."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.logger = Logger("ModelValidator")
    
    def validate(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Validate model on test data."""
        predictions = model.predict(X)
        
        if model.is_classification():
            metrics = MetricsCalculator.classification_metrics(y, predictions)
            
            # Add probability-based metrics if available
            try:
                probabilities = model.predict_proba(X)
                if probabilities is not None:
                    metrics.update(MetricsCalculator.classification_metrics(y, predictions, probabilities))
            except:
                pass
        else:
            metrics = MetricsCalculator.regression_metrics(y, predictions)
        
        return metrics
    
    def cross_validate(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation."""
        strategy = self.config.validation.cv_strategy
        folds = self.config.validation.cv_folds
        
        if strategy == "time_series":
            cv = TimeSeriesSplit(n_splits=folds)
        elif strategy == "stratified" and model.is_classification():
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.config.random_seed)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=folds, shuffle=True, random_state=self.config.random_seed)
        
        # Determine scoring
        if model.is_classification():
            scoring = 'accuracy'
        else:
            scoring = 'neg_mean_squared_error'
        
        # Perform cross-validation
        try:
            scores = cross_val_score(model.model, X, y, cv=cv, scoring=scoring)
            
            results = {
                'cv_scores': scores.tolist(),
                'cv_mean': scores.mean(),
                'cv_std': scores.std(),
                'cv_strategy': strategy,
                'scoring': scoring
            }
            
            self.logger.info(f"Cross-validation {scoring}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {e}")
            results = {'error': str(e)}
        
        return results