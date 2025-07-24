"""
Base classes for STAS_ML v2

Provides abstract base classes and common functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from .config import Config


class BaseModel(ABC):
    """Abstract base class for all ML models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.training_metrics = {}
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities (for classification models)."""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        pass
    
    @abstractmethod
    def save(self, filepath: str):
        """Save model to file."""
        pass
    
    @abstractmethod
    def load(self, filepath: str):
        """Load model from file."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return getattr(self.model, 'get_params', lambda: {})()
    
    def set_params(self, **params):
        """Set model parameters."""
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
    
    def is_classification(self) -> bool:
        """Check if this is a classification model."""
        return self.config.model.target_type.value == "direction"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'model_type': self.config.model.model_type.value,
            'target_type': self.config.model.target_type.value,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'training_metrics': self.training_metrics
        }


class BasePreprocessor(ABC):
    """Abstract base class for data preprocessors."""
    
    def __init__(self, config: Config):
        self.config = config
        self.is_fitted = False
        self.feature_names = []
        self.preprocessing_params = {}
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> 'BasePreprocessor':
        """Fit the preprocessor to data."""
        pass
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        pass
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform data."""
        return self.fit(data).transform(data)
    
    @abstractmethod
    def get_feature_names(self) -> list:
        """Get output feature names."""
        pass
    
    def save_params(self, filepath: str):
        """Save preprocessing parameters."""
        import joblib
        joblib.dump({
            'preprocessing_params': self.preprocessing_params,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }, filepath)
    
    def load_params(self, filepath: str):
        """Load preprocessing parameters."""
        import joblib
        data = joblib.load(filepath)
        self.preprocessing_params = data['preprocessing_params']
        self.feature_names = data['feature_names']
        self.is_fitted = data['is_fitted']


class BaseValidator(ABC):
    """Abstract base class for model validators."""
    
    def __init__(self, config: Config):
        self.config = config
        self.validation_results = {}
    
    @abstractmethod
    def validate(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Validate model performance."""
        pass
    
    @abstractmethod
    def cross_validate(self, model: BaseModel, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation."""
        pass
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        return self.validation_results


class BaseBacktester(ABC):
    """Abstract base class for backtesting engines."""
    
    def __init__(self, config: Config):
        self.config = config
        self.backtest_results = {}
    
    @abstractmethod
    def run_backtest(self, model: BaseModel, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest."""
        pass
    
    @abstractmethod
    def calculate_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics."""
        pass
    
    def get_backtest_summary(self) -> Dict[str, Any]:
        """Get backtest summary."""
        return self.backtest_results


class ModelFactory:
    """Factory for creating models."""
    
    @staticmethod
    def create_model(config: Config) -> BaseModel:
        """Create model based on config."""
        from ..models.xgboost_model import XGBoostModel
        from ..models.rf_model import RandomForestModel
        from ..models.lstm_model import LSTMModel
        from ..models.ensemble import EnsembleModel
        from ..models.linear_model import LinearModel
        
        model_map = {
            'xgboost': XGBoostModel,
            'random_forest': RandomForestModel,
            'lstm': LSTMModel,
            'ensemble': EnsembleModel,
            'linear': LinearModel
        }
        
        model_type = config.model.model_type.value
        if model_type not in model_map:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model_map[model_type](config)


class MetricsCalculator:
    """Utility class for calculating various metrics."""
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                             y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            confusion_matrix, classification_report, roc_auc_score
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_proba is not None and y_proba.shape[1] == 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except ValueError:
                metrics['roc_auc'] = 0.5
        
        return metrics
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
    
    @staticmethod
    def trading_metrics(returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """Calculate trading performance metrics."""
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** 252 - 1  # Assuming daily returns
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio
        }
        
        # Add benchmark comparison if provided
        if benchmark_returns is not None:
            benchmark_total = (1 + benchmark_returns).prod() - 1
            metrics['excess_return'] = total_return - benchmark_total
            
            # Beta calculation
            if len(benchmark_returns) == len(returns):
                covariance = np.cov(returns, benchmark_returns)[0][1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
                metrics['beta'] = beta
        
        return metrics


class Logger:
    """Simple logging utility."""
    
    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.level = level
    
    def info(self, message: str):
        print(f"[INFO] {self.name}: {message}")
    
    def warning(self, message: str):
        print(f"[WARNING] {self.name}: {message}")
    
    def error(self, message: str):
        print(f"[ERROR] {self.name}: {message}")
    
    def debug(self, message: str):
        if self.level == "DEBUG":
            print(f"[DEBUG] {self.name}: {message}")