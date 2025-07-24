"""
STAS_ML v2 - Redesigned ML System for Crypto Trading
====================================================

Clean, modular, and efficient machine learning system for cryptocurrency trading.

Key Features:
- Modular architecture with clear separation of concerns
- Advanced feature engineering pipeline
- Robust model validation and testing
- Comprehensive backtesting framework
- Model versioning and experiment tracking
- Production-ready deployment capabilities

Main Components:
- core/: Core business logic and base classes
- data/: Advanced data processing and feature engineering
- models/: ML model implementations and ensembles
- validation/: Cross-validation and model evaluation
- backtesting/: Trading strategy backtesting
- deployment/: Model deployment and serving
- experiments/: Experiment tracking and management
"""

from .core.config import Config
from .core.trainer import ModelTrainer
from .data.processor import DataProcessor
from .models.ensemble import EnsembleModel
from .validation.validator import ModelValidator
from .backtesting.engine import BacktestEngine

__version__ = "2.0.0"
__author__ = "STAS ML Team"

__all__ = [
    'Config',
    'ModelTrainer', 
    'DataProcessor',
    'EnsembleModel',
    'ModelValidator',
    'BacktestEngine'
]