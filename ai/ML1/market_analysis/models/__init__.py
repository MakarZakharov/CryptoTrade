"""
Models package for market analysis.

This package contains various model implementations for time series prediction,
including neural network models (LSTM, GRU, Transformer) and traditional
machine learning models (XGBoost), as well as ensemble models that combine
multiple base models.
"""

import os
import sys

# Handle both package import and direct script execution
try:
    from .base_model import BaseModel
    from .lstm_model import LSTMModel
    from .gru_model import GRUModel
    from .transformer_model import TransformerModel, TransformerBlock
    from .xgboost_model import XGBoostModel
    from .ensemble import StackingEnsembleModel, VotingEnsembleModel
    from .model_factory import ModelFactory
except ImportError:
    # Add current directory to path for direct script execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ensemble_dir = os.path.join(current_dir, 'ensemble')
    sys.path.append(current_dir)
    sys.path.append(ensemble_dir)
    
    from base_model import BaseModel
    from lstm_model import LSTMModel
    from gru_model import GRUModel
    from transformer_model import TransformerModel, TransformerBlock
    from xgboost_model import XGBoostModel
    # Import from ensemble directory
    from ensemble.stacking import StackingEnsembleModel
    from ensemble.voting import VotingEnsembleModel
    from model_factory import ModelFactory

__all__ = [
    'BaseModel',
    'LSTMModel',
    'GRUModel',
    'TransformerModel',
    'TransformerBlock',
    'XGBoostModel',
    'StackingEnsembleModel',
    'VotingEnsembleModel',
    'ModelFactory'
]