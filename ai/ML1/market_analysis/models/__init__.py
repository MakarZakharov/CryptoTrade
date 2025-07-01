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
    # Add parent directory to path for direct script execution
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from market_analysis.models.base_model import BaseModel
    from market_analysis.models.lstm_model import LSTMModel
    from market_analysis.models.gru_model import GRUModel
    from market_analysis.models.transformer_model import TransformerModel, TransformerBlock
    from market_analysis.models.xgboost_model import XGBoostModel
    from market_analysis.models.ensemble import StackingEnsembleModel, VotingEnsembleModel
    from market_analysis.models.model_factory import ModelFactory

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