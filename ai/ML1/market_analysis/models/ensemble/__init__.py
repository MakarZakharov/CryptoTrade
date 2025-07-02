"""
Ensemble models package for market analysis.

This package contains implementations of ensemble models that combine
multiple base models to improve prediction performance, including:
- Stacking ensemble: Uses a meta-model to combine base model predictions
- Voting ensemble: Combines predictions using weighted averaging
"""

from .stacking import StackingEnsembleModel
from .voting import VotingEnsembleModel

__all__ = [
    'StackingEnsembleModel',
    'VotingEnsembleModel'
]