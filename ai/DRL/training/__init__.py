"""Система обучения DRL агентов."""

from .trainer import Trainer
from .experiment_manager import ExperimentManager

# Callbacks будут добавлены позже
try:
    from .callbacks import TradingCallback, EvaluationCallback, CheckpointCallback
    __all__ = [
        'Trainer',
        'ExperimentManager', 
        'TradingCallback',
        'EvaluationCallback',
        'CheckpointCallback'
    ]
except ImportError:
    __all__ = [
        'Trainer',
        'ExperimentManager'
    ]