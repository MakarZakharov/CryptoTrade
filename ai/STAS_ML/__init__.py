"""
STAS_ML - Модуль машинного обучения для торговли криптовалютами.
Простые и эффективные ML модели для прогнозирования цен и торговых сигналов.
"""

__version__ = "1.0.0"
__author__ = "STAS ML Team"

from .config.ml_config import MLConfig
from .training.trainer import MLTrainer
from .models.predictor import CryptoPricePredictor

__all__ = [
    'MLConfig',
    'MLTrainer', 
    'CryptoPricePredictor'
]