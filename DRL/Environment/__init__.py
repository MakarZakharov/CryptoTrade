"""
DRL Trading Environment Package

OpenAI Gym compatible environment for cryptocurrency trading with DRL agents.
"""

from .crypto_trading_env import CryptoTradingEnv
from .config import EnvConfig, get_config
from .indicators import TechnicalIndicators, compute_indicators, normalize_features

__all__ = [
    'CryptoTradingEnv',
    'EnvConfig',
    'get_config',
    'TechnicalIndicators',
    'compute_indicators',
    'normalize_features'
]

__version__ = '1.0.0'
