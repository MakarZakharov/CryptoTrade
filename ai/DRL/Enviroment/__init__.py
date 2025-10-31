"""
DRL Торговое Окружение для Криптовалют
Полнофункциональная система для обучения DRL агентов
Совместимо с Gymnasium и Stable-Baselines3
"""

# Основное окружение
from .env import CryptoTradingEnv, ActionSpace, RewardType

# Загрузка данных
from .data_loader import DataLoader

# Симуляция рынка
from .simulator import (
    MarketSimulator,
    OrderSide,
    OrderType,
    SlippageModel,
    OrderResult,
    MarketState
)

# Метрики
from .metrics import (
    MetricsCalculator,
    PerformanceMetrics,
    TradeMetrics
)

# Визуализация
from .visualization import TradingVisualizer

__version__ = "1.0.0"

__all__ = [
    # Environment
    'CryptoTradingEnv',
    'ActionSpace',
    'RewardType',

    # Data
    'DataLoader',

    # Simulator
    'MarketSimulator',
    'OrderSide',
    'OrderType',
    'SlippageModel',
    'OrderResult',
    'MarketState',

    # Metrics
    'MetricsCalculator',
    'PerformanceMetrics',
    'TradeMetrics',

    # Visualization
    'TradingVisualizer',
]
