"""DRL (Deep Reinforcement Learning) система для криптовалютной торговли.

Этот модуль содержит полную DRL систему для обучения торговых агентов.
Включает в себя:
- Обработку данных и технические индикаторы
- Торговые среды (environments)
- DRL агентов (PPO, DQN, SAC)
- Систему обучения и оценки
"""

__version__ = "1.0.0"

# Основные импорты
from .config import DRLConfig, TradingConfig
from .utils import DRLLogger, TradingMetrics
from .data import CSVDataLoader, TechnicalIndicators, DataPreprocessor