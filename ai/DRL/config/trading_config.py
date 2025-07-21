"""Торговые настройки для DRL системы."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TradingConfig:
    """Конфигурация торговых параметров."""
    
    # Торговые инструменты
    symbol: str = "BTCUSDT"  # Торговая пара
    exchange: str = "binance"  # Биржа
    timeframe: str = "1d"  # Таймфрейм (1h, 4h, 1d)
    
    # Капитал и управление рисками
    initial_balance: float = 10000.0  # Начальный баланс в USDT
    max_risk_per_trade: float = 0.02  # Максимальный риск на сделку (2%)
    max_position_size: float = 0.8  # Максимальный размер позиции (80% от капитала)
    min_trade_amount: float = 10.0  # Минимальная сумма сделки в USDT
    
    # Комиссии и проскальзывание
    commission_rate: float = 0.001  # Комиссия торговли (0.1%)
    slippage_rate: float = 0.0005  # Проскальзывание (0.05%)
    spread_rate: float = 0.0002  # Спред bid/ask (0.02%)
    
    # Стоп-лосс и тейк-профит
    enable_stop_loss: bool = True  # Включить стоп-лосс
    stop_loss_pct: float = 0.05  # Стоп-лосс в процентах (5%)
    enable_take_profit: bool = True  # Включить тейк-профит
    take_profit_pct: float = 0.15  # Тейк-профит в процентах (15%)
    trailing_stop: bool = False  # Trailing stop
    trailing_stop_pct: float = 0.03  # Trailing stop в процентах (3%)
    
    # Контроль просадки
    max_drawdown_limit: float = 0.20  # Максимально допустимая просадка (20%)
    drawdown_protection: bool = True  # Защита от просадки
    position_size_on_drawdown: float = 0.5  # Уменьшение позиций при просадке
    
    # Данные и индикаторы
    data_dir: str = "CryptoTrade/data"  # Директория с данными
    lookback_window: int = 50  # Размер окна наблюдения
    include_technical_indicators: bool = True  # Включить технические индикаторы
    technical_indicators: List[str] = field(default_factory=lambda: [
        "sma_20", "ema_20", "rsi_14", "macd", "bollinger_bands", "atr_14"
    ])  # Список технических индикаторов
    
    # Нормализация данных
    normalize_features: bool = True  # Нормализовать фичи
    normalization_method: str = "zscore"  # minmax, zscore
    
    # Разделение данных
    train_split: float = 0.7  # Доля данных для обучения
    val_split: float = 0.2  # Доля данных для валидации
    test_split: float = 0.1  # Доля данных для тестирования
    
    # Целевая доходность
    target_monthly_return: float = 0.10  # Целевая месячная доходность (10%)
    benchmark_symbol: str = "BTC"  # Символ для бенчмарка (buy & hold)
    
    # Действия агента
    action_type: str = "continuous"  # continuous, discrete
    action_bounds: Tuple[float, float] = (-1.0, 1.0)  # Границы действий для continuous
    discrete_actions: List[str] = field(default_factory=lambda: ["buy", "sell", "hold"])  # Для discrete
    
    # Reward система
    reward_scheme: str = "profit_based"  # profit_based, sharpe_based, risk_adjusted
    reward_scaling: float = 1.0  # Масштабирование награды
    penalty_for_inaction: float = 0.001  # Штраф за бездействие
    
    # Условия завершения эпизода
    max_episode_steps: int = 1000  # Максимальное количество шагов в эпизоде
    early_stopping_loss: float = 0.5  # Остановка при потере 50% капитала
    
    # Особенности криптовалютного рынка
    market_hours: str = "24/7"  # Время работы рынка
    weekend_trading: bool = True  # Торговля в выходные
    handle_gaps: bool = True  # Обработка гэпов в данных
    
    # Валидация данных
    min_data_points: int = 1000  # Минимальное количество точек данных
    data_quality_threshold: float = 0.95  # Порог качества данных
    
    def __post_init__(self):
        """Валидация конфигурации после инициализации."""
        if self.initial_balance <= 0:
            raise ValueError("initial_balance должен быть положительным")
        
        if not (0 < self.max_risk_per_trade <= 1):
            raise ValueError("max_risk_per_trade должен быть в диапазоне (0, 1]")
        
        if not (0 < self.max_position_size <= 1):
            raise ValueError("max_position_size должен быть в диапазоне (0, 1]")
        
        if self.lookback_window <= 0:
            raise ValueError("lookback_window должен быть положительным")
        
        if abs(self.train_split + self.val_split + self.test_split - 1.0) > 1e-6:
            raise ValueError("Сумма train_split + val_split + test_split должна быть равна 1.0")
        
        if self.target_monthly_return <= 0:
            raise ValueError("target_monthly_return должен быть положительным")
        
        if self.action_type not in ["continuous", "discrete"]:
            raise ValueError(f"Неподдерживаемый тип действий: {self.action_type}")
        
        if self.reward_scheme not in ["profit_based", "sharpe_based", "risk_adjusted"]:
            raise ValueError(f"Неподдерживаемая схема наград: {self.reward_scheme}")
    
    def get_observation_space_size(self) -> int:
        """Получить размер пространства наблюдений."""
        base_features = 5  # OHLCV
        tech_indicators = len(self.technical_indicators) if self.include_technical_indicators else 0
        portfolio_features = 3  # баланс USDT, баланс криптовалюты, общая стоимость
        
        return (base_features + tech_indicators + portfolio_features) * self.lookback_window
    
    def get_action_space_info(self) -> Dict:
        """Получить информацию о пространстве действий."""
        if self.action_type == "continuous":
            return {
                "type": "continuous",
                "shape": (1,),
                "low": self.action_bounds[0],
                "high": self.action_bounds[1]
            }
        else:
            return {
                "type": "discrete",
                "n": len(self.discrete_actions),
                "actions": self.discrete_actions
            }
    
    def get_data_path(self) -> str:
        """Получить путь к данным."""
        return f"CryptoTrade/data/{self.exchange}/{self.symbol}/{self.timeframe}"
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Получить лимиты рисков."""
        return {
            "max_risk_per_trade": self.max_risk_per_trade,
            "max_position_size": self.max_position_size,
            "max_drawdown_limit": self.max_drawdown_limit,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "early_stopping_loss": self.early_stopping_loss
        }