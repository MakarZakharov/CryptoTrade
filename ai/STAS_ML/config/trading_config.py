"""
Конфігурація для торгового середовища STAS_ML.
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TradingConfig:
    """Конфігурація торгового середовища."""
    
    # Дані
    exchange: str = 'binance'
    symbol: str = 'BTCUSDT'
    timeframe: str = '1d'  # '1d', '4h'
    data_path: Optional[str] = None
    
    # Торгові параметри
    initial_balance: float = 10000.0  # USDT ($10k starting balance)
    commission_rate: float = 0.001  # 0.1% як на Binance Spot
    slippage_rate: float = 0.0005  # 0.05% проскальзування
    spread_rate: float = 0.0002  # 0.02% спред bid/ask
    
    # Управління капіталом
    min_trade_amount: float = 10.0  # Мінімальна сума угоди в USDT
    max_position_size: float = 1.0  # Максимальна позиція (100% капіталу)
    
    # НОВЕ: Управління ризиками
    enable_position_sizing: bool = True  # Увімкнути динамічне управління розміром позицій
    max_risk_per_trade: float = 0.02  # Максимальний ризик на угоду (2% від капіталу)
    position_size_method: str = 'kelly'  # 'fixed', 'kelly', 'volatility_based'
    
    # Stop-Loss параметри
    enable_stop_loss: bool = True  # Увімкнути stop-loss
    stop_loss_type: str = 'percentage'  # 'percentage', 'atr', 'trailing'
    stop_loss_percentage: float = 0.05  # 5% stop-loss
    trailing_stop_percentage: float = 0.03  # 3% trailing stop
    atr_multiplier: float = 2.0  # Множник ATR для stop-loss
    
    # Контроль просадки
    max_drawdown_limit: float = 0.15  # Максимальна допустима просадка (15%)
    reduce_position_on_drawdown: bool = True  # Зменшувати позиції при просадці
    
    # Ліквідність та виконання
    enable_partial_fills: bool = True
    liquidity_impact_threshold: float = 0.001  # Поріг впливу на ліквідність
    max_order_size_ratio: float = 0.1  # Максимальний розмір ордеру відносно обсягу
    
    # Технічні індикатори - ОПТИМІЗОВАНО для швидкості
    include_technical_indicators: bool = True
    indicator_periods: Dict[str, List[int]] = field(default_factory=lambda: {
        'sma': [20],  # Тільки один SMA
        'ema': [20],  # Тільки один EMA
        'rsi': [14],  # RSI
        'macd': [12, 26, 9],  # MACD (важливий)
        'bollinger': [20],  # Bollinger Bands
        'atr': [14],  # Волатильність
        # Видалено менш важливі індикатори для швидкості:
        # 'adx', 'momentum', 'stochastic', 'williams_r', 'obv', 'ichimoku', 'vwap'
    })
    
    # Навчання - ОПТИМІЗОВАНО для швидкості
    lookback_window: int = 20  # Зменшено вікно для швидшого обчислення
    
    # Винагороди
    reward_scheme: str = 'default'  # 'default', 'conservative', 'aggressive', 'optimized', 'custom'
    custom_reward_weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.data_path is None:
            self.data_path = os.path.join(
                'data', self.exchange, self.symbol, 
                self.timeframe, '2018_01_01-now.csv'
            )


class DataManager:
    """Менеджер даних для торгового середовища."""
    
    @staticmethod
    def get_available_pairs() -> Dict[str, List[str]]:
        """Отримати доступні торгові пари по біржам."""
        data_root = os.path.join('data')
        available_pairs = {}
        
        if not os.path.exists(data_root):
            return available_pairs
        
        for exchange in os.listdir(data_root):
            exchange_path = os.path.join(data_root, exchange)
            if not os.path.isdir(exchange_path):
                continue
                
            pairs = []
            for symbol in os.listdir(exchange_path):
                symbol_path = os.path.join(exchange_path, symbol)
                if os.path.isdir(symbol_path):
                    pairs.append(symbol)
            
            if pairs:
                available_pairs[exchange] = sorted(pairs)
        
        return available_pairs
    
    @staticmethod
    def get_available_timeframes(exchange: str, symbol: str) -> List[str]:
        """Отримати доступні таймфрейми для пари."""
        symbol_path = os.path.join('data', exchange, symbol)
        timeframes = []
        
        if os.path.exists(symbol_path):
            for timeframe in os.listdir(symbol_path):
                timeframe_path = os.path.join(symbol_path, timeframe)
                if os.path.isdir(timeframe_path):
                    timeframes.append(timeframe)
        
        return sorted(timeframes)


def interactive_config_creator() -> TradingConfig:
    """Інтерактивне створення конфігурації."""
    print("=== Створення конфігурації торгового середовища ===\n")
    
    # Отримуємо доступні дані
    available_pairs = DataManager.get_available_pairs()
    
    # Вибір біржі
    print("Доступні біржі:")
    exchanges = list(available_pairs.keys())
    for i, exchange in enumerate(exchanges, 1):
        print(f"{i}. {exchange}")
    
    while True:
        try:
            choice = int(input(f"Оберіть біржу (1-{len(exchanges)}): ")) - 1
            if 0 <= choice < len(exchanges):
                selected_exchange = exchanges[choice]
                break
        except ValueError:
            pass
        print("Невірний вибір!")
    
    # Вибір пари
    print(f"\nДоступні пари на {selected_exchange}:")
    pairs = available_pairs[selected_exchange]
    for i, pair in enumerate(pairs, 1):
        print(f"{i}. {pair}")
    
    while True:
        try:
            choice = int(input(f"Оберіть пару (1-{len(pairs)}): ")) - 1
            if 0 <= choice < len(pairs):
                selected_pair = pairs[choice]
                break
        except ValueError:
            pass
        print("Невірний вибір!")
    
    # Вибір таймфрейму
    print(f"\nДоступні таймфрейми для {selected_pair}:")
    timeframes = DataManager.get_available_timeframes(selected_exchange, selected_pair)
    for i, timeframe in enumerate(timeframes, 1):
        print(f"{i}. {timeframe}")
    
    while True:
        try:
            choice = int(input(f"Оберіть таймфрейм (1-{len(timeframes)}): ")) - 1
            if 0 <= choice < len(timeframes):
                selected_timeframe = timeframes[choice]
                break
        except ValueError:
            pass
        print("Невірний вибір!")
    
    # Створюємо конфігурацію
    config = TradingConfig(
        exchange=selected_exchange,
        symbol=selected_pair,
        timeframe=selected_timeframe,
        reward_scheme='optimized',
        initial_balance=10000.0
    )
    
    print(f"\n=== Створена конфігурація ===")
    print(f"Біржа: {config.exchange}")
    print(f"Пара: {config.symbol}")
    print(f"Таймфрейм: {config.timeframe}")
    print(f"Схема винагород: {config.reward_scheme}")
    print(f"Початковий капітал: {config.initial_balance} USDT")
    
    return config