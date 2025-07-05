"""
Конфигурация для торговой среды DRL.
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class TradingConfig:
    """Конфигурация торговой среды."""
    
    # Данные
    exchange: str = 'binance'
    symbol: str = 'BTCUSDT'
    timeframe: str = '1d'  # '1d', '4h'
    data_path: Optional[str] = None
    
    # Торговые параметры
    initial_balance: float = 100.0  # USDT
    commission_rate: float = 0.001  # 0.1% как на Binance Spot
    slippage_rate: float = 0.0005  # 0.05% проскальзывание
    spread_rate: float = 0.0002  # 0.02% спред bid/ask
    
    # Управление капиталом
    min_trade_amount: float = 10.0  # Минимальная сумма сделки в USDT
    max_position_size: float = 1.0  # Максимальная позиция (100% капитала)
    
    # Ликвидность и исполнение
    enable_partial_fills: bool = True
    liquidity_impact_threshold: float = 0.001  # Порог влияния на ликвидность
    max_order_size_ratio: float = 0.1  # Максимальный размер ордера относительно объема
    
    # Технические индикаторы
    include_technical_indicators: bool = True
    indicator_periods: Dict[str, List[int]] = field(default_factory=lambda: {
        'sma': [5, 20, 50],
        'ema': [5, 20, 50],
        'rsi': [14],
        'macd': [12, 26, 9],
        'bollinger': [20],
        'atr': [14],
        'adx': [14],
        'momentum': [5, 10, 20],
        'stochastic': [14],
        'williams_r': [14],
        'obv': [],
        'ichimoku': [9, 26, 52],
        'vwap': []
    })
    
    # Обучение
    lookback_window: int = 50  # Размер окна наблюдения
    
    # Награды
    reward_scheme: str = 'default'  # 'default', 'conservative', 'aggressive', 'optimized', 'custom'
    custom_reward_weights: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.data_path is None:
            self.data_path = os.path.join(
                'CryptoTrade', 'data', self.exchange, self.symbol, 
                self.timeframe, '2018_01_01-now.csv'
            )


class DataManager:
    """Менеджер данных для торговой среды."""
    
    @staticmethod
    def get_available_pairs() -> Dict[str, List[str]]:
        """Получить доступные торговые пары по биржам."""
        data_root = os.path.join('CryptoTrade', 'data')
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
        """Получить доступные таймфреймы для пары."""
        symbol_path = os.path.join('CryptoTrade', 'data', exchange, symbol)
        timeframes = []
        
        if os.path.exists(symbol_path):
            for timeframe in os.listdir(symbol_path):
                timeframe_path = os.path.join(symbol_path, timeframe)
                if os.path.isdir(timeframe_path):
                    timeframes.append(timeframe)
        
        return sorted(timeframes)
    
    @staticmethod
    def validate_config(config: TradingConfig) -> bool:
        """Проверить корректность конфигурации."""
        available_pairs = DataManager.get_available_pairs()
        
        if config.exchange not in available_pairs:
            print(f"Биржа {config.exchange} недоступна")
            return False
        
        if config.symbol not in available_pairs[config.exchange]:
            print(f"Пара {config.symbol} недоступна на бирже {config.exchange}")
            return False
        
        available_timeframes = DataManager.get_available_timeframes(
            config.exchange, config.symbol
        )
        
        if config.timeframe not in available_timeframes:
            print(f"Таймфрейм {config.timeframe} недоступен для {config.symbol}")
            return False
        
        if not os.path.exists(config.data_path):
            print(f"Файл данных не найден: {config.data_path}")
            return False
        
        return True


def create_config_selector():
    """Создать интерактивный селектор конфигурации."""
    available_pairs = DataManager.get_available_pairs()
    
    print("Доступные биржи и пары:")
    for exchange, pairs in available_pairs.items():
        print(f"\n{exchange}:")
        for pair in pairs:
            timeframes = DataManager.get_available_timeframes(exchange, pair)
            print(f"  {pair}: {timeframes}")
    
    print("\nПример создания конфигурации:")
    print("config = TradingConfig(")
    print("    exchange='binance',")
    print("    symbol='BTCUSDT',")
    print("    timeframe='1d',")
    print("    initial_balance=100.0")
    print(")")
    
    return available_pairs


def interactive_config_creator() -> TradingConfig:
    """Интерактивное создание конфигурации."""
    print("=== Создание конфигурации торговой среды ===\n")
    
    # Получаем доступные данные
    available_pairs = DataManager.get_available_pairs()
    
    # Выбор биржи
    print("Доступные биржи:")
    exchanges = list(available_pairs.keys())
    for i, exchange in enumerate(exchanges, 1):
        print(f"{i}. {exchange}")
    
    while True:
        try:
            choice = int(input(f"Выберите биржу (1-{len(exchanges)}): ")) - 1
            if 0 <= choice < len(exchanges):
                selected_exchange = exchanges[choice]
                break
        except ValueError:
            pass
        print("Неверный выбор!")
    
    # Выбор пары
    print(f"\nДоступные пары на {selected_exchange}:")
    pairs = available_pairs[selected_exchange]
    for i, pair in enumerate(pairs, 1):
        print(f"{i}. {pair}")
    
    while True:
        try:
            choice = int(input(f"Выберите пару (1-{len(pairs)}): ")) - 1
            if 0 <= choice < len(pairs):
                selected_pair = pairs[choice]
                break
        except ValueError:
            pass
        print("Неверный выбор!")
    
    # Выбор таймфрейма
    print(f"\nДоступные таймфреймы для {selected_pair}:")
    timeframes = DataManager.get_available_timeframes(selected_exchange, selected_pair)
    for i, timeframe in enumerate(timeframes, 1):
        print(f"{i}. {timeframe}")
    
    while True:
        try:
            choice = int(input(f"Выберите таймфрейм (1-{len(timeframes)}): ")) - 1
            if 0 <= choice < len(timeframes):
                selected_timeframe = timeframes[choice]
                break
        except ValueError:
            pass
        print("Неверный выбор!")
    
    # Выбор схемы наград
    print("\nДоступные схемы наград:")
    reward_schemes = ['default', 'conservative', 'aggressive', 'optimized', 'custom']
    scheme_descriptions = [
        'default - сбалансированная схема',
        'conservative - акцент на стабильность',
        'aggressive - акцент на прибыль',
        'optimized - максимальная прибыль с контролем рисков (win rate >60%, drawdown <20%)',
        'custom - настраиваемые веса'
    ]
    
    for i, (scheme, desc) in enumerate(zip(reward_schemes, scheme_descriptions), 1):
        print(f"{i}. {desc}")
    
    while True:
        try:
            choice = int(input(f"Выберите схему наград (1-{len(reward_schemes)}): ")) - 1
            if 0 <= choice < len(reward_schemes):
                selected_reward = reward_schemes[choice]
                break
        except ValueError:
            pass
        print("Неверный выбор!")
    
    # Создаем конфигурацию
    config = TradingConfig(
        exchange=selected_exchange,
        symbol=selected_pair,
        timeframe=selected_timeframe,
        reward_scheme=selected_reward,
        initial_balance=100.0
    )
    
    print(f"\n=== Создана конфигурация ===")
    print(f"Биржа: {config.exchange}")
    print(f"Пара: {config.symbol}")
    print(f"Таймфрейм: {config.timeframe}")
    print(f"Схема наград: {config.reward_scheme}")
    print(f"Начальный капитал: {config.initial_balance} USDT")
    print(f"Комиссия: {config.commission_rate*100:.2f}%")
    print(f"Проскальзывание: {config.slippage_rate*100:.3f}%")
    print(f"Спред: {config.spread_rate*100:.3f}%")
    
    # Проверяем корректность
    if DataManager.validate_config(config):
        print("✅ Конфигурация валидна!")
        return config
    else:
        print("❌ Ошибка в конфигурации!")
        return None


def create_multiple_configs(pairs: List[str] = None, timeframes: List[str] = None) -> List[TradingConfig]:
    """Создать несколько конфигураций для массового тестирования."""
    configs = []
    available_pairs = DataManager.get_available_pairs()
    
    # Используем все доступные пары если не указаны
    if pairs is None:
        pairs = []
        for exchange_pairs in available_pairs.values():
            pairs.extend(exchange_pairs)
    
    # Используем популярные таймфреймы если не указаны
    if timeframes is None:
        timeframes = ['1d', '4h', '1h']
    
    for exchange, exchange_pairs in available_pairs.items():
        for pair in pairs:
            if pair in exchange_pairs:
                available_timeframes = DataManager.get_available_timeframes(exchange, pair)
                for timeframe in timeframes:
                    if timeframe in available_timeframes:
                        config = TradingConfig(
                            exchange=exchange,
                            symbol=pair,
                            timeframe=timeframe,
                            initial_balance=100.0
                        )
                        if DataManager.validate_config(config):
                            configs.append(config)
    
    return configs


def get_popular_configs() -> List[TradingConfig]:
    """Получить популярные предустановленные конфигурации."""
    popular_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    popular_timeframes = ['1d', '4h']
    
    configs = []
    available_pairs = DataManager.get_available_pairs()
    
    for exchange, pairs in available_pairs.items():
        for pair in popular_pairs:
            if pair in pairs:
                for timeframe in popular_timeframes:
                    available_timeframes = DataManager.get_available_timeframes(exchange, pair)
                    if timeframe in available_timeframes:
                        config = TradingConfig(
                            exchange=exchange,
                            symbol=pair,
                            timeframe=timeframe,
                            initial_balance=100.0
                        )
                        if DataManager.validate_config(config):
                            configs.append(config)
    
    return configs


if __name__ == "__main__":
    # Показать доступные данные
    create_config_selector()