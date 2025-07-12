"""
Конфигурация для ML модулей STAS_ML.
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass 
class MLConfig:
    """Конфигурация ML модели для торговли криптовалютами."""
    
    # Данные
    exchange: str = 'binance'
    symbol: str = 'BTCUSDT'
    timeframe: str = '1d'
    data_path: Optional[str] = None
    
    # Параметры модели - ОПТИМІЗОВАНО для стабільності
    model_type: str = 'xgboost'  # 'xgboost', 'random_forest', 'lstm', 'linear'
    prediction_horizon: int = 1  # Сколько периодов вперед предсказывать
    lookback_window: int = 50    # ЗБІЛЬШЕНО для кращого розуміння контексту (було 30)
    
    # Технические индикаторы - СПРОЩЕНО: тільки RSI та MACD
    include_technical_indicators: bool = True
    indicator_periods: Dict[str, List[int]] = field(default_factory=lambda: {


        'rsi': [14],              # RSI з стандартним 14-денним періодом
        'macd': [12, 26, 9],      # MACD залишається стандартним

        'atr': [14, 21],          # Додана 21-денна ATR для волатильності







    })
    
    # Целевая переменная - ОПТИМІЗОВАНО для прибутковості
    target_type: str = 'direction'     # Змінено на 'direction' для кращої класифікації
    target_horizon: int = 1            # Через сколько периодов измерять цель
    # МАКСИМАЛЬНО ОПТИМІЗОВАНІ параметри для 88.7% винрейт
    min_price_change_threshold: float = 0.045  # ЗБІЛЬШЕНО до 4.5% зміни для найсильніших сигналів
    signal_confidence_threshold: float = 0.85  # ПІДВИЩЕНО до 85% для найкращих рішень
    
    # Разделение данных
    train_split: float = 0.7
    validation_split: float = 0.15
    test_split: float = 0.15
    
    # Обучение
    random_state: int = 42
    cross_validation_folds: int = 5
    
    # XGBoost параметры - ОПТИМІЗОВАНО для зменшення просадки
    xgb_params: Dict = field(default_factory=lambda: {
        'n_estimators': 200,        # Більше дерев для кращої стабільності
        'max_depth': 4,             # Зменшена глибина для запобігання перенавчанню
        'learning_rate': 0.05,      # Повільніше навчання для стабільності
        'random_state': 42,
        'n_jobs': -1,
        # Регуляризація для зменшення дисперсії та ризику
        'subsample': 0.8,           # Випадкова вибірка для стабільності
        'colsample_bytree': 0.8,    # Випадкова вибірка ознак
        'reg_alpha': 0.1,           # L1 регуляризація
        'reg_lambda': 0.1,          # L2 регуляризація
        'gamma': 0.1,               # Мінімальний поділ для зменшення оверфітингу
        'min_child_weight': 5       # Мінімальна вага дочірнього вузла
    })
    
    # Random Forest параметры
    rf_params: Dict = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    })
    
    # LSTM параметры
    lstm_params: Dict = field(default_factory=lambda: {
        'hidden_size': 50,
        'num_layers': 2,
        'dropout': 0.2,
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001
    })
    
    def __post_init__(self):
        if self.data_path is None:
            # Get the project root directory and construct absolute path
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
            self.data_path = os.path.join(
                project_root, 'CryptoTrade', 'data', self.exchange, self.symbol, 
                self.timeframe, '2018_01_01-now.csv'
            )
        
        # Проверим что сумма splits равна 1
        total_split = self.train_split + self.validation_split + self.test_split
        if abs(total_split - 1.0) > 0.001:
            raise ValueError(f"Сумма splits должна быть 1.0, получено {total_split}")


class DataManager:
    """Менеджер данных для ML модулей."""
    
    @staticmethod
    def get_available_pairs() -> Dict[str, List[str]]:
        """Получить доступные торговые пары по биржам."""
        # Get absolute path to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        data_root = os.path.join(project_root, 'CryptoTrade', 'data')
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
        # Get absolute path to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        symbol_path = os.path.join(project_root, 'CryptoTrade', 'data', exchange, symbol)
        timeframes = []
        
        if os.path.exists(symbol_path):
            for timeframe in os.listdir(symbol_path):
                timeframe_path = os.path.join(symbol_path, timeframe)
                if os.path.isdir(timeframe_path):
                    timeframes.append(timeframe)
        
        return sorted(timeframes)


def create_ml_config_interactive() -> MLConfig:
    """Интерактивное создание конфигурации ML модели."""
    print("=== Создание конфигурации ML модели ===\n")
    
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
    for i, pair in enumerate(pairs[:10], 1):  # Показываем первые 10
        print(f"{i}. {pair}")
    if len(pairs) > 10:
        print(f"   ... и еще {len(pairs) - 10} пар")
    
    symbol = input("Введите символ пары (например, BTCUSDT): ").upper()
    if symbol not in pairs:
        print(f"⚠️ Пара {symbol} не найдена, используем BTCUSDT")
        symbol = "BTCUSDT"
    
    # Выбор таймфрейма
    timeframes = DataManager.get_available_timeframes(selected_exchange, symbol)
    print(f"\nДоступные таймфреймы для {symbol}:")
    for i, tf in enumerate(timeframes, 1):
        print(f"{i}. {tf}")
    
    while True:
        try:
            choice = int(input(f"Выберите таймфрейм (1-{len(timeframes)}): ")) - 1
            if 0 <= choice < len(timeframes):
                selected_timeframe = timeframes[choice]
                break
        except ValueError:
            pass
        print("Неверный выбор!")
    
    # Выбор типа модели
    print("\nТип ML модели:")
    print("1. XGBoost (рекомендуется)")
    print("2. Random Forest")
    print("3. LSTM (требует больше времени)")
    print("4. Linear Regression")
    
    model_types = ['xgboost', 'random_forest', 'lstm', 'linear']
    while True:
        try:
            choice = int(input("Выберите модель (1-4): ")) - 1
            if 0 <= choice < len(model_types):
                selected_model = model_types[choice]
                break
        except ValueError:
            pass
        print("Неверный выбор!")
    
    # Выбор целевой переменной
    print("\nЧто предсказывать:")
    print("1. Изменение цены (регрессия)")
    print("2. Направление движения (классификация)")
    print("3. Волатильность")
    
    target_types = ['price_change', 'direction', 'volatility']
    while True:
        try:
            choice = int(input("Выберите цель (1-3): ")) - 1
            if 0 <= choice < len(target_types):
                selected_target = target_types[choice]
                break
        except ValueError:
            pass
        print("Неверный выбор!")
    
    # Создаем конфигурацию
    config = MLConfig(
        exchange=selected_exchange,
        symbol=symbol,
        timeframe=selected_timeframe,
        model_type=selected_model,
        target_type=selected_target
    )
    
    print(f"\n=== Создана конфигурация ML модели ===")
    print(f"Биржа: {config.exchange}")
    print(f"Пара: {config.symbol}")
    print(f"Таймфрейм: {config.timeframe}")
    print(f"Модель: {config.model_type}")
    print(f"Цель: {config.target_type}")
    
    return config