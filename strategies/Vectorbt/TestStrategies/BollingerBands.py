import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any, Tuple


class BollingerBandsStrategy:
    """
    Стратегия Bollinger Bands для vectorbt
    
    Сигналы:
    - Покупка: Цена касается нижней линии Bollinger Bands
    - Продажа: Цена касается верхней линии Bollinger Bands
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Инициализация параметров стратегии
        
        Args:
            period: Период для расчета скользящей средней и стандартного отклонения
            std_dev: Множитель стандартного отклонения для полос
        """
        self.period = period
        self.std_dev = std_dev
        self.name = "Bollinger Bands Strategy"
    
    def calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Расчет Bollinger Bands
        
        Args:
            prices: Серия цен закрытия
            
        Returns:
            Tuple с верхней, средней и нижней полосами
        """
        # Средняя линия (SMA)
        middle_band = prices.rolling(window=self.period).mean()
        
        # Стандартное отклонение
        std = prices.rolling(window=self.period).std()
        
        # Верхняя и нижняя полосы
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)
        
        return upper_band, middle_band, lower_band
    
    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Генерация торговых сигналов
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            Tuple с сигналами входа и выхода
        """
        # Расчет Bollinger Bands
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(data['close'])
        
        # Генерация сигналов
        # Покупка когда цена касается или пробивает нижнюю полосу
        entries = (data['close'] <= lower_band) & (data['close'].shift(1) > lower_band.shift(1))
        
        # Продажа когда цена касается или пробивает верхнюю полосу
        exits = (data['close'] >= upper_band) & (data['close'].shift(1) < upper_band.shift(1))
        
        return entries, exits
    
    def backtest(self, data: pd.DataFrame, initial_cash: float = 100000, 
                 fees: float = 0.001) -> vbt.Portfolio:
        """
        Запуск бэктеста стратегии
        
        Args:
            data: DataFrame с данными OHLCV
            initial_cash: Начальный капитал
            fees: Комиссия за сделку
            
        Returns:
            Объект Portfolio с результатами бэктеста
        """
        # Генерация сигналов
        entries, exits = self.generate_signals(data)
        
        # Создание портфеля
        portfolio = vbt.Portfolio.from_signals(
            data['close'],
            entries=entries,
            exits=exits,
            init_cash=initial_cash,
            fees=fees,
            freq='D'  # Дневная частота
        )
        
        return portfolio
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """
        Возвращает параметры стратегии
        
        Returns:
            Словарь с параметрами
        """
        return {
            'period': self.period,
            'std_dev': self.std_dev,
            'strategy_name': self.name
        }


class BollingerBandsMeanReversionStrategy:
    """
    Стратегия возврата к среднему на основе Bollinger Bands
    
    Сигналы:
    - Покупка: Цена ниже нижней полосы, ожидание возврата к средней
    - Продажа: Цена выше верхней полосы, ожидание возврата к средней
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, exit_at_middle: bool = True):
        """
        Инициализация параметров стратегии
        
        Args:
            period: Период для расчета
            std_dev: Множитель стандартного отклонения
            exit_at_middle: Выходить ли на средней линии
        """
        self.period = period
        self.std_dev = std_dev
        self.exit_at_middle = exit_at_middle
        self.name = "Bollinger Bands Mean Reversion Strategy"
    
    def calculate_bollinger_bands(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Расчет Bollinger Bands
        
        Args:
            prices: Серия цен закрытия
            
        Returns:
            Tuple с верхней, средней и нижней полосами
        """
        middle_band = prices.rolling(window=self.period).mean()
        std = prices.rolling(window=self.period).std()
        upper_band = middle_band + (std * self.std_dev)
        lower_band = middle_band - (std * self.std_dev)
        
        return upper_band, middle_band, lower_band
    
    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Генерация торговых сигналов для mean reversion
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            Tuple с сигналами входа и выхода
        """
        upper_band, middle_band, lower_band = self.calculate_bollinger_bands(data['close'])
        
        # Покупка при пересечении нижней полосы (ожидание отскока вверх)
        entries = (data['close'] <= lower_band) & (data['close'].shift(1) > lower_band)
        
        if self.exit_at_middle:
            # Выход при возврате к средней линии
            exits = data['close'] >= middle_band
        else:
            # Выход при касании верхней полосы
            exits = data['close'] >= upper_band
        
        return entries, exits
    
    def backtest(self, data: pd.DataFrame, initial_cash: float = 100000, 
                 fees: float = 0.001) -> vbt.Portfolio:
        """
        Запуск бэктеста стратегии
        """
        entries, exits = self.generate_signals(data)
        
        portfolio = vbt.Portfolio.from_signals(
            data['close'],
            entries=entries,
            exits=exits,
            init_cash=initial_cash,
            fees=fees,
            freq='D'
        )
        
        return portfolio
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """
        Возвращает параметры стратегии
        """
        return {
            'period': self.period,
            'std_dev': self.std_dev,
            'exit_at_middle': self.exit_at_middle,
            'strategy_name': self.name
        }


def run_simple_bb_strategy(data_path: str, strategy_type: str = "breakout", **kwargs) -> Dict[str, Any]:
    """
    Запуск Bollinger Bands стратегии
    
    Args:
        data_path: Путь к CSV файлу с данными
        strategy_type: Тип стратегии ("breakout" или "mean_reversion")
        **kwargs: Дополнительные параметры стратегии
        
    Returns:
        Словарь с результатами бэктеста
    """
    # Загрузка данных
    data = pd.read_csv(data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    
    # Создание стратегии
    if strategy_type == "mean_reversion":
        strategy = BollingerBandsMeanReversionStrategy(**kwargs)
    else:
        strategy = BollingerBandsStrategy(**kwargs)
    
    # Запуск бэктеста
    portfolio = strategy.backtest(data)
    
    # Получение результатов
    results = {
        'strategy_params': strategy.get_strategy_params(),
        'total_return': portfolio.total_return(),
        'total_return_pct': portfolio.total_return() * 100,
        'sharpe_ratio': portfolio.sharpe_ratio(),
        'max_drawdown': portfolio.max_drawdown(),
        'max_drawdown_pct': portfolio.max_drawdown() * 100,
        'total_trades': portfolio.orders.count(),
        'win_rate': portfolio.trades.win_rate(),
        'profit_factor': portfolio.trades.profit_factor(),
        'portfolio': portfolio
    }
    
    return results