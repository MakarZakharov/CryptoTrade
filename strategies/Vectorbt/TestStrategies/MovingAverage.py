import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any, Tuple


class MovingAverageCrossoverStrategy:
    """
    Стратегия пересечения скользящих средних для vectorbt
    
    Сигналы:
    - Покупка: Быстрая MA пересекает медленную MA снизу вверх
    - Продажа: Быстрая MA пересекается медленную MA сверху вниз
    """
    
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        """
        Инициализация параметров стратегии
        
        Args:
            fast_period: Период быстрой скользящей средней
            slow_period: Период медленной скользящей средней
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = "Moving Average Crossover Strategy"
    
    def calculate_moving_averages(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Расчет скользящих средних
        
        Args:
            prices: Серия цен закрытия
            
        Returns:
            Tuple с быстрой и медленной MA
        """
        fast_ma = prices.rolling(window=self.fast_period).mean()
        slow_ma = prices.rolling(window=self.slow_period).mean()
        
        return fast_ma, slow_ma
    
    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Генерация торговых сигналов
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            Tuple с сигналами входа и выхода
        """
        # Расчет скользящих средних
        fast_ma, slow_ma = self.calculate_moving_averages(data['close'])
        
        # Генерация сигналов
        # Покупка когда быстрая MA пересекает медленную снизу вверх
        entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        
        # Продажа когда быстрая MA пересекает медленную сверху вниз
        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
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
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'strategy_name': self.name
        }


def run_simple_ma_strategy(data_path: str, **kwargs) -> Dict[str, Any]:
    """
    Запуск простой MA стратегии
    
    Args:
        data_path: Путь к CSV файлу с данными
        **kwargs: Дополнительные параметры стратегии
        
    Returns:
        Словарь с результатами бэктеста
    """
    # Загрузка данных
    data = pd.read_csv(data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    
    # Создание стратегии
    strategy = MovingAverageCrossoverStrategy(**kwargs)
    
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