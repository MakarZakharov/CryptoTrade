import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any, Tuple


class RSIVectorbtStrategy:
    """
    RSI стратегия для vectorbt
    
    Сигналы:
    - Покупка: RSI < oversold (перепроданность)
    - Продажа: RSI > overbought (перекупленность)
    """
    
    def __init__(self, rsi_period: int = 14, oversold: float = 30, overbought: float = 70):
        """
        Инициализация параметров стратегии
        
        Args:
            rsi_period: Период для расчета RSI
            oversold: Уровень перепроданности (сигнал покупки)
            overbought: Уровень перекупленности (сигнал продажи)
        """
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.name = "RSI Vectorbt Strategy"
    
    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """
        Расчет RSI индикатора с использованием vectorbt
        
        Args:
            prices: Серия цен закрытия
            
        Returns:
            Серия значений RSI
        """
        # Используем vectorbt RSI индикатор
        rsi = vbt.RSI.run(prices, window=self.rsi_period)
        return rsi.rsi
    
    def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Генерация торговых сигналов
        
        Args:
            data: DataFrame с данными OHLCV
            
        Returns:
            Tuple с сигналами входа и выхода
        """
        # Расчет RSI
        rsi = self.calculate_rsi(data['close'])
        
        # Генерация сигналов
        # Покупка когда RSI выходит из зоны перепроданности (пересекает oversold снизу вверх)
        entries = (rsi > self.oversold) & (rsi.shift(1) <= self.oversold)
        
        # Продажа когда RSI выходит из зоны перекупленности (пересекает overbought сверху вниз)  
        exits = (rsi < self.overbought) & (rsi.shift(1) >= self.overbought)
        
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
            'rsi_period': self.rsi_period,
            'oversold': self.oversold,
            'overbought': self.overbought,
            'strategy_name': self.name
        }
    



def run_simple_rsi_strategy(data_path: str, **kwargs) -> Dict[str, Any]:
    """
    Запуск простой RSI стратегии
    
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
    strategy = RSIVectorbtStrategy(**kwargs)
    
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