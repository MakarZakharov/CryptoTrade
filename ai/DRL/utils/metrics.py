"""Финансовые метрики для оценки торговых стратегий."""

import numpy as np
import pandas as pd
from typing import List, Union, Dict, Any


class TradingMetrics:
    """Класс для расчета торговых метрик."""
    
    @staticmethod
    def total_return(returns: Union[List, np.ndarray, pd.Series]) -> float:
        """Общая доходность."""
        returns = np.array(returns)
        return float(np.prod(1 + returns) - 1)
    
    @staticmethod
    def annualized_return(returns: Union[List, np.ndarray, pd.Series], periods_per_year: int = 365) -> float:
        """Годовая доходность."""
        returns = np.array(returns)
        total_ret = TradingMetrics.total_return(returns)
        return float((1 + total_ret) ** (periods_per_year / len(returns)) - 1)
    
    @staticmethod
    def volatility(returns: Union[List, np.ndarray, pd.Series], annualized: bool = True, periods_per_year: int = 365) -> float:
        """Волатильность доходности."""
        returns = np.array(returns)
        vol = float(np.std(returns, ddof=1))
        return vol * np.sqrt(periods_per_year) if annualized else vol
    
    @staticmethod
    def sharpe_ratio(returns: Union[List, np.ndarray, pd.Series], risk_free_rate: float = 0.0, periods_per_year: int = 365) -> float:
        """Коэффициент Шарпа."""
        returns = np.array(returns)
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - risk_free_rate / periods_per_year
        return float(np.mean(excess_returns) / np.std(returns, ddof=1) * np.sqrt(periods_per_year))
    
    @staticmethod
    def max_drawdown(portfolio_values: Union[List, np.ndarray, pd.Series]) -> float:
        """Максимальная просадка."""
        values = np.array(portfolio_values)
        if len(values) == 0:
            return 0.0
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        return float(np.max(drawdown))
    
    @staticmethod
    def sortino_ratio(returns: Union[List, np.ndarray, pd.Series], target_return: float = 0.0, periods_per_year: int = 365) -> float:
        """Коэффициент Сортино."""
        returns = np.array(returns)
        if len(returns) == 0:
            return 0.0
        downside_returns = returns[returns < target_return]
        if len(downside_returns) == 0:
            return np.inf if np.mean(returns) > target_return else 0.0
        downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2))
        excess_return = np.mean(returns) - target_return
        return float(excess_return / downside_deviation * np.sqrt(periods_per_year))
    
    @staticmethod
    def calmar_ratio(returns: Union[List, np.ndarray, pd.Series], portfolio_values: Union[List, np.ndarray, pd.Series] = None) -> float:
        """Коэффициент Кальмара (годовая доходность / максимальная просадка)."""
        if portfolio_values is None:
            # Рассчитываем значения портфеля из доходности
            portfolio_values = np.cumprod(1 + np.array(returns))
        
        annual_return = TradingMetrics.annualized_return(returns)
        max_dd = TradingMetrics.max_drawdown(portfolio_values)
        
        return annual_return / max_dd if max_dd != 0 else np.inf
    
    @staticmethod
    def win_rate(trades: List[float]) -> float:
        """Процент прибыльных сделок."""
        if len(trades) == 0:
            return 0.0
        profitable_trades = sum(1 for trade in trades if trade > 0)
        return profitable_trades / len(trades)
    
    @staticmethod
    def profit_factor(trades: List[float]) -> float:
        """Коэффициент прибыли (общая прибыль / общие убытки)."""
        if len(trades) == 0:
            return 0.0
        
        profits = sum(trade for trade in trades if trade > 0)
        losses = abs(sum(trade for trade in trades if trade < 0))
        
        return profits / losses if losses != 0 else np.inf if profits > 0 else 0.0