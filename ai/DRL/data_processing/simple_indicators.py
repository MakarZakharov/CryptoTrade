"""
Простые технические индикаторы без зависимости от pandas-ta.
"""

import pandas as pd
import numpy as np


def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Простое скользящее среднее."""
    return series.rolling(window=window).mean()


def exponential_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Экспоненциальное скользящее среднее."""
    return series.ewm(span=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gains = delta.where(delta > 0, 0)
    losses = (-delta).where(delta < 0, 0)
    
    avg_gains = gains.rolling(window=window).mean()
    avg_losses = losses.rolling(window=window).mean()
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return rsi


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2):
    """Bollinger Bands."""
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    
    return upper, sma, lower


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD indicator."""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range."""
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()