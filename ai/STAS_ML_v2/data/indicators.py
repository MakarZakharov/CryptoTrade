"""
Technical indicators for STAS_ML v2
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from ..core.config import Config
from ..core.base import Logger

# Try to import TA-Lib, fallback to pandas implementations
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


class TechnicalIndicators:
    """Technical indicators calculator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger("TechnicalIndicators")
        
        if not TALIB_AVAILABLE:
            self.logger.warning("TA-Lib not available, using pandas implementations")
    
    def add_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all configured technical indicators."""
        df = data.copy()
        
        # RSI
        for period in self.config.features.rsi_periods:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
        
        # MACD
        macd_config = self.config.features.macd_config
        macd, signal, histogram = self._calculate_macd(
            df['close'], 
            macd_config['fast'], 
            macd_config['slow'], 
            macd_config['signal']
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_histogram'] = histogram
        
        # Bollinger Bands
        for period in self.config.features.bollinger_periods:
            upper, middle, lower = self._calculate_bollinger_bands(df['close'], period)
            df[f'bb_upper_{period}'] = upper
            df[f'bb_middle_{period}'] = middle
            df[f'bb_lower_{period}'] = lower
            df[f'bb_width_{period}'] = (upper - lower) / middle
        
        # ATR
        for period in self.config.features.atr_periods:
            df[f'atr_{period}'] = self._calculate_atr(df['high'], df['low'], df['close'], period)
        
        # Moving averages
        for window in self.config.features.price_windows:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        if TALIB_AVAILABLE:
            return talib.RSI(prices.values, timeperiod=period)
        else:
            # Pandas implementation
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD."""
        if TALIB_AVAILABLE:
            macd, signal_line, histogram = talib.MACD(prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return pd.Series(macd, index=prices.index), pd.Series(signal_line, index=prices.index), pd.Series(histogram, index=prices.index)
        else:
            # Pandas implementation
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            return macd, signal_line, histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2):
        """Calculate Bollinger Bands."""
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(prices.values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            return pd.Series(upper, index=prices.index), pd.Series(middle, index=prices.index), pd.Series(lower, index=prices.index)
        else:
            # Pandas implementation
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, sma, lower
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        if TALIB_AVAILABLE:
            return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
        else:
            # Pandas implementation
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range.rolling(window=period).mean()