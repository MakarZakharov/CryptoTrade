"""
Technical Indicators module for market analysis
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class TechnicalIndicators:
    """Class for calculating technical indicators on market data."""
    
    @staticmethod
    def add_all_indicators(data: pd.DataFrame, include: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Add all technical indicators to the dataframe.
        
        Args:
            data: DataFrame with OHLCV data (open, high, low, close, volume)
            include: List of indicators to include. If None, includes all.
            
        Returns:
            DataFrame with added technical indicators
        """
        df = data.copy()
        
        if include is None:
            include = ['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'adx', 
                      'momentum', 'stochastic', 'williams_r', 'obv', 'ichimoku', 'vwap']
        
        # Simple Moving Average
        if 'sma' in include:
            df = TechnicalIndicators._add_sma(df, [5, 20, 50])
            
        # Exponential Moving Average  
        if 'ema' in include:
            df = TechnicalIndicators._add_ema(df, [5, 20, 50])
            
        # RSI
        if 'rsi' in include:
            df = TechnicalIndicators._add_rsi(df, 14)
            
        # MACD
        if 'macd' in include:
            df = TechnicalIndicators._add_macd(df, 12, 26, 9)
            
        # Bollinger Bands
        if 'bollinger' in include:
            df = TechnicalIndicators._add_bollinger(df, 20)
            
        # ATR
        if 'atr' in include:
            df = TechnicalIndicators._add_atr(df, 14)
            
        # ADX
        if 'adx' in include:
            df = TechnicalIndicators._add_adx(df, 14)
            
        # Momentum
        if 'momentum' in include:
            df = TechnicalIndicators._add_momentum(df, [5, 10, 20])
            
        # Stochastic
        if 'stochastic' in include:
            df = TechnicalIndicators._add_stochastic(df, 14)
            
        # Williams %R
        if 'williams_r' in include:
            df = TechnicalIndicators._add_williams_r(df, 14)
            
        # OBV
        if 'obv' in include:
            df = TechnicalIndicators._add_obv(df)
            
        # Ichimoku
        if 'ichimoku' in include:
            df = TechnicalIndicators._add_ichimoku(df, 9, 26, 52)
            
        # VWAP
        if 'vwap' in include:
            df = TechnicalIndicators._add_vwap(df)
            
        return df
    
    @staticmethod
    def _add_sma(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add Simple Moving Average indicators."""
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    @staticmethod
    def _add_ema(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add Exponential Moving Average indicators."""
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        return df
    
    @staticmethod
    def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI indicator."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def _add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Add MACD indicators."""
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        df[f'macd_{fast}_{slow}'] = ema_fast - ema_slow
        df[f'macd_signal_{signal}'] = df[f'macd_{fast}_{slow}'].ewm(span=signal).mean()
        df[f'macd_histogram'] = df[f'macd_{fast}_{slow}'] - df[f'macd_signal_{signal}']
        return df
    
    @staticmethod
    def _add_bollinger(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Bollinger Bands."""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df[f'bb_upper_{period}'] = sma + (std * 2)
        df[f'bb_lower_{period}'] = sma - (std * 2)
        df[f'bb_middle_{period}'] = sma
        return df
    
    @staticmethod
    def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.DataFrame([high_low, high_close, low_close]).max()
        df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        return df
    
    @staticmethod
    def _add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index (simplified version)."""
        # Simplified ADX calculation
        df[f'adx_{period}'] = df['high'].rolling(window=period).std() / df['close'] * 100
        return df
    
    @staticmethod
    def _add_momentum(df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Add Momentum indicators."""
        for period in periods:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        return df
    
    @staticmethod
    def _add_stochastic(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Stochastic oscillator."""
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        df[f'stoch_k_{period}'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
        return df
    
    @staticmethod
    def _add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Williams %R."""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        df[f'williams_r_{period}'] = -100 * (high_max - df['close']) / (high_max - low_min)
        return df
    
    @staticmethod
    def _add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Add On Balance Volume."""
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        return df
    
    @staticmethod
    def _add_ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou: int = 52) -> pd.DataFrame:
        """Add Ichimoku Cloud indicators."""
        # Tenkan-sen
        high_9 = df['high'].rolling(window=tenkan).max()
        low_9 = df['low'].rolling(window=tenkan).min()
        df[f'ichimoku_tenkan_{tenkan}'] = (high_9 + low_9) / 2
        
        # Kijun-sen
        high_26 = df['high'].rolling(window=kijun).max()
        low_26 = df['low'].rolling(window=kijun).min()
        df[f'ichimoku_kijun_{kijun}'] = (high_26 + low_26) / 2
        
        # Senkou Span A
        df[f'ichimoku_senkou_a'] = ((df[f'ichimoku_tenkan_{tenkan}'] + df[f'ichimoku_kijun_{kijun}']) / 2).shift(kijun)
        
        # Senkou Span B
        high_52 = df['high'].rolling(window=senkou).max()
        low_52 = df['low'].rolling(window=senkou).min()
        df[f'ichimoku_senkou_b'] = ((high_52 + low_52) / 2).shift(kijun)
        
        return df
    
    @staticmethod
    def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume Weighted Average Price."""
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        return df