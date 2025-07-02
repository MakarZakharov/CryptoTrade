"""
Technical indicators for market analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class TechnicalIndicators:
    """
    Class for calculating technical indicators.
    """
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame, include: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Add all available technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            include: List of indicators to include (default: all)
            
        Returns:
            DataFrame with added technical indicators
        """
        result = df.copy()
        
        # Define all available indicators
        all_indicators = {
            'sma': TechnicalIndicators.add_sma,
            'ema': TechnicalIndicators.add_ema,
            'rsi': TechnicalIndicators.add_rsi,
            'macd': TechnicalIndicators.add_macd,
            'bollinger': TechnicalIndicators.add_bollinger_bands,
            'atr': TechnicalIndicators.add_atr,
            'adx': TechnicalIndicators.add_adx,
            'momentum': TechnicalIndicators.add_momentum,
            'stochastic': TechnicalIndicators.add_stochastic,
            'williams_r': TechnicalIndicators.add_williams_r,
            'obv': TechnicalIndicators.add_obv,
            'ichimoku': TechnicalIndicators.add_ichimoku,
            'vwap': TechnicalIndicators.add_vwap,
        }
        
        # Filter indicators if include is specified
        indicators_to_add = all_indicators
        if include is not None:
            indicators_to_add = {k: v for k, v in all_indicators.items() if k in include}
        
        # Add each indicator
        for indicator_func in indicators_to_add.values():
            result = indicator_func(result)
        
        # Fill NaN values
        result = result.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return result
    
    @staticmethod
    def add_sma(df: pd.DataFrame, periods: List[int] = [5, 20, 50, 200]) -> pd.DataFrame:
        """
        Add Simple Moving Average (SMA) indicators.
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of periods for SMA calculation
            
        Returns:
            DataFrame with added SMA indicators
        """
        result = df.copy()
        for period in periods:
            result[f'sma_{period}'] = result['close'].rolling(window=period).mean()
        return result
    
    @staticmethod
    def add_ema(df: pd.DataFrame, periods: List[int] = [5, 20, 50, 200]) -> pd.DataFrame:
        """
        Add Exponential Moving Average (EMA) indicators.
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of periods for EMA calculation
            
        Returns:
            DataFrame with added EMA indicators
        """
        result = df.copy()
        for period in periods:
            result[f'ema_{period}'] = result['close'].ewm(span=period, adjust=False).mean()
        return result
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI) indicator.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for RSI calculation
            
        Returns:
            DataFrame with added RSI indicator
        """
        result = df.copy()
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        result[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        return result
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) indicators.
        
        Args:
            df: DataFrame with OHLCV data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal EMA period
            
        Returns:
            DataFrame with added MACD indicators
        """
        result = df.copy()
        
        # Calculate MACD components
        ema_fast = result['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = result['close'].ewm(span=slow_period, adjust=False).mean()
        
        result['macd'] = ema_fast - ema_slow
        result['macd_signal'] = result['macd'].ewm(span=signal_period, adjust=False).mean()
        result['macd_hist'] = result['macd'] - result['macd_signal']
        
        return result
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands indicators.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for moving average
            std_dev: Number of standard deviations
            
        Returns:
            DataFrame with added Bollinger Bands indicators
        """
        result = df.copy()
        
        result['bb_middle'] = result['close'].rolling(window=period).mean()
        result['bb_std'] = result['close'].rolling(window=period).std()
        
        result['bb_upper'] = result['bb_middle'] + (result['bb_std'] * std_dev)
        result['bb_lower'] = result['bb_middle'] - (result['bb_std'] * std_dev)
        
        # Add bandwidth and %B indicators
        result['bb_bandwidth'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        result['bb_percent_b'] = (result['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])
        
        return result
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (ATR) indicator.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for ATR calculation
            
        Returns:
            DataFrame with added ATR indicator
        """
        result = df.copy()
        
        high_low = result['high'] - result['low']
        high_close = np.abs(result['high'] - result['close'].shift())
        low_close = np.abs(result['low'] - result['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        result[f'atr_{period}'] = true_range.rolling(period).mean()
        
        return result
    
    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average Directional Index (ADX) indicator.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for ADX calculation
            
        Returns:
            DataFrame with added ADX indicator
        """
        result = df.copy()
        
        # Calculate True Range
        result['tr'] = np.maximum(
            np.maximum(
                result['high'] - result['low'],
                np.abs(result['high'] - result['close'].shift())
            ),
            np.abs(result['low'] - result['close'].shift())
        )
        
        # Calculate Directional Movement
        result['up_move'] = result['high'] - result['high'].shift()
        result['down_move'] = result['low'].shift() - result['low']
        
        result['plus_dm'] = np.where(
            (result['up_move'] > result['down_move']) & (result['up_move'] > 0),
            result['up_move'],
            0
        )
        
        result['minus_dm'] = np.where(
            (result['down_move'] > result['up_move']) & (result['down_move'] > 0),
            result['down_move'],
            0
        )
        
        # Calculate smoothed averages
        result[f'atr_{period}'] = result['tr'].rolling(period).mean()
        result['plus_di'] = 100 * (result['plus_dm'].rolling(period).mean() / result[f'atr_{period}'])
        result['minus_di'] = 100 * (result['minus_dm'].rolling(period).mean() / result[f'atr_{period}'])
        
        # Calculate ADX
        result['dx'] = 100 * np.abs(result['plus_di'] - result['minus_di']) / (result['plus_di'] + result['minus_di'])
        result[f'adx_{period}'] = result['dx'].rolling(period).mean()
        
        # Drop intermediate columns
        result = result.drop(['tr', 'up_move', 'down_move', 'plus_dm', 'minus_dm', 'dx'], axis=1)
        
        return result
    
    @staticmethod
    def add_momentum(df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Add Momentum indicators.
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of periods for momentum calculation
            
        Returns:
            DataFrame with added Momentum indicators
        """
        result = df.copy()
        
        for period in periods:
            result[f'momentum_{period}'] = result['close'] / result['close'].shift(period) - 1
        
        return result
    
    @staticmethod
    def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Add Stochastic Oscillator indicators.
        
        Args:
            df: DataFrame with OHLCV data
            k_period: Period for %K calculation
            d_period: Period for %D calculation
            
        Returns:
            DataFrame with added Stochastic Oscillator indicators
        """
        result = df.copy()
        
        # Calculate %K
        low_min = result['low'].rolling(window=k_period).min()
        high_max = result['high'].rolling(window=k_period).max()
        
        result['stoch_k'] = 100 * ((result['close'] - low_min) / (high_max - low_min))
        
        # Calculate %D
        result['stoch_d'] = result['stoch_k'].rolling(window=d_period).mean()
        
        return result
    
    @staticmethod
    def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Williams %R indicator.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for Williams %R calculation
            
        Returns:
            DataFrame with added Williams %R indicator
        """
        result = df.copy()
        
        high_max = result['high'].rolling(window=period).max()
        low_min = result['low'].rolling(window=period).min()
        
        result[f'williams_r_{period}'] = -100 * ((high_max - result['close']) / (high_max - low_min))
        
        return result
    
    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add On-Balance Volume (OBV) indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added OBV indicator
        """
        result = df.copy()
        
        result['obv'] = np.where(
            result['close'] > result['close'].shift(),
            result['volume'],
            np.where(
                result['close'] < result['close'].shift(),
                -result['volume'],
                0
            )
        ).cumsum()
        
        return result
    
    @staticmethod
    def add_ichimoku(df: pd.DataFrame, conversion_period: int = 9, base_period: int = 26, lagging_span_period: int = 52) -> pd.DataFrame:
        """
        Add Ichimoku Cloud indicators.
        
        Args:
            df: DataFrame with OHLCV data
            conversion_period: Period for Conversion Line
            base_period: Period for Base Line
            lagging_span_period: Period for Lagging Span
            
        Returns:
            DataFrame with added Ichimoku Cloud indicators
        """
        result = df.copy()
        
        # Conversion Line (Tenkan-sen)
        high_tenkan = result['high'].rolling(window=conversion_period).max()
        low_tenkan = result['low'].rolling(window=conversion_period).min()
        result['ichimoku_conversion'] = (high_tenkan + low_tenkan) / 2
        
        # Base Line (Kijun-sen)
        high_kijun = result['high'].rolling(window=base_period).max()
        low_kijun = result['low'].rolling(window=base_period).min()
        result['ichimoku_base'] = (high_kijun + low_kijun) / 2
        
        # Leading Span A (Senkou Span A)
        result['ichimoku_span_a'] = ((result['ichimoku_conversion'] + result['ichimoku_base']) / 2).shift(base_period)
        
        # Leading Span B (Senkou Span B)
        high_senkou = result['high'].rolling(window=lagging_span_period).max()
        low_senkou = result['low'].rolling(window=lagging_span_period).min()
        result['ichimoku_span_b'] = ((high_senkou + low_senkou) / 2).shift(base_period)
        
        # Lagging Span (Chikou Span)
        result['ichimoku_lagging'] = result['close'].shift(-base_period)
        
        return result
    
    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Volume Weighted Average Price (VWAP) indicator.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added VWAP indicator
        """
        result = df.copy()
        
        # Calculate typical price
        result['typical_price'] = (result['high'] + result['low'] + result['close']) / 3
        
        # Calculate VWAP
        result['vwap'] = (result['typical_price'] * result['volume']).cumsum() / result['volume'].cumsum()
        
        # Drop intermediate columns
        result = result.drop(['typical_price'], axis=1)
        
        return result