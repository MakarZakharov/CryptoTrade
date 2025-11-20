"""
Technical indicators module for CryptoTradingEnv.

Implements common technical analysis indicators used in observations.
All indicators guarantee no lookahead bias - only use past data.
"""

import numpy as np
import pandas as pd
from typing import Optional


class TechnicalIndicators:
    """
    Technical indicator calculations for trading environment.

    All methods are static and operate on pandas Series/DataFrames.
    Ensures no lookahead bias - indicators at time t only use data up to t.
    """

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.

        Returns values in range [0, 100].
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence.

        Returns DataFrame with columns: macd, signal, histogram
        """
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Bollinger Bands.

        Returns DataFrame with columns: upper, middle, lower
        """
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        })

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range - volatility indicator.
        """
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator.

        Returns DataFrame with columns: k, d
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()

        return pd.DataFrame({'k': k, 'd': d})

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()

    @staticmethod
    def returns(close: pd.Series, periods: int = 1) -> pd.Series:
        """Simple returns."""
        return close.pct_change(periods=periods)

    @staticmethod
    def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
        """Log returns."""
        return np.log(close / close.shift(periods))

    @staticmethod
    def volatility(returns: pd.Series, period: int = 20) -> pd.Series:
        """Rolling volatility (standard deviation of returns)."""
        return returns.rolling(window=period).std()


def compute_indicators(df: pd.DataFrame, indicator_list: list) -> pd.DataFrame:
    """
    Compute multiple technical indicators on OHLCV data.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        indicator_list: List of indicator names (e.g., ["ema_10", "rsi_14", "macd"])

    Returns:
        DataFrame with original data + computed indicators

    Supported indicators:
        - ema_{period}: EMA with specified period
        - sma_{period}: SMA with specified period
        - rsi_{period}: RSI with specified period
        - macd: MACD (returns macd, signal, histogram columns)
        - bb_upper, bb_middle, bb_lower: Bollinger Bands
        - atr_{period}: Average True Range
        - stoch: Stochastic Oscillator (returns stoch_k, stoch_d)
        - obv: On-Balance Volume
        - vwap: Volume Weighted Average Price
        - returns_{period}: Simple returns
        - log_returns_{period}: Log returns
        - volatility_{period}: Rolling volatility
    """
    result_df = df.copy()
    ti = TechnicalIndicators()

    for indicator in indicator_list:
        try:
            # Parse indicator name and parameters
            parts = indicator.split('_')
            name = parts[0]

            if name == 'ema':
                period = int(parts[1]) if len(parts) > 1 else 10
                result_df[f'ema_{period}'] = ti.ema(df['close'], period)

            elif name == 'sma':
                period = int(parts[1]) if len(parts) > 1 else 20
                result_df[f'sma_{period}'] = ti.sma(df['close'], period)

            elif name == 'rsi':
                period = int(parts[1]) if len(parts) > 1 else 14
                result_df[f'rsi_{period}'] = ti.rsi(df['close'], period)

            elif name == 'macd':
                macd_df = ti.macd(df['close'])
                result_df['macd'] = macd_df['macd']
                result_df['macd_signal'] = macd_df['signal']
                result_df['macd_histogram'] = macd_df['histogram']

            elif name in ['bb', 'bollinger']:
                bb_df = ti.bollinger_bands(df['close'])
                result_df['bb_upper'] = bb_df['upper']
                result_df['bb_middle'] = bb_df['middle']
                result_df['bb_lower'] = bb_df['lower']

            elif name == 'atr':
                period = int(parts[1]) if len(parts) > 1 else 14
                result_df[f'atr_{period}'] = ti.atr(df['high'], df['low'], df['close'], period)

            elif name in ['stoch', 'stochastic']:
                stoch_df = ti.stochastic(df['high'], df['low'], df['close'])
                result_df['stoch_k'] = stoch_df['k']
                result_df['stoch_d'] = stoch_df['d']

            elif name == 'obv':
                result_df['obv'] = ti.obv(df['close'], df['volume'])

            elif name == 'vwap':
                result_df['vwap'] = ti.vwap(df['high'], df['low'], df['close'], df['volume'])

            elif name == 'returns':
                period = int(parts[1]) if len(parts) > 1 else 1
                result_df[f'returns_{period}'] = ti.returns(df['close'], period)

            elif name == 'log' and len(parts) > 1 and parts[1] == 'returns':
                period = int(parts[2]) if len(parts) > 2 else 1
                result_df[f'log_returns_{period}'] = ti.log_returns(df['close'], period)

            elif name == 'volatility':
                period = int(parts[1]) if len(parts) > 1 else 20
                returns = ti.returns(df['close'])
                result_df[f'volatility_{period}'] = ti.volatility(returns, period)

            else:
                print(f"Warning: Unknown indicator '{indicator}', skipping...")

        except Exception as e:
            print(f"Error computing indicator '{indicator}': {e}")
            continue

    return result_df


def normalize_features(df: pd.DataFrame, method: str = "zscore",
                      window: int = 200, columns: Optional[list] = None) -> pd.DataFrame:
    """
    Normalize features for neural network input.

    Args:
        df: DataFrame with features
        method: Normalization method ("zscore", "minmax", "none")
        window: Rolling window for computing normalization statistics
        columns: List of columns to normalize (None = all numeric columns)

    Returns:
        DataFrame with normalized features
    """
    result_df = df.copy()

    if method == "none":
        return result_df

    if columns is None:
        # Normalize all numeric columns except timestamp
        columns = result_df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in result_df.columns:
            continue

        if method == "zscore":
            # Z-score normalization: (x - mean) / std
            rolling_mean = result_df[col].rolling(window=window, min_periods=1).mean()
            rolling_std = result_df[col].rolling(window=window, min_periods=1).std()
            result_df[col] = (result_df[col] - rolling_mean) / (rolling_std + 1e-8)

        elif method == "minmax":
            # Min-max normalization: (x - min) / (max - min)
            rolling_min = result_df[col].rolling(window=window, min_periods=1).min()
            rolling_max = result_df[col].rolling(window=window, min_periods=1).max()
            result_df[col] = (result_df[col] - rolling_min) / (rolling_max - rolling_min + 1e-8)

    # Replace any inf/nan with 0
    result_df = result_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return result_df
