"""
Feature Extractor with comprehensive technical indicators
Optimized for performance on limited hardware
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import talib
from numba import jit
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """
    Extracts technical indicators and features for RL trading
    Optimized for performance with caching and vectorization
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.indicators_config = config.get('INDICATORS_CONFIG', {})
        self.state_window = config.get('STATE_WINDOW', 50)
        self.scale_features = config.get('FEATURE_SCALING', True)
        
        # Cache for computed indicators
        self._cache = {}
        
    def extract_features(self, df: pd.DataFrame, timeframe: str) -> np.ndarray:
        """
        Extract all features from price data
        
        Args:
            df: DataFrame with OHLCV data
            timeframe: Timeframe of the data
            
        Returns:
            Feature matrix
        """
        # Clear cache for new data
        self._cache.clear()
        
        features = {}
        
        # Price-based features
        features.update(self._extract_price_features(df))
        
        # Trend indicators
        features.update(self._extract_trend_indicators(df))
        
        # Momentum indicators
        features.update(self._extract_momentum_indicators(df))
        
        # Volatility indicators
        features.update(self._extract_volatility_indicators(df))
        
        # Volume indicators
        features.update(self._extract_volume_indicators(df))
        
        # Market structure
        features.update(self._extract_market_structure(df))
        
        # Multi-timeframe features
        features.update(self._extract_mtf_features(df, timeframe))
        
        # Convert to array
        feature_array = self._features_to_array(features, df)
        
        # Scale features if enabled
        if self.scale_features:
            feature_array = self._scale_features(feature_array)
        
        return feature_array
    
    def _extract_price_features(self, df: pd.DataFrame) -> Dict:
        """Extract basic price features"""
        features = {}
        
        # Price ratios
        features['close_open_ratio'] = df['close'] / df['open']
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_high_ratio'] = df['close'] / df['high']
        features['close_low_ratio'] = df['close'] / df['low']
        
        # Price changes
        features['price_change'] = df['close'].pct_change()
        features['price_change_2'] = df['close'].pct_change(2)
        features['price_change_5'] = df['close'].pct_change(5)
        features['price_change_10'] = df['close'].pct_change(10)
        
        # Log returns
        features['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price position in range
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        return features
    
    def _extract_trend_indicators(self, df: pd.DataFrame) -> Dict:
        """Extract trend indicators"""
        features = {}
        
        # Simple Moving Averages
        for period in self.indicators_config.get('sma', []):
            sma = talib.SMA(df['close'], timeperiod=period)
            features[f'sma_{period}'] = df['close'] / sma
            features[f'sma_{period}_slope'] = sma.diff() / sma.shift(1)
        
        # Exponential Moving Averages
        for period in self.indicators_config.get('ema', []):
            ema = talib.EMA(df['close'], timeperiod=period)
            features[f'ema_{period}'] = df['close'] / ema
            features[f'ema_{period}_slope'] = ema.diff() / ema.shift(1)
        
        # Weighted Moving Average
        for period in self.indicators_config.get('wma', []):
            wma = talib.WMA(df['close'], timeperiod=period)
            features[f'wma_{period}'] = df['close'] / wma
        
        # MACD
        for fast, slow, signal in self.indicators_config.get('macd', []):
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'], 
                fastperiod=fast, 
                slowperiod=slow, 
                signalperiod=signal
            )
            features[f'macd_{fast}_{slow}'] = macd
            features[f'macd_signal_{fast}_{slow}'] = macd_signal
            features[f'macd_hist_{fast}_{slow}'] = macd_hist
        
        # ADX
        for period in self.indicators_config.get('adx', []):
            features[f'adx_{period}'] = talib.ADX(
                df['high'], df['low'], df['close'], timeperiod=period
            )
            features[f'plus_di_{period}'] = talib.PLUS_DI(
                df['high'], df['low'], df['close'], timeperiod=period
            )
            features[f'minus_di_{period}'] = talib.MINUS_DI(
                df['high'], df['low'], df['close'], timeperiod=period
            )
        
        # Parabolic SAR
        for accel, maximum in self.indicators_config.get('psar', []):
            features[f'psar_{accel}_{maximum}'] = talib.SAR(
                df['high'], df['low'], acceleration=accel, maximum=maximum
            )
        
        # Ichimoku
        if self.indicators_config.get('ichimoku', False):
            features.update(self._calculate_ichimoku(df))
        
        # Supertrend
        for period, multiplier in self.indicators_config.get('supertrend', []):
            features[f'supertrend_{period}_{multiplier}'] = self._calculate_supertrend(
                df, period, multiplier
            )
        
        return features
    
    def _extract_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Extract momentum indicators"""
        features = {}
        
        # RSI
        for period in self.indicators_config.get('rsi', []):
            features[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
        
        # Stochastic
        for k_period, d_period in self.indicators_config.get('stochastic', []):
            slowk, slowd = talib.STOCH(
                df['high'], df['low'], df['close'],
                fastk_period=k_period, slowk_period=d_period, slowd_period=d_period
            )
            features[f'stoch_k_{k_period}'] = slowk
            features[f'stoch_d_{k_period}'] = slowd
        
        # Williams %R
        for period in self.indicators_config.get('williams_r', []):
            features[f'williams_r_{period}'] = talib.WILLR(
                df['high'], df['low'], df['close'], timeperiod=period
            )
        
        # ROC
        for period in self.indicators_config.get('roc', []):
            features[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)
        
        # Momentum
        for period in self.indicators_config.get('momentum', []):
            features[f'momentum_{period}'] = talib.MOM(df['close'], timeperiod=period)
        
        # CCI
        for period in self.indicators_config.get('cci', []):
            features[f'cci_{period}'] = talib.CCI(
                df['high'], df['low'], df['close'], timeperiod=period
            )
        
        # Aroon
        for period in self.indicators_config.get('aroon', []):
            aroon_up, aroon_down = talib.AROON(
                df['high'], df['low'], timeperiod=period
            )
            features[f'aroon_up_{period}'] = aroon_up
            features[f'aroon_down_{period}'] = aroon_down
            features[f'aroon_osc_{period}'] = talib.AROONOSC(
                df['high'], df['low'], timeperiod=period
            )
        
        return features
    
    def _extract_volatility_indicators(self, df: pd.DataFrame) -> Dict:
        """Extract volatility indicators"""
        features = {}
        
        # Bollinger Bands
        for period, std_dev in self.indicators_config.get('bollinger_bands', []):
            upper, middle, lower = talib.BBANDS(
                df['close'], timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev
            )
            features[f'bb_upper_{period}_{std_dev}'] = df['close'] / upper
            features[f'bb_lower_{period}_{std_dev}'] = df['close'] / lower
            features[f'bb_width_{period}_{std_dev}'] = (upper - lower) / middle
            features[f'bb_position_{period}_{std_dev}'] = (df['close'] - lower) / (upper - lower)
        
        # ATR
        for period in self.indicators_config.get('atr', []):
            features[f'atr_{period}'] = talib.ATR(
                df['high'], df['low'], df['close'], timeperiod=period
            )
            features[f'atr_ratio_{period}'] = features[f'atr_{period}'] / df['close']
        
        # Keltner Channels
        for period, multiplier in self.indicators_config.get('keltner_channels', []):
            kc_upper, kc_lower = self._calculate_keltner_channels(df, period, multiplier)
            features[f'kc_upper_{period}_{multiplier}'] = df['close'] / kc_upper
            features[f'kc_lower_{period}_{multiplier}'] = df['close'] / kc_lower
        
        # Donchian Channels
        for period in self.indicators_config.get('donchian_channels', []):
            dc_upper = df['high'].rolling(window=period).max()
            dc_lower = df['low'].rolling(window=period).min()
            features[f'dc_upper_{period}'] = df['close'] / dc_upper
            features[f'dc_lower_{period}'] = df['close'] / dc_lower
            features[f'dc_position_{period}'] = (df['close'] - dc_lower) / (dc_upper - dc_lower)
        
        # Historical Volatility
        features['volatility_20'] = df['close'].pct_change().rolling(20).std()
        features['volatility_50'] = df['close'].pct_change().rolling(50).std()
        
        return features
    
    def _extract_volume_indicators(self, df: pd.DataFrame) -> Dict:
        """Extract volume indicators"""
        features = {}
        
        # OBV
        if self.indicators_config.get('obv', False):
            features['obv'] = talib.OBV(df['close'], df['volume'])
            features['obv_ema'] = talib.EMA(features['obv'], timeperiod=21)
            features['obv_signal'] = features['obv'] / features['obv_ema']
        
        # MFI
        for period in self.indicators_config.get('mfi', []):
            features[f'mfi_{period}'] = talib.MFI(
                df['high'], df['low'], df['close'], df['volume'], timeperiod=period
            )
        
        # VWAP
        if self.indicators_config.get('vwap', False):
            features['vwap'] = self._calculate_vwap(df)
            features['price_vwap_ratio'] = df['close'] / features['vwap']
        
        # Volume SMA
        for period in self.indicators_config.get('volume_sma', []):
            vol_sma = talib.SMA(df['volume'], timeperiod=period)
            features[f'volume_sma_{period}'] = df['volume'] / vol_sma
        
        # CMF
        for period in self.indicators_config.get('cmf', []):
            features[f'cmf_{period}'] = self._calculate_cmf(df, period)
        
        # Force Index
        for period in self.indicators_config.get('fi', []):
            features[f'fi_{period}'] = self._calculate_force_index(df, period)
        
        # Volume Rate of Change
        features['volume_roc'] = talib.ROC(df['volume'], timeperiod=10)
        
        return features
    
    def _extract_market_structure(self, df: pd.DataFrame) -> Dict:
        """Extract market structure features"""
        features = {}
        
        # Pivot Points
        if self.indicators_config.get('pivot_points', False):
            pp_features = self._calculate_pivot_points(df)
            features.update(pp_features)
        
        # Support and Resistance
        n_levels = self.indicators_config.get('support_resistance', 3)
        if n_levels > 0:
            sr_features = self._calculate_support_resistance(df, n_levels)
            features.update(sr_features)
        
        # Fibonacci Retracement
        if self.indicators_config.get('fibonacci_retracement', False):
            fib_features = self._calculate_fibonacci_levels(df)
            features.update(fib_features)
        
        # Heikin Ashi
        if self.indicators_config.get('heikin_ashi', False):
            ha_features = self._calculate_heikin_ashi(df)
            features.update(ha_features)
        
        return features
    
    def _extract_mtf_features(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Extract multi-timeframe features"""
        features = {}
        
        # Get current timeframe in minutes
        from ..config import TIMEFRAMES
        current_tf_minutes = TIMEFRAMES.get(timeframe, 60)
        
        # Extract features from higher timeframes
        for tf_name, tf_minutes in TIMEFRAMES.items():
            if tf_minutes > current_tf_minutes and tf_minutes <= current_tf_minutes * 4:
                # Resample to higher timeframe
                resampled = self._resample_data(df, tf_minutes)
                
                # Calculate basic indicators on higher timeframe
                features[f'mtf_{tf_name}_rsi'] = talib.RSI(
                    resampled['close'], timeperiod=14
                ).reindex(df.index, method='ffill')
                
                features[f'mtf_{tf_name}_trend'] = (
                    resampled['close'] / talib.SMA(resampled['close'], timeperiod=20)
                ).reindex(df.index, method='ffill')
        
        return features
    
    # Helper methods
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> Dict:
        """Calculate Ichimoku Cloud"""
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        tenkan_sen = (high_9 + low_9) / 2
        
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        kijun_sen = (high_26 + low_26) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)
        
        chikou_span = df['close'].shift(-26)
        
        return {
            'ichimoku_tenkan': df['close'] / tenkan_sen,
            'ichimoku_kijun': df['close'] / kijun_sen,
            'ichimoku_senkou_a': df['close'] / senkou_span_a,
            'ichimoku_senkou_b': df['close'] / senkou_span_b,
            'ichimoku_cloud_thickness': (senkou_span_a - senkou_span_b) / df['close']
        }
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_supertrend_jit(high, low, close, atr, period, multiplier):
        """JIT compiled Supertrend calculation"""
        hl_avg = (high + low) / 2
        final_ub = np.zeros_like(close)
        final_lb = np.zeros_like(close)
        supertrend = np.zeros_like(close)
        
        for i in range(period, len(close)):
            basic_ub = hl_avg[i] + multiplier * atr[i]
            basic_lb = hl_avg[i] - multiplier * atr[i]
            
            if i == period:
                final_ub[i] = basic_ub
                final_lb[i] = basic_lb
            else:
                final_ub[i] = basic_ub if basic_ub < final_ub[i-1] or close[i-1] > final_ub[i-1] else final_ub[i-1]
                final_lb[i] = basic_lb if basic_lb > final_lb[i-1] or close[i-1] < final_lb[i-1] else final_lb[i-1]
            
            if close[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            else:
                supertrend[i] = final_lb[i]
        
        return supertrend
    
    def _calculate_supertrend(self, df: pd.DataFrame, period: int, multiplier: float) -> pd.Series:
        """Calculate Supertrend indicator"""
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        
        supertrend = self._calculate_supertrend_jit(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            atr.values,
            period,
            multiplier
        )
        
        return pd.Series(supertrend, index=df.index) / df['close']
    
    def _calculate_keltner_channels(self, df: pd.DataFrame, period: int, multiplier: float) -> Tuple[pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        ema = talib.EMA(df['close'], timeperiod=period)
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        
        upper = ema + (multiplier * atr)
        lower = ema - (multiplier * atr)
        
        return upper, lower
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    def _calculate_cmf(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mf_volume = mf_multiplier * df['volume']
        return mf_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    
    def _calculate_force_index(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Force Index"""
        fi = df['close'].diff() * df['volume']
        return talib.EMA(fi, timeperiod=period)
    
    def _calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """Calculate Pivot Points"""
        pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        
        r1 = 2 * pivot - df['low'].shift(1)
        r2 = pivot + (df['high'].shift(1) - df['low'].shift(1))
        r3 = r1 + (df['high'].shift(1) - df['low'].shift(1))
        
        s1 = 2 * pivot - df['high'].shift(1)
        s2 = pivot - (df['high'].shift(1) - df['low'].shift(1))
        s3 = s1 - (df['high'].shift(1) - df['low'].shift(1))
        
        return {
            'pivot': df['close'] / pivot,
            'r1': df['close'] / r1,
            'r2': df['close'] / r2,
            's1': df['close'] / s1,
            's2': df['close'] / s2
        }
    
    def _calculate_support_resistance(self, df: pd.DataFrame, n_levels: int) -> Dict:
        """Calculate Support and Resistance levels"""
        features = {}
        
        # Find local maxima and minima
        window = 20
        local_max = df['high'].rolling(window=window, center=True).max() == df['high']
        local_min = df['low'].rolling(window=window, center=True).min() == df['low']
        
        resistance_levels = df['high'][local_max].nlargest(n_levels)
        support_levels = df['low'][local_min].nsmallest(n_levels)
        
        for i, level in enumerate(resistance_levels):
            features[f'resistance_{i+1}'] = df['close'] / level
        
        for i, level in enumerate(support_levels):
            features[f'support_{i+1}'] = df['close'] / level
        
        return features
    
    def _calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate Fibonacci retracement levels"""
        # Use 50-period high and low
        period = 50
        high = df['high'].rolling(window=period).max()
        low = df['low'].rolling(window=period).min()
        diff = high - low
        
        fib_levels = {
            'fib_0': low,
            'fib_236': low + 0.236 * diff,
            'fib_382': low + 0.382 * diff,
            'fib_500': low + 0.500 * diff,
            'fib_618': low + 0.618 * diff,
            'fib_786': low + 0.786 * diff,
            'fib_1000': high
        }
        
        features = {}
        for name, level in fib_levels.items():
            features[name] = df['close'] / level
        
        return features
    
    def _calculate_heikin_ashi(self, df: pd.DataFrame) -> Dict:
        """Calculate Heikin Ashi candles"""
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        ha_open = pd.Series(index=df.index, dtype=float)
        ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2
        
        for i in range(1, len(df)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2
        
        ha_high = pd.concat([df['high'], ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([df['low'], ha_open, ha_close], axis=1).min(axis=1)
        
        return {
            'ha_close': ha_close / df['close'],
            'ha_trend': (ha_close > ha_open).astype(int),
            'ha_body_size': np.abs(ha_close - ha_open) / df['close']
        }
    
    def _resample_data(self, df: pd.DataFrame, target_minutes: int) -> pd.DataFrame:
        """Resample data to different timeframe"""
        rule = f'{target_minutes}T'
        
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return resampled.dropna()
    
    def _features_to_array(self, features: Dict, df: pd.DataFrame) -> np.ndarray:
        """Convert features dictionary to numpy array"""
        # Ensure all features have the same length
        feature_list = []
        
        for name, values in features.items():
            if isinstance(values, pd.Series):
                feature_list.append(values.values.reshape(-1, 1))
            else:
                feature_list.append(np.array(values).reshape(-1, 1))
        
        # Concatenate all features
        feature_array = np.concatenate(feature_list, axis=1)
        
        # Handle NaN values
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feature_array
    
    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using robust scaling"""
        # Use percentiles for robust scaling
        percentile_25 = np.percentile(features, 25, axis=0)
        percentile_75 = np.percentile(features, 75, axis=0)
        
        iqr = percentile_75 - percentile_25
        iqr[iqr == 0] = 1  # Avoid division by zero
        
        median = np.median(features, axis=0)
        
        scaled = (features - median) / iqr
        
        # Clip extreme values
        scaled = np.clip(scaled, -10, 10)
        
        return scaled
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        # This would need to be implemented based on actual features extracted
        return list(self._cache.keys()) if self._cache else []