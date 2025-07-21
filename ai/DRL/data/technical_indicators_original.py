"""Технические индикаторы для DRL системы."""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple, Dict
from ..utils import DRLLogger


class TechnicalIndicators:
    """Класс для расчета технических индикаторов."""
    
    def __init__(self, logger: Optional[DRLLogger] = None):
        """Инициализация класса индикаторов."""
        self.logger = logger or DRLLogger("technical_indicators")
    
    @staticmethod
    def sma(data: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
        """Простая скользящая средняя (Simple Moving Average)."""
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: Union[pd.Series, np.ndarray], period: int, alpha: Optional[float] = None) -> pd.Series:
        """Экспоненциальная скользящая средняя (Exponential Moving Average)."""
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        
        if alpha is None:
            alpha = 2.0 / (period + 1)
        
        return data.ewm(alpha=alpha, adjust=False).mean()
    
    @staticmethod
    def rsi(data: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
        """Индекс относительной силы (Relative Strength Index)."""
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Используем экспоненциальное скользящее среднее как в стандартном RSI
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
        
        # Избегаем деления на ноль
        rs = avg_gain / (avg_loss + 1e-14)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(data: Union[pd.Series, np.ndarray], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD индикатор.
        
        Returns:
            Tuple[macd_line, signal_line, histogram]
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: Union[pd.Series, np.ndarray], period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Полосы Боллинджера.
        
        Returns:
            Tuple[upper_band, middle_band, lower_band]
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(data)
        
        middle_band = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def atr(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
            close: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
        """Average True Range."""
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        
        high_low = high - low
        high_close_prev = np.abs(high - close.shift(1))
        low_close_prev = np.abs(low - close.shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def stochastic(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
                   close: Union[pd.Series, np.ndarray], k_period: int = 14, 
                   d_period: int = 3, smooth_k: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Стохастический осциллятор.
        
        Returns:
            Tuple[%K, %D]
        """
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_percent_smooth = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent_smooth.rolling(window=d_period).mean()
        
        return k_percent_smooth, d_percent
    
    @staticmethod
    def adx(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
            close: Union[pd.Series, np.ndarray], period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index.
        
        Returns:
            Tuple[ADX, +DI, -DI]
        """
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        
        # True Range
        atr_val = TechnicalIndicators.atr(high, low, close, period)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=high.index).rolling(window=period).mean()
        minus_dm = pd.Series(minus_dm, index=low.index).rolling(window=period).mean()
        
        # Directional Indicators - избегаем деления на ноль
        plus_di = 100 * (plus_dm / (atr_val + 1e-14))
        minus_di = 100 * (minus_dm / (atr_val + 1e-14))
        
        # ADX - избегаем деления на ноль
        dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di + 1e-14))
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def obv(close: Union[pd.Series, np.ndarray], volume: Union[pd.Series, np.ndarray]) -> pd.Series:
        """On-Balance Volume."""
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)
        
        price_change = close.diff()
        obv_values = np.where(price_change > 0, volume, 
                             np.where(price_change < 0, -volume, 0))
        
        return pd.Series(obv_values, index=close.index).cumsum()
    
    @staticmethod
    def vwap(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
             close: Union[pd.Series, np.ndarray], volume: Union[pd.Series, np.ndarray], 
             period: int = 20) -> pd.Series:
        """Volume Weighted Average Price."""
        if isinstance(high, np.ndarray):
            high = pd.Series(high)
        if isinstance(low, np.ndarray):
            low = pd.Series(low)
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        if isinstance(volume, np.ndarray):
            volume = pd.Series(volume)
        
        typical_price = (high + low + close) / 3
        volume_price = typical_price * volume
        
        cum_volume_price = volume_price.rolling(window=period).sum()
        cum_volume = volume.rolling(window=period).sum()
        
        # Избегаем деления на ноль
        vwap = cum_volume_price / (cum_volume + 1e-14)
        
        return vwap
    
    def add_all_indicators(self, df: pd.DataFrame, 
                          indicators: Optional[Dict[str, Union[int, list]]] = None) -> pd.DataFrame:
        """
        Добавить все технические индикаторы к DataFrame.
        
        Args:
            df: DataFrame с OHLCV данными
            indicators: Словарь с настройками индикаторов
            
        Returns:
            DataFrame с добавленными индикаторами
        """
        # Валидация входных данных
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")
        
        if df.empty:
            raise ValueError("DataFrame не может быть пустым")
        
        if indicators is None:
            indicators = {
                'sma': [20, 50],
                'ema': [12, 26],
                'rsi': [14],
                'macd': [12, 26, 9],
                'bollinger': [20],
                'atr': [14],
                'stochastic': [14, 3, 3],
                'adx': [14],
                'obv': [],
                'vwap': [20]
            }
        
        # Создаем копию с оптимизацией памяти
        result_df = df.copy()
        
        # Оптимизация типов данных для новых колонок
        original_memory = result_df.memory_usage(deep=True).sum()
        self.logger.debug(f"Память до добавления индикаторов: {original_memory / 1024**2:.2f} MB")
        
        try:
            # Простые скользящие средние
            if 'sma' in indicators:
                periods = indicators['sma'] if isinstance(indicators['sma'], list) else [indicators['sma']]
                for period in periods:
                    result_df[f'sma_{period}'] = self.sma(df['close'], period)
            
            # Экспоненциальные скользящие средние
            if 'ema' in indicators:
                periods = indicators['ema'] if isinstance(indicators['ema'], list) else [indicators['ema']]
                for period in periods:
                    result_df[f'ema_{period}'] = self.ema(df['close'], period)
            
            # RSI
            if 'rsi' in indicators:
                periods = indicators['rsi'] if isinstance(indicators['rsi'], list) else [indicators['rsi']]
                for period in periods:
                    result_df[f'rsi_{period}'] = self.rsi(df['close'], period)
            
            # MACD
            if 'macd' in indicators:
                params = indicators['macd'] if isinstance(indicators['macd'], list) else [12, 26, 9]
                if len(params) >= 3:
                    macd_line, signal_line, histogram = self.macd(df['close'], params[0], params[1], params[2])
                    result_df['macd'] = macd_line
                    result_df['macd_signal'] = signal_line
                    result_df['macd_histogram'] = histogram
            
            # Bollinger Bands
            if 'bollinger' in indicators:
                periods = indicators['bollinger'] if isinstance(indicators['bollinger'], list) else [indicators['bollinger']]
                for period in periods:
                    upper, middle, lower = self.bollinger_bands(df['close'], period)
                    result_df[f'bb_upper_{period}'] = upper
                    result_df[f'bb_middle_{period}'] = middle
                    result_df[f'bb_lower_{period}'] = lower
            
            # ATR
            if 'atr' in indicators:
                periods = indicators['atr'] if isinstance(indicators['atr'], list) else [indicators['atr']]
                for period in periods:
                    result_df[f'atr_{period}'] = self.atr(df['high'], df['low'], df['close'], period)
            
            # Stochastic
            if 'stochastic' in indicators:
                params = indicators['stochastic'] if isinstance(indicators['stochastic'], list) else [14, 3, 3]
                if len(params) >= 3:
                    k_percent, d_percent = self.stochastic(df['high'], df['low'], df['close'], 
                                                         params[0], params[1], params[2])
                    result_df['stoch_k'] = k_percent
                    result_df['stoch_d'] = d_percent
            
            # ADX
            if 'adx' in indicators:
                periods = indicators['adx'] if isinstance(indicators['adx'], list) else [indicators['adx']]
                for period in periods:
                    adx_val, plus_di, minus_di = self.adx(df['high'], df['low'], df['close'], period)
                    result_df[f'adx_{period}'] = adx_val
                    result_df[f'plus_di_{period}'] = plus_di
                    result_df[f'minus_di_{period}'] = minus_di
            
            # OBV
            if 'obv' in indicators and 'volume' in df.columns:
                result_df['obv'] = self.obv(df['close'], df['volume'])
            
            # VWAP
            if 'vwap' in indicators and 'volume' in df.columns:
                periods = indicators['vwap'] if isinstance(indicators['vwap'], list) else [indicators['vwap']]
                for period in periods:
                    result_df[f'vwap_{period}'] = self.vwap(df['high'], df['low'], df['close'], df['volume'], period)
            
            # Дополнительные фичи
            result_df['price_change'] = df['close'].pct_change()
            result_df['volume_change'] = df['volume'].pct_change()
            result_df['high_low_ratio'] = df['high'] / df['low']
            result_df['close_open_ratio'] = df['close'] / df['open']
            
            # Оптимизация типов данных для экономии памяти
            float_columns = result_df.select_dtypes(include=['float64']).columns
            result_df[float_columns] = result_df[float_columns].astype('float32')
            
            final_memory = result_df.memory_usage(deep=True).sum()
            memory_saved = (original_memory - final_memory) / 1024**2
            
            self.logger.info(f"Добавлено {len(result_df.columns) - len(df.columns)} технических индикаторов")
            self.logger.debug(f"Память после оптимизации: {final_memory / 1024**2:.2f} MB (сэкономлено: {memory_saved:.2f} MB)")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Ошибка при добавлении индикаторов: {e}")
            return df
    
    def get_feature_importance_score(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Получить оценку важности технических индикаторов.
        Простая оценка на основе корреляции с будущими ценовыми изменениями.
        """
        if 'close' in df.columns:
            # Создаем целевую переменную - будущее изменение цены
            future_returns = df['close'].shift(-1).pct_change()
            
            importance_scores = {}
            
            # Находим все технические индикаторы (не базовые OHLCV колонки)
            base_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            technical_columns = [col for col in df.columns if col not in base_columns]
            
            for col in technical_columns:
                if df[col].notna().sum() > 100:  # Минимум 100 не-NaN значений
                    correlation = df[col].corr(future_returns)
                    if not np.isnan(correlation):
                        importance_scores[col] = abs(correlation)
            
            # Сортируем по важности
            return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
        
        return {}