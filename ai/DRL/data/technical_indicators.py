"""Технические индикаторы для DRL системы с использованием TA-Lib."""

import numpy as np
import pandas as pd
import talib
from typing import Union, Optional, Tuple, Dict
from ..utils import DRLLogger


class TechnicalIndicators:
    """Класс для расчета технических индикаторов."""
    
    def __init__(self, logger: Optional[DRLLogger] = None):
        """Инициализация класса индикаторов."""
        self.logger = logger or DRLLogger("technical_indicators")
    
    @staticmethod
    def sma(data: Union[pd.Series, np.ndarray], period: int) -> pd.Series:
        """Простая скользящая средняя (Simple Moving Average) с использованием TA-Lib."""
        if isinstance(data, pd.Series):
            data_values = data.values
            index = data.index
        else:
            data_values = np.asarray(data, dtype=np.float64)
            index = None
        
        result = talib.SMA(data_values, timeperiod=period)
        
        if index is not None:
            return pd.Series(result, index=index)
        else:
            return pd.Series(result)
    
    @staticmethod
    def ema(data: Union[pd.Series, np.ndarray], period: int, alpha: Optional[float] = None) -> pd.Series:
        """Экспоненциальная скользящая средняя (Exponential Moving Average) с использованием TA-Lib."""
        if isinstance(data, pd.Series):
            data_values = data.values
            index = data.index
        else:
            data_values = np.asarray(data, dtype=np.float64)
            index = None
        
        result = talib.EMA(data_values, timeperiod=period)
        
        if index is not None:
            return pd.Series(result, index=index)
        else:
            return pd.Series(result)
    
    @staticmethod
    def rsi(data: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
        """Индекс относительной силы (Relative Strength Index) с использованием TA-Lib."""
        if isinstance(data, pd.Series):
            data_values = data.values
            index = data.index
        else:
            data_values = np.asarray(data, dtype=np.float64)
            index = None
        
        result = talib.RSI(data_values, timeperiod=period)
        
        if index is not None:
            return pd.Series(result, index=index)
        else:
            return pd.Series(result)
    
    @staticmethod
    def macd(data: Union[pd.Series, np.ndarray], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD индикатор с использованием TA-Lib.
        
        Returns:
            Tuple[macd_line, signal_line, histogram]
        """
        if isinstance(data, pd.Series):
            data_values = data.values
            index = data.index
        else:
            data_values = np.asarray(data, dtype=np.float64)
            index = None
        
        macd, macdsignal, macdhist = talib.MACD(data_values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        
        if index is not None:
            return pd.Series(macd, index=index), pd.Series(macdsignal, index=index), pd.Series(macdhist, index=index)
        else:
            return pd.Series(macd), pd.Series(macdsignal), pd.Series(macdhist)
    
    @staticmethod
    def bollinger_bands(data: Union[pd.Series, np.ndarray], period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Полосы Боллинджера с использованием TA-Lib.
        
        Returns:
            Tuple[upper_band, middle_band, lower_band]
        """
        if isinstance(data, pd.Series):
            data_values = data.values
            index = data.index
        else:
            data_values = np.asarray(data, dtype=np.float64)
            index = None
        
        upper, middle, lower = talib.BBANDS(data_values, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0)
        
        if index is not None:
            return pd.Series(upper, index=index), pd.Series(middle, index=index), pd.Series(lower, index=index)
        else:
            return pd.Series(upper), pd.Series(middle), pd.Series(lower)
    
    @staticmethod
    def atr(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
            close: Union[pd.Series, np.ndarray], period: int = 14) -> pd.Series:
        """Average True Range с использованием TA-Lib."""
        # Приведение к numpy arrays для TA-Lib
        if isinstance(high, pd.Series):
            high_values = high.values
            index = high.index
        else:
            high_values = np.asarray(high, dtype=np.float64)
            index = None
        
        if isinstance(low, pd.Series):
            low_values = low.values
        else:
            low_values = np.asarray(low, dtype=np.float64)
            
        if isinstance(close, pd.Series):
            close_values = close.values
        else:
            close_values = np.asarray(close, dtype=np.float64)
        
        result = talib.ATR(high_values, low_values, close_values, timeperiod=period)
        
        if index is not None:
            return pd.Series(result, index=index)
        else:
            return pd.Series(result)
    
    @staticmethod
    def stochastic(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
                   close: Union[pd.Series, np.ndarray], k_period: int = 14, 
                   d_period: int = 3, smooth_k: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Стохастический осциллятор с использованием TA-Lib.
        
        Returns:
            Tuple[%K, %D]
        """
        # Приведение к numpy arrays для TA-Lib
        if isinstance(high, pd.Series):
            high_values = high.values
            index = high.index
        else:
            high_values = np.asarray(high, dtype=np.float64)
            index = None
        
        if isinstance(low, pd.Series):
            low_values = low.values
        else:
            low_values = np.asarray(low, dtype=np.float64)
            
        if isinstance(close, pd.Series):
            close_values = close.values
        else:
            close_values = np.asarray(close, dtype=np.float64)
        
        slowk, slowd = talib.STOCH(high_values, low_values, close_values, 
                                  fastk_period=k_period, slowk_period=smooth_k, 
                                  slowk_matype=0, slowd_period=d_period, slowd_matype=0)
        
        if index is not None:
            return pd.Series(slowk, index=index), pd.Series(slowd, index=index)
        else:
            return pd.Series(slowk), pd.Series(slowd)
    
    @staticmethod
    def adx(high: Union[pd.Series, np.ndarray], low: Union[pd.Series, np.ndarray], 
            close: Union[pd.Series, np.ndarray], period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index с использованием TA-Lib.
        
        Returns:
            Tuple[ADX, +DI, -DI]
        """
        # Приведение к numpy arrays для TA-Lib
        if isinstance(high, pd.Series):
            high_values = high.values
            index = high.index
        else:
            high_values = np.asarray(high, dtype=np.float64)
            index = None
        
        if isinstance(low, pd.Series):
            low_values = low.values
        else:
            low_values = np.asarray(low, dtype=np.float64)
            
        if isinstance(close, pd.Series):
            close_values = close.values
        else:
            close_values = np.asarray(close, dtype=np.float64)
        
        # Используем TA-Lib функции для ADX и DI
        adx_val = talib.ADX(high_values, low_values, close_values, timeperiod=period)
        plus_di = talib.PLUS_DI(high_values, low_values, close_values, timeperiod=period)
        minus_di = talib.MINUS_DI(high_values, low_values, close_values, timeperiod=period)
        
        if index is not None:
            return pd.Series(adx_val, index=index), pd.Series(plus_di, index=index), pd.Series(minus_di, index=index)
        else:
            return pd.Series(adx_val), pd.Series(plus_di), pd.Series(minus_di)
    
    @staticmethod
    def obv(close: Union[pd.Series, np.ndarray], volume: Union[pd.Series, np.ndarray]) -> pd.Series:
        """On-Balance Volume с использованием TA-Lib."""
        if isinstance(close, pd.Series):
            close_values = close.values
            index = close.index
        else:
            close_values = np.asarray(close, dtype=np.float64)
            index = None
            
        if isinstance(volume, pd.Series):
            volume_values = volume.values
        else:
            volume_values = np.asarray(volume, dtype=np.float64)
        
        result = talib.OBV(close_values, volume_values)
        
        if index is not None:
            return pd.Series(result, index=index)
        else:
            return pd.Series(result)
    
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
        
        # Создаем копию с оптимизацией памяти и приведением к float64 для TA-Lib
        result_df = df.copy()
        
        # TA-Lib требует float64
        float_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in float_columns:
            if col in result_df.columns:
                result_df[col] = result_df[col].astype(np.float64)
        
        # Оптимизация типов данных для новых колонок
        original_memory = result_df.memory_usage(deep=True).sum()
        self.logger.debug(f"Память до добавления индикаторов: {original_memory / 1024**2:.2f} MB")
        
        try:
            # Простые скользящие средние
            if 'sma' in indicators:
                periods = indicators['sma'] if isinstance(indicators['sma'], list) else [indicators['sma']]
                for period in periods:
                    result_df[f'sma_{period}'] = self.sma(result_df['close'], period)
            
            # Экспоненциальные скользящие средние
            if 'ema' in indicators:
                periods = indicators['ema'] if isinstance(indicators['ema'], list) else [indicators['ema']]
                for period in periods:
                    result_df[f'ema_{period}'] = self.ema(result_df['close'], period)
            
            # RSI
            if 'rsi' in indicators:
                periods = indicators['rsi'] if isinstance(indicators['rsi'], list) else [indicators['rsi']]
                for period in periods:
                    result_df[f'rsi_{period}'] = self.rsi(result_df['close'], period)
            
            # MACD
            if 'macd' in indicators:
                params = indicators['macd'] if isinstance(indicators['macd'], list) else [12, 26, 9]
                if len(params) >= 3:
                    macd_line, signal_line, histogram = self.macd(result_df['close'], params[0], params[1], params[2])
                    result_df['macd'] = macd_line
                    result_df['macd_signal'] = signal_line
                    result_df['macd_histogram'] = histogram
            
            # Bollinger Bands
            if 'bollinger' in indicators:
                periods = indicators['bollinger'] if isinstance(indicators['bollinger'], list) else [indicators['bollinger']]
                for period in periods:
                    upper, middle, lower = self.bollinger_bands(result_df['close'], period)
                    result_df[f'bb_upper_{period}'] = upper
                    result_df[f'bb_middle_{period}'] = middle
                    result_df[f'bb_lower_{period}'] = lower
            
            # ATR
            if 'atr' in indicators:
                periods = indicators['atr'] if isinstance(indicators['atr'], list) else [indicators['atr']]
                for period in periods:
                    result_df[f'atr_{period}'] = self.atr(result_df['high'], result_df['low'], result_df['close'], period)
            
            # Stochastic
            if 'stochastic' in indicators:
                params = indicators['stochastic'] if isinstance(indicators['stochastic'], list) else [14, 3, 3]
                if len(params) >= 3:
                    k_percent, d_percent = self.stochastic(result_df['high'], result_df['low'], result_df['close'], 
                                                         params[0], params[1], params[2])
                    result_df['stoch_k'] = k_percent
                    result_df['stoch_d'] = d_percent
            
            # ADX
            if 'adx' in indicators:
                periods = indicators['adx'] if isinstance(indicators['adx'], list) else [indicators['adx']]
                for period in periods:
                    adx_val, plus_di, minus_di = self.adx(result_df['high'], result_df['low'], result_df['close'], period)
                    result_df[f'adx_{period}'] = adx_val
                    result_df[f'plus_di_{period}'] = plus_di
                    result_df[f'minus_di_{period}'] = minus_di
            
            # OBV
            if 'obv' in indicators and 'volume' in result_df.columns:
                result_df['obv'] = self.obv(result_df['close'], result_df['volume'])
            
            # VWAP
            if 'vwap' in indicators and 'volume' in result_df.columns:
                periods = indicators['vwap'] if isinstance(indicators['vwap'], list) else [indicators['vwap']]
                for period in periods:
                    result_df[f'vwap_{period}'] = self.vwap(result_df['high'], result_df['low'], result_df['close'], result_df['volume'], period)
            
            # Дополнительные фичи
            result_df['price_change'] = result_df['close'].pct_change(fill_method=None)
            result_df['volume_change'] = result_df['volume'].pct_change() if 'volume' in result_df.columns else pd.Series(index=result_df.index)
            result_df['high_low_ratio'] = result_df['high'] / result_df['low']
            result_df['close_open_ratio'] = result_df['close'] / result_df['open']
            
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
            future_returns = df['close'].shift(-1).pct_change(fill_method=None)
            
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