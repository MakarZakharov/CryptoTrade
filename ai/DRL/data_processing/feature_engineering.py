"""
Генерация признаков и технических индикаторов для DRL торговли.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

# Try to import pandas_ta, fallback to simple indicators if it fails
try:
    import pandas_ta as ta
    USE_PANDAS_TA = True
except ImportError as e:
    print(f"Warning: pandas_ta not available ({e}). Using simple indicators.")
    from . import simple_indicators
    USE_PANDAS_TA = False


class FeatureEngineer:
    """Класс для генерации признаков из исторических данных."""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Настройка логирования."""
        logger = logging.getLogger('FeatureEngineer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление технических индикаторов к данным.
        
        Args:
            df: DataFrame с OHLCV данными
            
        Returns:
            DataFrame с добавленными техническими индикаторами
        """
        data = df.copy()
        
        try:
            if USE_PANDAS_TA:
                # Using pandas_ta library
                # Moving Averages (Скользящие средние)
                data['sma_7'] = ta.sma(data['close'], length=7)
                data['sma_21'] = ta.sma(data['close'], length=21)
                data['sma_50'] = ta.sma(data['close'], length=50)
                data['ema_12'] = ta.ema(data['close'], length=12)
                data['ema_26'] = ta.ema(data['close'], length=26)
                
                # RSI (Relative Strength Index)
                data['rsi_14'] = ta.rsi(data['close'], length=14)
                data['rsi_30'] = ta.rsi(data['close'], length=30)
                
                # MACD (Moving Average Convergence Divergence)
                macd = ta.macd(data['close'])
                data['macd'] = macd['MACD_12_26_9']
                data['macd_signal'] = macd['MACDs_12_26_9']
                data['macd_histogram'] = macd['MACDh_12_26_9']
                
                # Bollinger Bands
                bbands = ta.bbands(data['close'], length=20)
                data['bb_upper'] = bbands['BBU_20_2.0']
                data['bb_middle'] = bbands['BBM_20_2.0']
                data['bb_lower'] = bbands['BBL_20_2.0']
                data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
                
                # Average True Range (ATR)
                data['atr'] = ta.atr(data['high'], data['low'], data['close'])
                
                # Volume indicators
                data['volume_sma'] = ta.sma(data['volume'], length=20)
                data['volume_ratio'] = data['volume'] / data['volume_sma']
                
            else:
                # Using simple indicators as fallback
                # Moving Averages (Скользящие средние)
                data['sma_7'] = simple_indicators.simple_moving_average(data['close'], 7)
                data['sma_21'] = simple_indicators.simple_moving_average(data['close'], 21)
                data['sma_50'] = simple_indicators.simple_moving_average(data['close'], 50)
                data['ema_12'] = simple_indicators.exponential_moving_average(data['close'], 12)
                data['ema_26'] = simple_indicators.exponential_moving_average(data['close'], 26)
                
                # RSI (Relative Strength Index)
                data['rsi_14'] = simple_indicators.rsi(data['close'], 14)
                data['rsi_30'] = simple_indicators.rsi(data['close'], 30)
                
                # MACD (Moving Average Convergence Divergence)
                macd, macd_signal, macd_histogram = simple_indicators.macd(data['close'])
                data['macd'] = macd
                data['macd_signal'] = macd_signal
                data['macd_histogram'] = macd_histogram
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = simple_indicators.bollinger_bands(data['close'], 20)
                data['bb_upper'] = bb_upper
                data['bb_middle'] = bb_middle
                data['bb_lower'] = bb_lower
                data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
                
                # Average True Range (ATR)
                data['atr'] = simple_indicators.atr(data['high'], data['low'], data['close'])
                
                # Volume indicators
                data['volume_sma'] = simple_indicators.simple_moving_average(data['volume'], 20)
                data['volume_ratio'] = data['volume'] / data['volume_sma']
            
            self.logger.info("Технические индикаторы добавлены успешно")
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления технических индикаторов: {e}")
            
        return data
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление ценовых признаков."""
        data = df.copy()
        
        try:
            # Процентные изменения цены
            data['price_change_1h'] = data['close'].pct_change(1)
            data['price_change_4h'] = data['close'].pct_change(4)
            data['price_change_24h'] = data['close'].pct_change(24)
            
            # Высокие и низкие цены относительно закрытия
            data['high_low_ratio'] = (data['high'] - data['low']) / data['close']
            data['open_close_ratio'] = (data['close'] - data['open']) / data['open']
            
            # Логарифмические доходности
            data['log_return'] = np.log(data['close'] / data['close'].shift(1))
            
            # Кумулятивные доходности
            data['cumulative_return'] = (1 + data['price_change_1h']).cumprod() - 1
            
            # Реализованная волатильность
            data['realized_volatility'] = data['log_return'].rolling(window=24).std()
            
            self.logger.info("Ценовые признаки добавлены успешно")
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления ценовых признаков: {e}")
            
        return data
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление временных признаков."""
        data = df.copy()
        
        try:
            # Циклические признаки времени
            data['hour'] = data.index.hour
            data['day_of_week'] = data.index.dayofweek
            data['day_of_month'] = data.index.day
            data['month'] = data.index.month
            
            # Синусоидальное представление времени
            data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
            
            data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
            
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            
            # Является ли торговым днем (упрощенная версия)
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
            
            self.logger.info("Временные признаки добавлены успешно")
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления временных признаков: {e}")
            
        return data
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление признаков волатильности."""
        data = df.copy()
        
        try:
            # Различные периоды волатильности
            for period in [7, 14, 30]:
                data[f'volatility_{period}'] = data['close'].rolling(window=period).std()
                data[f'volatility_ratio_{period}'] = data[f'volatility_{period}'] / data['close']
            
            # Волатильность высоких и низких цен
            data['hl_volatility'] = (data['high'] - data['low']).rolling(window=14).std()
            
            # Парkinson волатильность (более эффективная оценка)
            data['parkinson_vol'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                (np.log(data['high'] / data['low'])).rolling(window=20).var()
            )
            
            # Garman-Klass волатильность
            data['gk_vol'] = np.sqrt(
                (np.log(data['high'] / data['close']) * np.log(data['high'] / data['open']) +
                 np.log(data['low'] / data['close']) * np.log(data['low'] / data['open']))
                .rolling(window=20).mean()
            )
            
            self.logger.info("Признаки волатильности добавлены успешно")
            
        except Exception as e:
            self.logger.error(f"Ошибка добавления признаков волатильности: {e}")
            
        return data
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавление всех признаков к данным."""
        self.logger.info("Начало генерации всех признаков")
        
        data = df.copy()
        
        # Добавляем все виды признаков
        data = self.add_technical_indicators(data)
        data = self.add_price_features(data)
        data = self.add_time_features(data)
        data = self.add_volatility_features(data)
        
        # Удаляем строки с NaN значениями
        initial_rows = len(data)
        data = data.dropna()
        final_rows = len(data)
        
        self.logger.info(f"Генерация признаков завершена. Удалено {initial_rows - final_rows} строк с NaN")
        self.logger.info(f"Итоговое количество признаков: {len(data.columns)}")
        
        return data


class DataNormalizer:
    """Класс для нормализации данных."""
    
    def __init__(self):
        self.scalers = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Настройка логирования."""
        logger = logging.getLogger('DataNormalizer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def normalize_features(
        self, 
        df: pd.DataFrame, 
        method: str = 'minmax',
        exclude_columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Нормализация признаков.
        
        Args:
            df: DataFrame с признаками
            method: Метод нормализации ('minmax', 'standard', 'robust')
            exclude_columns: Колонки для исключения из нормализации
            
        Returns:
            Нормализованный DataFrame
        """
        data = df.copy()
        exclude_columns = exclude_columns or []
        
        try:
            from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
            
            # Выбор скейлера
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                raise ValueError(f"Неизвестный метод нормализации: {method}")
            
            # Колонки для нормализации
            columns_to_normalize = [col for col in data.columns if col not in exclude_columns]
            
            # Нормализация
            data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
            
            # Сохранение скейлера для обратного преобразования
            self.scalers[method] = scaler
            
            self.logger.info(f"Нормализация {method} применена к {len(columns_to_normalize)} колонкам")
            
        except Exception as e:
            self.logger.error(f"Ошибка нормализации: {e}")
            
        return data


def main():
    """Пример использования генератора признаков."""
    # Создание тестовых данных
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    
    test_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Обеспечиваем логичность OHLC данных
    test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
    test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
    
    print("Исходные данные:")
    print(test_data.head())
    print(f"Форма данных: {test_data.shape}")
    
    # Генерация признаков
    feature_engineer = FeatureEngineer()
    enhanced_data = feature_engineer.add_all_features(test_data)
    
    print(f"\nПосле добавления признаков:")
    print(f"Форма данных: {enhanced_data.shape}")
    print(f"Колонки: {list(enhanced_data.columns)}")
    
    # Нормализация
    normalizer = DataNormalizer()
    normalized_data = normalizer.normalize_features(enhanced_data, method='minmax')
    
    print(f"\nПосле нормализации:")
    print(normalized_data.describe())


if __name__ == "__main__":
    main()