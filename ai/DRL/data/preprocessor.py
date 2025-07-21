"""Предобработчик данных для DRL системы."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from ..utils import DRLLogger, normalize_data, split_data
from ..config import TradingConfig


class DataPreprocessor:
    """Предобработчик данных для DRL обучения."""
    
    def __init__(self, config: TradingConfig, logger: Optional[DRLLogger] = None):
        """
        Инициализация предобработчика.
        
        Args:
            config: конфигурация торговых параметров
            logger: логгер для записи операций
        """
        self.config = config
        self.logger = logger or DRLLogger("data_preprocessor")
        
        # Скалеры для разных типов данных
        self.price_scaler = None
        self.volume_scaler = None
        self.indicator_scaler = None
        
        # Статистики для нормализации
        self.feature_stats = {}
        self.is_fitted = False
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создание дополнительных фичей для обучения.
        
        Args:
            df: DataFrame с базовыми данными и техническими индикаторами
            
        Returns:
            DataFrame с расширенными фичами
        """
        # Валидация входных данных
        if df.empty:
            raise ValueError("DataFrame не может быть пустым")
        
        # Проверка на достаточное количество данных для создания фичей
        min_rows = max(50, self.config.lookback_window)  # Минимум для надежных расчетов
        if len(df) < min_rows:
            self.logger.warning(f"Недостаточно данных для создания фичей: {len(df)} < {min_rows}")
        
        result_df = df.copy()
        
        # Оптимизация памяти - отслеживаем использование
        initial_memory = result_df.memory_usage(deep=True).sum() / 1024**2
        self.logger.debug(f"Начальное использование памяти: {initial_memory:.2f} MB")
        
        try:
            # Ценовые фичи
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                result_df['price_range'] = (df['high'] - df['low']) / df['close']
                result_df['body_size'] = abs(df['close'] - df['open']) / df['close']
                result_df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
                result_df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
                
                # Паттерны свечей
                result_df['is_green'] = (df['close'] > df['open']).astype(int)
                result_df['is_doji'] = (abs(df['close'] - df['open']) / df['close'] < 0.001).astype(int)
            
            # Временные фичи
            if isinstance(df.index, pd.DatetimeIndex):
                result_df['hour'] = df.index.hour / 23.0
                result_df['day_of_week'] = df.index.dayofweek / 6.0
                result_df['day_of_month'] = df.index.day / 31.0
                result_df['month'] = df.index.month / 12.0
                result_df['quarter'] = df.index.quarter / 4.0
            
            # Лаговые фичи для важных индикаторов
            lag_periods = [1, 2, 3, 5]
            important_features = ['close', 'volume']
            
            # Добавляем лаги для цены и объема
            for feature in important_features:
                if feature in df.columns:
                    for lag in lag_periods:
                        result_df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
            
            # Скользящие статистики
            for window in [5, 10, 20]:
                if 'close' in df.columns:
                    result_df[f'close_std_{window}'] = df['close'].rolling(window).std()
                    result_df[f'close_skew_{window}'] = df['close'].rolling(window).skew()
                
                if 'volume' in df.columns:
                    result_df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
                    result_df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            
            # Относительные позиции
            for window in [10, 20, 50]:
                if 'close' in df.columns:
                    rolling_min = df['close'].rolling(window).min()
                    rolling_max = df['close'].rolling(window).max()
                    result_df[f'close_position_{window}'] = (df['close'] - rolling_min) / (rolling_max - rolling_min)
            
            # Momentum фичи
            if 'close' in df.columns:
                for period in [1, 5, 10, 20]:
                    result_df[f'momentum_{period}'] = df['close'].pct_change(period)
                    result_df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
            
            # Волатильность фичи
            if 'close' in df.columns:
                for window in [5, 10, 20]:
                    returns = df['close'].pct_change()
                    result_df[f'volatility_{window}'] = returns.rolling(window).std()
                    result_df[f'realized_vol_{window}'] = returns.rolling(window).apply(lambda x: np.sqrt(np.sum(x**2)))
            
            # Оптимизация типов данных для экономии памяти
            float_columns = result_df.select_dtypes(include=['float64']).columns
            if len(float_columns) > 0:
                result_df[float_columns] = result_df[float_columns].astype('float32')
            
            final_memory = result_df.memory_usage(deep=True).sum() / 1024**2
            memory_saved = initial_memory - final_memory
            
            self.logger.info(f"Создано {len(result_df.columns) - len(df.columns)} дополнительных фичей")
            self.logger.debug(f"Память после создания фичей: {final_memory:.2f} MB (изменение: {memory_saved:+.2f} MB)")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Ошибка создания фичей: {e}")
            return df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
        """
        Обработка пропущенных значений.
        
        Args:
            df: DataFrame с данными
            method: метод обработки ("forward_fill", "interpolate", "drop", "mean")
            
        Returns:
            DataFrame без пропущенных значений
        """
        result_df = df.copy()
        
        if method == "forward_fill":
            result_df = result_df.ffill().bfill()
        elif method == "interpolate":
            result_df = result_df.interpolate(method='linear')
        elif method == "drop":
            result_df = result_df.dropna()
        elif method == "mean":
            result_df = result_df.fillna(result_df.mean())
        
        missing_before = df.isnull().sum().sum()
        missing_after = result_df.isnull().sum().sum()
        
        self.logger.info(f"Пропущенные значения: {missing_before} -> {missing_after}")
        
        return result_df
    
    def remove_outliers(self, df: pd.DataFrame, method: str = "iqr", threshold: float = 3.0) -> pd.DataFrame:
        """
        Удаление выбросов.
        
        Args:
            df: DataFrame с данными
            method: метод детекции ("iqr", "zscore", "isolation_forest")
            threshold: порог для определения выбросов
            
        Returns:
            DataFrame без выбросов
        """
        result_df = df.copy()
        
        # Выбираем только числовые колонки
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns
        
        if method == "iqr":
            for col in numeric_columns:
                Q1 = result_df[col].quantile(0.25)
                Q3 = result_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Заменяем выбросы на граничные значения (winsorizing)
                result_df[col] = result_df[col].clip(lower_bound, upper_bound)
        
        elif method == "zscore":
            for col in numeric_columns:
                z_scores = np.abs((result_df[col] - result_df[col].mean()) / result_df[col].std())
                result_df = result_df[z_scores < threshold]
        
        rows_before = len(df)
        rows_after = len(result_df)
        
        self.logger.info(f"Удаление выбросов ({method}): {rows_before} -> {rows_after} строк")
        
        return result_df
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Нормализация фичей.
        
        Args:
            df: DataFrame с данными
            fit: обучить скалеры на данных
            
        Returns:
            Нормализованный DataFrame
        """
        result_df = df.copy()
        
        # Разделяем колонки по типам
        price_columns = [col for col in df.columns if any(price_word in col.lower() 
                        for price_word in ['open', 'high', 'low', 'close', 'price', 'sma', 'ema', 'bb_', 'vwap'])]
        
        volume_columns = [col for col in df.columns if any(vol_word in col.lower() 
                         for vol_word in ['volume', 'obv'])]
        
        indicator_columns = [col for col in df.columns if col not in price_columns + volume_columns 
                           and df[col].dtype in ['float64', 'int64']]
        
        if fit:
            # Инициализируем скалеры
            if self.config.normalization_method == "standard":
                self.price_scaler = StandardScaler()
                self.volume_scaler = StandardScaler()
                self.indicator_scaler = StandardScaler()
            elif self.config.normalization_method == "minmax":
                self.price_scaler = MinMaxScaler()
                self.volume_scaler = MinMaxScaler()
                self.indicator_scaler = MinMaxScaler()
            elif self.config.normalization_method == "robust":
                self.price_scaler = RobustScaler()
                self.volume_scaler = RobustScaler()
                self.indicator_scaler = RobustScaler()
            else:  # zscore как fallback
                self.price_scaler = StandardScaler()
                self.volume_scaler = StandardScaler()
                self.indicator_scaler = StandardScaler()
        
        # Нормализация по группам
        if price_columns and self.price_scaler:
            if fit:
                result_df[price_columns] = self.price_scaler.fit_transform(df[price_columns].fillna(0))
            else:
                result_df[price_columns] = self.price_scaler.transform(df[price_columns].fillna(0))
        
        if volume_columns and self.volume_scaler:
            if fit:
                result_df[volume_columns] = self.volume_scaler.fit_transform(df[volume_columns].fillna(0))
            else:
                result_df[volume_columns] = self.volume_scaler.transform(df[volume_columns].fillna(0))
        
        if indicator_columns and self.indicator_scaler:
            if fit:
                result_df[indicator_columns] = self.indicator_scaler.fit_transform(df[indicator_columns].fillna(0))
            else:
                result_df[indicator_columns] = self.indicator_scaler.transform(df[indicator_columns].fillna(0))
        
        if fit:
            self.is_fitted = True
            self.logger.info(f"Нормализация: {len(price_columns)} ценовых, {len(volume_columns)} объемных, {len(indicator_columns)} индикаторных фичей")
        
        return result_df
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создание последовательностей для обучения.
        
        Args:
            df: DataFrame с данными
            sequence_length: длина последовательности (по умолчанию из config)
            
        Returns:
            Tuple[sequences, targets] - последовательности и целевые значения
        """
        if sequence_length is None:
            sequence_length = self.config.lookback_window
        
        # Удаляем строки с NaN
        clean_df = df.dropna()
        
        if len(clean_df) < sequence_length + 1:
            raise ValueError(f"Недостаточно данных для создания последовательностей. Нужно минимум {sequence_length + 1}, есть {len(clean_df)}")
        
        # Создаем последовательности
        sequences = []
        targets = []
        
        for i in range(len(clean_df) - sequence_length):
            # Берем последовательность
            seq = clean_df.iloc[i:i + sequence_length].values
            # Целевое значение - следующая цена (или изменение цены)
            target = clean_df.iloc[i + sequence_length]['close'] if 'close' in clean_df.columns else 0
            
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        self.logger.info(f"Создано {len(sequences)} последовательностей длиной {sequence_length}")
        
        return sequences, targets
    
    def prepare_for_drl(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Полная подготовка данных для DRL обучения.
        
        Args:
            df: исходный DataFrame
            
        Returns:
            Подготовленный DataFrame
        """
        self.logger.info("Начало полной подготовки данных для DRL")
        
        # 1. Создание дополнительных фичей
        df = self.create_features(df)
        
        # 2. Обработка пропущенных значений
        df = self.handle_missing_values(df, method="forward_fill")
        
        # 3. Удаление выбросов
        df = self.remove_outliers(df, method="iqr", threshold=2.5)
        
        # 4. Нормализация фичей
        if self.config.normalize_features:
            df = self.normalize_features(df, fit=True)
        
        # 5. Финальная очистка
        df = df.dropna()
        
        self.logger.info(f"Подготовка завершена. Итоговый размер: {df.shape}")
        
        return df
    
    def split_data_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разделение данных на обучающую, валидационную и тестовую выборки.
        
        Args:
            df: подготовленный DataFrame
            
        Returns:
            Tuple[train_df, val_df, test_df]
        """
        return split_data(
            df, 
            train_ratio=self.config.train_split,
            val_ratio=self.config.val_split,
            test_ratio=self.config.test_split
        )
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """Получить сводку по фичам."""
        summary = {
            'total_features': len(df.columns),
            'total_samples': len(df),
            'missing_values': df.isnull().sum().sum(),
            'feature_types': {
                'price_features': len([col for col in df.columns if any(word in col.lower() 
                                     for word in ['open', 'high', 'low', 'close', 'price', 'sma', 'ema'])]),
                'volume_features': len([col for col in df.columns if 'volume' in col.lower()]),
                'technical_indicators': len([col for col in df.columns if any(word in col.lower() 
                                           for word in ['rsi', 'macd', 'bb_', 'atr', 'stoch', 'adx'])]),
                'momentum_features': len([col for col in df.columns if 'momentum' in col.lower() or 'return' in col.lower()]),
                'volatility_features': len([col for col in df.columns if 'volatility' in col.lower() or 'std' in col.lower()]),
                'temporal_features': len([col for col in df.columns if any(word in col.lower() 
                                        for word in ['hour', 'day', 'week', 'month', 'quarter'])])
            },
            'data_quality': {
                'completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'start_date': str(df.index[0]) if isinstance(df.index, pd.DatetimeIndex) else 'N/A',
                'end_date': str(df.index[-1]) if isinstance(df.index, pd.DatetimeIndex) else 'N/A',
                'days_covered': (df.index[-1] - df.index[0]).days if isinstance(df.index, pd.DatetimeIndex) else 'N/A'
            }
        }
        
        return summary