"""
Модуль загрузки данных для торгового окружения DRL
Поддерживает CSV и Parquet форматы с валидацией и preprocessing
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import warnings


class DataLoader:
    """
    Загрузчик данных для торгового окружения.
    Поддерживает CSV/Parquet, валидацию, нормализацию и технические индикаторы.
    """

    REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    OPTIONAL_COLUMNS = ['orderbook_bids', 'orderbook_asks', 'trade_ticks']

    def __init__(
        self,
        data_path: Optional[str] = None,
        symbol: str = "BTCUSDT",
        timeframe: str = "1d",
        normalize: bool = True,
        add_indicators: bool = True,
        validate: bool = True
    ):
        """
        Инициализация загрузчика данных.

        Args:
            data_path: Путь к файлу данных (CSV/Parquet)
            symbol: Символ торговой пары (BTCUSDT, ETHUSDT и т.д.)
            timeframe: Таймфрейм (15m, 1h, 4h, 1d)
            normalize: Нормализовать ли данные
            add_indicators: Добавить ли технические индикаторы
            validate: Валидировать ли данные
        """
        self.data_path = data_path
        self.symbol = symbol
        self.timeframe = timeframe
        self.normalize = normalize
        self.add_indicators = add_indicators
        self.validate = validate

        self.data: Optional[pd.DataFrame] = None
        self.raw_data: Optional[pd.DataFrame] = None
        self.normalization_params: Dict[str, Any] = {}

    def load(self, start_index: int = 0, end_index: Optional[int] = None) -> pd.DataFrame:
        """
        Загрузить данные из файла или автоматически найти файл.

        Args:
            start_index: Начальный индекс для среза данных
            end_index: Конечный индекс для среза данных

        Returns:
            DataFrame с загруженными данными
        """
        # Если путь не указан, пытаемся найти автоматически
        if self.data_path is None:
            self.data_path = self._find_data_file()

        # Загружаем данные
        if self.data_path.endswith('.parquet'):
            self.raw_data = pd.read_parquet(self.data_path)
        elif self.data_path.endswith('.csv'):
            self.raw_data = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")

        # Валидация
        if self.validate:
            self._validate_data()

        # Преобразование timestamp
        if 'timestamp' in self.raw_data.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.raw_data['timestamp']):
                self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])

        # Копируем для обработки
        self.data = self.raw_data.copy()

        # Добавляем технические индикаторы
        if self.add_indicators:
            self.data = self._add_technical_indicators(self.data)

        # Нормализация
        if self.normalize:
            self.data, self.normalization_params = self._normalize_data(self.data)

        # Срез данных
        if end_index is None:
            end_index = len(self.data)

        self.data = self.data.iloc[start_index:end_index].reset_index(drop=True)

        return self.data

    def _find_data_file(self) -> str:
        """
        Автоматически найти файл данных на основе symbol и timeframe.
        """
        # Путь к данным в проекте
        base_path = Path(__file__).parent.parent / "EnviromentData" / "Date" / "binance"
        data_path = base_path / self.symbol / self.timeframe / f"{self.symbol}_{self.timeframe}.parquet"

        if data_path.exists():
            return str(data_path)

        # Пробуем CSV
        data_path_csv = data_path.with_suffix('.csv')
        if data_path_csv.exists():
            return str(data_path_csv)

        raise FileNotFoundError(
            f"Data file not found for {self.symbol} {self.timeframe}\n"
            f"Expected path: {data_path}"
        )

    def _validate_data(self):
        """Валидация загруженных данных."""
        if self.raw_data is None or len(self.raw_data) == 0:
            raise ValueError("Data is empty!")

        # Проверяем обязательные колонки
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.raw_data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Проверяем на NaN
        if self.raw_data[self.REQUIRED_COLUMNS].isnull().any().any():
            warnings.warn("Data contains NaN values, filling with forward fill...")
            self.raw_data[self.REQUIRED_COLUMNS] = self.raw_data[self.REQUIRED_COLUMNS].fillna(method='ffill')

        # Проверяем OHLC логику
        invalid_rows = (
            (self.raw_data['high'] < self.raw_data['low']) |
            (self.raw_data['high'] < self.raw_data['open']) |
            (self.raw_data['high'] < self.raw_data['close']) |
            (self.raw_data['low'] > self.raw_data['open']) |
            (self.raw_data['low'] > self.raw_data['close'])
        )

        if invalid_rows.any():
            warnings.warn(f"Found {invalid_rows.sum()} rows with invalid OHLC data!")

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавить технические индикаторы.

        Args:
            df: DataFrame с OHLCV данными

        Returns:
            DataFrame с добавленными индикаторами
        """
        df = df.copy()

        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving Averages
        for period in [7, 14, 21, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'])

        # ATR (Average True Range)
        df['atr_14'] = self._calculate_atr(df, 14)

        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']

        # Price momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100

        # Заполняем NaN нулями для начальных значений
        df = df.fillna(0)

        return df

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Расчет RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calculate_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Расчет MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _calculate_bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Расчет Bollinger Bands."""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Расчет ATR (Average True Range)."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    def _normalize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Нормализация данных для ML.

        Args:
            df: DataFrame с данными

        Returns:
            Tuple[нормализованный DataFrame, параметры нормализации]
        """
        df_normalized = df.copy()
        params = {}

        # Колонки для нормализации (price-based)
        price_cols = ['open', 'high', 'low', 'close']

        # Нормализация цен относительно первой цены закрытия
        if 'close' in df.columns:
            base_price = df['close'].iloc[0]
            params['base_price'] = base_price

            for col in price_cols:
                if col in df_normalized.columns:
                    df_normalized[col] = df_normalized[col] / base_price

        # Нормализация объема (log scale)
        if 'volume' in df_normalized.columns:
            df_normalized['volume_normalized'] = np.log1p(df_normalized['volume'])
            params['volume_log'] = True

        # Нормализация индикаторов
        indicator_cols = [col for col in df_normalized.columns if any(
            x in col for x in ['sma_', 'ema_', 'bb_', 'atr_', 'momentum_', 'macd']
        )]

        for col in indicator_cols:
            if col in df_normalized.columns:
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val > 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
                    params[f'{col}_mean'] = mean_val
                    params[f'{col}_std'] = std_val

        return df_normalized, params

    def get_window(self, start_idx: int, window_size: int) -> np.ndarray:
        """
        Получить окно данных для observation.

        Args:
            start_idx: Начальный индекс
            window_size: Размер окна

        Returns:
            Numpy array с данными окна
        """
        if self.data is None:
            raise ValueError("Data not loaded! Call load() first.")

        if start_idx < window_size:
            # Дополняем нулями если окно выходит за начало
            padding = np.zeros((window_size - start_idx, len(self.data.columns)))
            window_data = self.data.iloc[0:start_idx].values
            return np.vstack([padding, window_data])
        else:
            return self.data.iloc[start_idx - window_size:start_idx].values

    def get_price_at(self, idx: int, price_type: str = 'close') -> float:
        """
        Получить цену на определенном индексе.

        Args:
            idx: Индекс
            price_type: Тип цены (open, high, low, close)

        Returns:
            Цена
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded!")

        return self.raw_data.iloc[idx][price_type]

    def get_timestamp_at(self, idx: int) -> pd.Timestamp:
        """Получить timestamp на определенном индексе."""
        if self.raw_data is None:
            raise ValueError("Data not loaded!")

        return self.raw_data.iloc[idx]['timestamp']

    def __len__(self) -> int:
        """Длина загруженных данных."""
        if self.data is None:
            return 0
        return len(self.data)

    def get_feature_names(self) -> List[str]:
        """Получить список названий фич."""
        if self.data is None:
            return []
        return list(self.data.columns)

    def split_train_test(
        self,
        train_ratio: float = 0.8
    ) -> Tuple['DataLoader', 'DataLoader']:
        """
        Разделить данные на train и test.

        Args:
            train_ratio: Доля данных для обучения

        Returns:
            Tuple[train_loader, test_loader]
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded!")

        split_idx = int(len(self.raw_data) * train_ratio)

        train_loader = DataLoader(
            data_path=self.data_path,
            symbol=self.symbol,
            timeframe=self.timeframe,
            normalize=self.normalize,
            add_indicators=self.add_indicators,
            validate=False  # Уже провалидировано
        )
        train_loader.load(start_index=0, end_index=split_idx)

        test_loader = DataLoader(
            data_path=self.data_path,
            symbol=self.symbol,
            timeframe=self.timeframe,
            normalize=self.normalize,
            add_indicators=self.add_indicators,
            validate=False
        )
        test_loader.load(start_index=split_idx)

        return train_loader, test_loader


if __name__ == "__main__":
    # Пример использования
    print("=== DataLoader Test ===")

    # Загрузка данных
    loader = DataLoader(
        symbol="BTCUSDT",
        timeframe="1d",
        normalize=True,
        add_indicators=True
    )

    data = loader.load()
    print(f"\nLoaded {len(data)} rows")
    print(f"Features: {len(loader.get_feature_names())}")
    print(f"\nFirst 5 feature names: {loader.get_feature_names()[:5]}")
    print(f"\nData shape: {data.shape}")
    print(f"\nData info:")
    print(data.info())

    # Тест окна данных
    window = loader.get_window(100, 50)
    print(f"\nWindow shape: {window.shape}")

    # Тест разделения на train/test
    train_loader, test_loader = loader.split_train_test(train_ratio=0.8)
    print(f"\nTrain size: {len(train_loader)}")
    print(f"Test size: {len(test_loader)}")
