"""
Обработка данных для ML моделей STAS_ML.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Опциональный импорт TA-Lib
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib не установлен. Технические индикаторы будут вычисляться с помощью pandas.")
    print("💡 Для установки TA-Lib:")
    print("   pip install TA-Lib")
    print("   Или на Windows: conda install -c conda-forge ta-lib")


class CryptoDataProcessor:
    """Процессор данных для криптовалютной торговли."""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
    def load_data(self) -> pd.DataFrame:
        """Загрузить данные из CSV файла."""
        try:
            data = pd.read_csv(self.config.data_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp')
            
            print(f"✅ Загружено {len(data)} записей для {self.config.symbol}")
            print(f"📅 Период: {data.index[0]} - {data.index[-1]}")
            
            return data
        except Exception as e:
            raise ValueError(f"Ошибка загрузки данных: {e}")
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Добавить технические индикаторы."""
        if not self.config.include_technical_indicators:
            return data
        
        df = data.copy()
        
        if TALIB_AVAILABLE:
            return self._add_talib_indicators(df, data)
        else:
            return self._add_pandas_indicators(df, data)
    
    def _add_talib_indicators(self, df: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Добавить технические индикаторы используя TA-Lib."""
        # Простые скользящие средние
        if 'sma' in self.config.indicator_periods:
            for period in self.config.indicator_periods['sma']:
                df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
        
        # Экспоненциальные скользящие средние
        if 'ema' in self.config.indicator_periods:
            for period in self.config.indicator_periods['ema']:
                df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)
        
        # RSI
        if 'rsi' in self.config.indicator_periods:
            for period in self.config.indicator_periods['rsi']:
                df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)
        
        # MACD
        if 'macd' in self.config.indicator_periods:
            periods = self.config.indicator_periods['macd']
            if len(periods) >= 3:
                macd, macd_signal, macd_hist = talib.MACD(
                    df['close'], 
                    fastperiod=periods[0], 
                    slowperiod=periods[1], 
                    signalperiod=periods[2]
                )
                df['macd'] = macd
                df['macd_signal'] = macd_signal
                df['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        if 'bollinger' in self.config.indicator_periods:
            for period in self.config.indicator_periods['bollinger']:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    df['close'], timeperiod=period
                )
                df[f'bb_upper_{period}'] = bb_upper
                df[f'bb_middle_{period}'] = bb_middle
                df[f'bb_lower_{period}'] = bb_lower
                df[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle
                df[f'bb_position_{period}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        if 'atr' in self.config.indicator_periods:
            for period in self.config.indicator_periods['atr']:
                df[f'atr_{period}'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        
        # Stochastic
        if 'stochastic' in self.config.indicator_periods:
            periods = self.config.indicator_periods['stochastic']
            if len(periods) >= 3:
                slowk, slowd = talib.STOCH(
                    df['high'], df['low'], df['close'],
                    fastk_period=periods[0],
                    slowk_period=periods[1],
                    slowd_period=periods[2]
                )
                df['stoch_k'] = slowk
                df['stoch_d'] = slowd
        
        # OBV
        if 'obv' in self.config.indicator_periods:
            df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Дополнительные фичи
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['open_close_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Волатильность
        df['volatility'] = df['price_change'].rolling(window=14).std()
        
        print(f"✅ Добавлено {len(df.columns) - len(original_data.columns)} технических индикаторов (TA-Lib)")
        
        return df
    
    def _add_pandas_indicators(self, df: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
        """Добавить технические индикаторы используя pandas (fallback)."""
        # Простые скользящие средние
        if 'sma' in self.config.indicator_periods:
            for period in self.config.indicator_periods['sma']:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Экспоненциальные скользящие средние
        if 'ema' in self.config.indicator_periods:
            for period in self.config.indicator_periods['ema']:
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI (упрощенная версия)
        if 'rsi' in self.config.indicator_periods:
            for period in self.config.indicator_periods['rsi']:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD (упрощенная версия)
        if 'macd' in self.config.indicator_periods:
            periods = self.config.indicator_periods['macd']
            if len(periods) >= 3:
                ema_fast = df['close'].ewm(span=periods[0]).mean()
                ema_slow = df['close'].ewm(span=periods[1]).mean()
                df['macd'] = ema_fast - ema_slow
                df['macd_signal'] = df['macd'].ewm(span=periods[2]).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        if 'bollinger' in self.config.indicator_periods:
            for period in self.config.indicator_periods['bollinger']:
                sma = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                df[f'bb_upper_{period}'] = sma + (std * 2)
                df[f'bb_middle_{period}'] = sma
                df[f'bb_lower_{period}'] = sma - (std * 2)
                df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_middle_{period}']
                df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # ATR (упрощенная версия)
        if 'atr' in self.config.indicator_periods:
            for period in self.config.indicator_periods['atr']:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df[f'atr_{period}'] = true_range.rolling(window=period).mean()
        
        # Stochastic (упрощенная версия)
        if 'stochastic' in self.config.indicator_periods:
            periods = self.config.indicator_periods['stochastic']
            if len(periods) >= 3:
                lowest_low = df['low'].rolling(window=periods[0]).min()
                highest_high = df['high'].rolling(window=periods[0]).max()
                k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
                df['stoch_k'] = k_percent.rolling(window=periods[1]).mean()
                df['stoch_d'] = df['stoch_k'].rolling(window=periods[2]).mean()
        
        # OBV (упрощенная версия)
        if 'obv' in self.config.indicator_periods:
            obv = [0]
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.append(obv[-1] + df['volume'].iloc[i])
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.append(obv[-1] - df['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            df['obv'] = obv
        
        # Дополнительные фичи
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['open_close_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Волатильность
        df['volatility'] = df['price_change'].rolling(window=14).std()
        
        print(f"✅ Добавлено {len(df.columns) - len(original_data.columns)} технических индикаторов (pandas)")
        
        return df
    
    def create_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Создать матрицу признаков с окном lookback."""
        # Убираем NaN значения
        data_clean = data.dropna()
        
        # Выбираем числовые колонки (исключаем categorical если есть)
        numeric_columns = data_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        # Создаем признаки с окном lookback
        features = []
        feature_names = []
        
        for i in range(self.config.lookback_window, len(data_clean)):
            # Берем lookback_window периодов назад
            window_data = data_clean.iloc[i-self.config.lookback_window:i][numeric_columns]
            
            # Flatten данные окна в один вектор
            window_features = window_data.values.flatten()
            features.append(window_features)
            
            # Создаем имена признаков только один раз
            if len(feature_names) == 0:
                for lag in range(self.config.lookback_window):
                    for col in numeric_columns:
                        feature_names.append(f"{col}_lag_{lag}")
        
        self.feature_names = feature_names
        return np.array(features), feature_names
    
    def create_target(self, data: pd.DataFrame) -> np.ndarray:
        """Создать ПОКРАЩЕНУ целевую переменную з фільтрацією слабких сигналів."""
        data_clean = data.dropna()
        
        if self.config.target_type == 'price_change':
            # Процентное изменение цены через target_horizon периодов
            target = data_clean['close'].pct_change(periods=self.config.target_horizon).shift(-self.config.target_horizon)
            
        elif self.config.target_type == 'direction':
            # ПОКРАЩЕНЕ направление движения з мінімальним порогом
            price_change = data_clean['close'].pct_change(periods=self.config.target_horizon).shift(-self.config.target_horizon)
            
            # Використовуємо поріг для фільтрації слабких сигналів
            min_threshold = getattr(self.config, 'min_price_change_threshold', 0.02)  # За замовчуванням 2%
            
            # Створюємо 3-класову цільову змінну: -1 (продавати), 0 (утримувати), 1 (купувати)
            target_3class = np.where(price_change > min_threshold, 1,    # Сильний ріст
                            np.where(price_change < -min_threshold, 0,   # Сильне падіння  
                                   -1))  # Слабкий сигнал - утримувати
            
            # Фільтруємо тільки сильні сигнали (видаляємо -1)
            strong_signals_mask = target_3class != -1
            
            # Зберігаємо маску як pandas Series для подальшого використання з .iloc
            self._strong_signals_mask = pd.Series(strong_signals_mask, index=price_change.index)
            
            # Повертаємо бінарну класифікацію тільки для сильних сигналів
            target = target_3class
            
        elif self.config.target_type == 'volatility':
            # Волатильність через target_horizon периодов
            returns = data_clean['close'].pct_change()
            target = returns.rolling(window=self.config.target_horizon).std().shift(-self.config.target_horizon)
        
        else:
            raise ValueError(f"Неподдерживаемый тип цели: {self.config.target_type}")
        
        # Берем только те значения, которые соответствуют нашим признакам
        if hasattr(target, 'iloc'):
            # Если target это pandas Series
            target_values = target.iloc[self.config.lookback_window:].values
        else:
            # Если target это numpy array
            target_values = target[self.config.lookback_window:]
        
        return target_values
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Подготовить данные для обучения."""
        print("🔄 Загружаем и обрабатываем данные...")
        
        # Загружаем данные
        raw_data = self.load_data()
        
        # Добавляем технические индикаторы
        data_with_indicators = self.add_technical_indicators(raw_data)
        
        # Создаем признаки и цель
        features, feature_names = self.create_features(data_with_indicators)
        target = self.create_target(data_with_indicators)
        
        # Убираем NaN значения
        valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(target))
        features = features[valid_mask]
        target = target[valid_mask]
        
        # ФІЛЬТРАЦІЯ СЛАБКИХ СИГНАЛІВ для direction цілей
        if self.config.target_type == 'direction' and hasattr(self, '_strong_signals_mask'):
            # Застосовуємо маску сильних сигналів
            # _strong_signals_mask є pandas Series, але target вже numpy array
            if hasattr(self._strong_signals_mask, 'iloc'):
                # Якщо це pandas Series
                strong_mask = self._strong_signals_mask.iloc[self.config.lookback_window:].values[valid_mask]
            else:
                # Якщо це вже numpy array
                strong_mask = self._strong_signals_mask[self.config.lookback_window:][valid_mask]
            
            strong_signals_indices = strong_mask != -1
            
            if np.sum(strong_signals_indices) > 0:
                features = features[strong_signals_indices]
                target = target[strong_signals_indices]
                # Перетворюємо -1 класи на бінарні (0, 1)
                target = np.where(target == -1, 0, target)  # Але це не повинно статися після фільтрації
                
                print(f"🎯 Відфільтровано {np.sum(~strong_signals_indices)} слабких сигналів")
        
        print(f"✅ Підготовлено {len(features)} образцов з {features.shape[1]} ознаками (високоякісні сигнали)")
        
        # Разделяем на train/validation/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, target, 
            test_size=self.config.test_split,
            random_state=self.config.random_state,
            shuffle=False  # Для временных рядов не перемешиваем
        )
        
        val_size = self.config.validation_split / (1 - self.config.test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.config.random_state,
            shuffle=False
        )
        
        # Нормализуем признаки
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        print(f"📊 Разделение данных:")
        print(f"   Train: {len(X_train)} образцов")
        print(f"   Validation: {len(X_val)} образцов")
        print(f"   Test: {len(X_test)} образцов")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_recent_features(self, data: pd.DataFrame, n_samples: int = 1) -> np.ndarray:
        """Получить признаки для последних n образцов (для предсказания)."""
        data_with_indicators = self.add_technical_indicators(data)
        features, _ = self.create_features(data_with_indicators)
        
        # Берем последние n образцов
        recent_features = features[-n_samples:]
        
        # Нормализуем используя уже обученный scaler
        recent_features = self.scaler.transform(recent_features)
        
        return recent_features