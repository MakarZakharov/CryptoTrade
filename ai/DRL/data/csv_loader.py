"""Загрузчик CSV данных для DRL системы."""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from ..utils import DRLLogger, validate_data


class CSVDataLoader:
    """Загрузчик данных из CSV файлов."""
    
    def __init__(self, data_dir: str = "CryptoTrade/data", logger: Optional[DRLLogger] = None):
        """
        Инициализация загрузчика данных.
        
        Args:
            data_dir: путь к директории с данными
            logger: логгер для записи операций
        """
        self.data_dir = Path(data_dir)
        self.logger = logger or DRLLogger("csv_loader")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Директория данных не найдена: {data_dir}")
    
    def get_available_symbols(self, exchange: str = "binance") -> List[str]:
        """Получить список доступных торговых пар для биржи."""
        exchange_path = self.data_dir / exchange
        if not exchange_path.exists():
            self.logger.warning(f"Биржа {exchange} не найдена")
            return []
        
        symbols = []
        for symbol_dir in exchange_path.iterdir():
            if symbol_dir.is_dir():
                symbols.append(symbol_dir.name)
        
        self.logger.info(f"Найдено {len(symbols)} символов для {exchange}: {symbols}")
        return sorted(symbols)
    
    def get_available_timeframes(self, exchange: str, symbol: str) -> List[str]:
        """Получить доступные таймфреймы для пары."""
        symbol_path = self.data_dir / exchange / symbol
        if not symbol_path.exists():
            self.logger.warning(f"Символ {symbol} не найден на {exchange}")
            return []
        
        timeframes = []
        for tf_dir in symbol_path.iterdir():
            if tf_dir.is_dir():
                timeframes.append(tf_dir.name)
        
        return sorted(timeframes)
    
    def load_data(
        self,
        symbol: str,
        exchange: str = "binance",
        timeframe: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Загрузить данные для указанного символа.
        
        Args:
            symbol: торговая пара (например, "BTCUSDT")
            exchange: биржа
            timeframe: таймфрейм
            start_date: начальная дата (YYYY-MM-DD)
            end_date: конечная дата (YYYY-MM-DD)
            validate: проверить данные на корректность
            
        Returns:
            DataFrame с историческими данными
        """
        # Построение пути к файлу
        file_path = self.data_dir / exchange / symbol / timeframe / "2018_01_01-now.csv"
        
        if not file_path.exists():
            # Попытка найти альтернативный файл
            timeframe_dir = self.data_dir / exchange / symbol / timeframe
            if timeframe_dir.exists():
                csv_files = list(timeframe_dir.glob("*.csv"))
                if csv_files:
                    file_path = csv_files[0]
                    self.logger.info(f"Используем файл: {file_path}")
                else:
                    raise FileNotFoundError(f"CSV файл не найден в {timeframe_dir}")
            else:
                raise FileNotFoundError(f"Путь не существует: {file_path}")
        
        self.logger.info(f"Загрузка данных из {file_path}")
        
        try:
            # Загрузка данных с оптимизацией типов данных
            dtypes = {
                'open': 'float32',
                'high': 'float32', 
                'low': 'float32',
                'close': 'float32',
                'volume': 'float32',
                'quote_volume': 'float32'
            }
            
            df = pd.read_csv(file_path, dtype=dtypes, parse_dates=['timestamp'])
            
            # Установка индекса
            df.set_index('timestamp', inplace=True)
            
            # Фильтрация по датам
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df.index >= start_dt]
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
            
            # Валидация данных
            if validate:
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if not validate_data(df, required_columns):
                    raise ValueError(f"Данные не прошли валидацию. Требуемые колонки: {required_columns}")
            
            # Сортировка по времени
            df.sort_index(inplace=True)
            
            self.logger.info(f"Загружено {len(df)} записей с {df.index[0]} по {df.index[-1]}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки данных: {e}")
            raise
    
    def get_data_info(self, symbol: str, exchange: str = "binance", timeframe: str = "1d") -> Dict:
        """Получить информацию о данных без полной загрузки."""
        try:
            # Загружаем только первые и последние несколько строк
            file_path = self.data_dir / exchange / symbol / timeframe / "2018_01_01-now.csv"
            
            if not file_path.exists():
                timeframe_dir = self.data_dir / exchange / symbol / timeframe
                if timeframe_dir.exists():
                    csv_files = list(timeframe_dir.glob("*.csv"))
                    if csv_files:
                        file_path = csv_files[0]
                    else:
                        raise FileNotFoundError(f"CSV файл не найден в {timeframe_dir}")
                else:
                    raise FileNotFoundError(f"Путь не существует: {file_path}")
            
            # Читаем только заголовок и несколько строк
            df_head = pd.read_csv(file_path, nrows=5)
            
            # Получаем размер файла
            file_size = file_path.stat().st_size
            
            # Считаем приблизительное количество строк
            with open(file_path, 'r') as f:
                first_line = f.readline()
                line_length = len(first_line)
                approx_rows = file_size // line_length - 1  # -1 для заголовка
            
            # Читаем последние строки для получения даты окончания
            df_tail = pd.read_csv(file_path, skiprows=lambda x: x != 0 and x < approx_rows - 5)
            
            if not df_tail.empty:
                end_date = df_tail.iloc[-1]['timestamp']
            else:
                end_date = "unknown"
            
            return {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'file_path': str(file_path),
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'approx_rows': approx_rows,
                'columns': df_head.columns.tolist(),
                'start_date': df_head.iloc[0]['timestamp'] if not df_head.empty else "unknown",
                'end_date': end_date,
                'data_sample': df_head.head(3).to_dict('records')
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка получения информации о данных: {e}")
            return {}
    
    def load_multiple_symbols(
        self,
        symbols: List[str],
        exchange: str = "binance",
        timeframe: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Загрузить данные для нескольких символов.
        
        Args:
            symbols: список торговых пар
            exchange: биржа
            timeframe: таймфрейм
            start_date: начальная дата
            end_date: конечная дата
            
        Returns:
            Словарь {symbol: DataFrame}
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                df = self.load_data(symbol, exchange, timeframe, start_date, end_date)
                data_dict[symbol] = df
                self.logger.info(f"Успешно загружен {symbol}: {len(df)} записей")
            except Exception as e:
                self.logger.error(f"Не удалось загрузить {symbol}: {e}")
                continue
        
        self.logger.info(f"Загружено {len(data_dict)} из {len(symbols)} символов")
        return data_dict
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """Получить отчет о качестве данных."""
        report = {
            'total_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'date_range': {
                'start': str(df.index.min()),
                'end': str(df.index.max()),
                'days': (df.index.max() - df.index.min()).days
            }
        }
        
        # Проверка на аномалии в ценах
        if 'close' in df.columns:
            price_changes = df['close'].pct_change().dropna()
            report['price_anomalies'] = {
                'extreme_moves_up': (price_changes > 0.2).sum(),  # Рост > 20%
                'extreme_moves_down': (price_changes < -0.2).sum(),  # Падение > 20%
                'zero_prices': (df['close'] <= 0).sum()
            }
        
        # Проверка на пропуски во времени
        if len(df) > 1:
            time_diffs = df.index.to_series().diff().dropna()
            expected_freq = pd.infer_freq(df.index[:100])  # Определяем частоту по первым 100 записям
            if expected_freq:
                report['time_gaps'] = {
                    'expected_frequency': expected_freq,
                    'irregular_intervals': (time_diffs != time_diffs.mode().iloc[0]).sum()
                }
        
        return report