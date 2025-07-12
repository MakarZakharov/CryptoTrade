"""
Сборщик данных для DRL системы торговли криптовалютой.
Поддерживает множественные источники данных и API.
"""

import os
import sys
import pandas as pd
import numpy as np
import ccxt
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import time
from dataclasses import dataclass
import logging


@dataclass
class DataConfig:
    """Конфигурация для сбора данных."""
    symbol: str = 'BTC/USDT'
    timeframe: str = '1h'  # 1m, 5m, 15m, 1h, 4h, 1d
    start_date: str = '2020-01-01'
    end_date: Optional[str] = None
    exchange: str = 'binance'
    save_path: str = 'data'
    

class CryptoDataCollector:
    """Главный класс для сбора криптовалютных данных."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.exchanges = self._init_exchanges()
        
    def _setup_logger(self) -> logging.Logger:
        """Настройка логирования."""
        logger = logging.getLogger('DataCollector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _init_exchanges(self) -> Dict:
        """Инициализация подключений к биржам."""
        exchanges = {}
        
        try:
            # Binance
            exchanges['binance'] = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY', ''),
                'secret': os.getenv('BINANCE_SECRET', ''),
                'sandbox': True,  # Тестовый режим
                'rateLimit': 1200,
            })
            
            # Coinbase
            exchanges['coinbase'] = ccxt.coinbase({
                'apiKey': os.getenv('COINBASE_API_KEY', ''),
                'secret': os.getenv('COINBASE_SECRET', ''),
                'rateLimit': 1000,
            })
            
            self.logger.info("Биржи инициализированы успешно")
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации бирж: {e}")
            
        return exchanges
    
    def collect_ohlcv_data(
        self,
        symbol: str = None,
        timeframe: str = None,
        start_date: str = None,
        end_date: str = None,
        exchange: str = None
    ) -> pd.DataFrame:
        """
        Сбор OHLCV данных с биржи.
        
        Args:
            symbol: Торговая пара (например, 'BTC/USDT')
            timeframe: Таймфрейм ('1m', '5m', '15m', '1h', '4h', '1d')
            start_date: Начальная дата (YYYY-MM-DD)
            end_date: Конечная дата (YYYY-MM-DD)
            exchange: Название биржи
            
        Returns:
            DataFrame с OHLCV данными
        """
        # Используем параметры из конфига если не переданы
        symbol = symbol or self.config.symbol
        timeframe = timeframe or self.config.timeframe
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date or datetime.now().strftime('%Y-%m-%d')
        exchange = exchange or self.config.exchange
        
        self.logger.info(f"Сбор данных: {symbol} {timeframe} с {start_date} по {end_date}")
        
        if exchange not in self.exchanges:
            raise ValueError(f"Биржа {exchange} не поддерживается")
            
        exchange_obj = self.exchanges[exchange]
        
        try:
            # Конвертация дат в миллисекунды
            since = int(pd.Timestamp(start_date).timestamp() * 1000)
            until = int(pd.Timestamp(end_date).timestamp() * 1000)
            
            all_data = []
            current_since = since
            
            while current_since < until:
                try:
                    # Получаем данные порциями
                    ohlcv = exchange_obj.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        since=current_since,
                        limit=1000
                    )
                    
                    if not ohlcv:
                        break
                        
                    all_data.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 1
                    
                    # Пауза для соблюдения лимитов API
                    time.sleep(exchange_obj.rateLimit / 1000)
                    
                except Exception as e:
                    self.logger.error(f"Ошибка получения данных: {e}")
                    time.sleep(5)
                    continue
            
            # Создание DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Удаление дубликатов
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            self.logger.info(f"Собрано {len(df)} записей данных")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка сбора данных: {e}")
            return pd.DataFrame()
    
    def collect_coinmarketcap_data(self, symbol: str) -> Dict:
        """Сбор дополнительных данных с CoinMarketCap."""
        api_key = os.getenv('CMC_API_KEY')
        if not api_key:
            self.logger.warning("API ключ CoinMarketCap не найден")
            return {}
            
        url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': api_key,
        }
        
        parameters = {
            'symbol': symbol.split('/')[0],  # Берем базовую валюту
            'convert': 'USD'
        }
        
        try:
            response = requests.get(url, headers=headers, params=parameters)
            data = response.json()
            
            if response.status_code == 200:
                return data['data']
            else:
                self.logger.error(f"Ошибка CoinMarketCap API: {data}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Ошибка получения данных CoinMarketCap: {e}")
            return {}
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Сохранение данных в файл."""
        if filename is None:
            symbol_clean = self.config.symbol.replace('/', '_')
            filename = f"{symbol_clean}_{self.config.timeframe}_{self.config.start_date}_to_{self.config.end_date or 'now'}.csv"
        
        # Создание директории если не существует
        os.makedirs(self.config.save_path, exist_ok=True)
        
        filepath = os.path.join(self.config.save_path, filename)
        df.to_csv(filepath)
        
        self.logger.info(f"Данные сохранены: {filepath}")
        return filepath
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Загрузка данных из файла."""
        try:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            self.logger.info(f"Данные загружены: {filepath}")
            return df
        except Exception as e:
            self.logger.error(f"Ошибка загрузки данных: {e}")
            return pd.DataFrame()


def main():
    """Пример использования сборщика данных."""
    # Конфигурация
    config = DataConfig(
        symbol='BTC/USDT',
        timeframe='1h',
        start_date='2023-01-01',
        exchange='binance'
    )
    
    # Создание сборщика
    collector = CryptoDataCollector(config)
    
    # Сбор данных
    df = collector.collect_ohlcv_data()
    
    if not df.empty:
        # Сохранение данных
        filepath = collector.save_data(df)
        print(f"Данные успешно собраны и сохранены: {filepath}")
        print(f"Записей: {len(df)}")
        print(f"Период: {df.index.min()} - {df.index.max()}")
    else:
        print("Не удалось собрать данные")


if __name__ == "__main__":
    main()