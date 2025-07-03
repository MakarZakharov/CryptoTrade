"""
Binance data fetcher implementation.
"""

import requests
import pandas as pd
from typing import Optional, Union
from datetime import datetime
import time

from ai.ML1.market_analysis.data.fetchers.base_fetcher import BaseFetcher
from ai.ML1.market_analysis.config import BINANCE_API_URL, DEFAULT_LIMIT


class BinanceFetcher(BaseFetcher):
    """
    Fetcher for Binance exchange data.
    
    Fetches historical kline (candlestick) data from the Binance API.
    """
    
    def __init__(self, symbol: str, interval: str):
        """
        Initialize the Binance fetcher.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Timeframe interval (e.g., '1d', '4h', '1h', etc.)
        """
        super().__init__(symbol, interval)
        self.api_url = BINANCE_API_URL
        self.limit = DEFAULT_LIMIT
    
    def fetch_data(self, start_date: Union[str, datetime], end_date: Optional[Union[str, datetime]] = None,
                  symbol: Optional[str] = None, interval: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data from Binance API.
        
        Args:
            start_date: Start date for data fetching
            end_date: End date for data fetching (optional, defaults to current time)
            symbol: Trading symbol (optional, overrides the one set in constructor)
            interval: Timeframe interval (optional, overrides the one set in constructor)
            
        Returns:
            DataFrame with fetched data
        """
        # Use provided symbol/interval or fall back to instance variables
        symbol = symbol or self.symbol
        interval = interval or self.interval
        
        start_ts, end_ts = self._validate_dates(start_date, end_date)
        
        # Convert to milliseconds timestamp for Binance API
        start_ms = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000) if end_ts else None
        
        all_klines = []
        
        print(f"Fetching {self.symbol} {self.interval} data from Binance...")
        
        while True:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "limit": self.limit
            }
            
            if end_ms:
                params["endTime"] = end_ms
            
            try:
                response = requests.get(self.api_url, params=params)
                data = response.json()
                
                # Check if response is valid
                if not data or isinstance(data, dict) and "code" in data:
                    if isinstance(data, dict) and "code" in data:
                        print(f"Error from Binance API: {data}")
                    break
                
                all_klines += data
                
                # Update start_ms for next request
                start_ms = data[-1][6] + 1
                
                # Break if we've reached the end or got fewer than the limit
                if len(data) < self.limit or (end_ms and start_ms >= end_ms):
                    break
                
                # Add a small delay to avoid hitting rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error fetching data from Binance: {e}")
                break
        
        if not all_klines:
            print("No data fetched from Binance")
            return pd.DataFrame()
        
        # Process the klines data into a DataFrame
        return self._process_klines(all_klines)
    
    def _process_klines(self, klines: list) -> pd.DataFrame:
        """
        Process raw klines data from Binance API into a DataFrame.
        
        Args:
            klines: List of klines data from Binance API
            
        Returns:
            Processed DataFrame
        """
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
        ])
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        
        # Convert numeric columns to float
        numeric_columns = ["open", "high", "low", "close", "volume", "quote_volume"]
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # Set timestamp as index
        df.set_index("timestamp", inplace=True)
        
        # Keep only the essential columns
        df = df[["open", "high", "low", "close", "volume"]]
        
        return df