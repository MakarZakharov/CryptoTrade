"""
CSV data fetcher implementation.
"""

import pandas as pd
from typing import Optional, Union
from datetime import datetime
import os

from ai.ML1.market_analysis.data.fetchers.base_fetcher import BaseFetcher


class CSVFetcher(BaseFetcher):
    """
    Fetcher for data from local CSV files.
    
    Loads historical price data from CSV files with a specific format.
    """
    
    def __init__(self, symbol: str, interval: str, base_path: str = None):
        """
        Initialize the CSV fetcher.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Timeframe interval (e.g., '1d', '4h', '1h', etc.)
            base_path: Base directory path where CSV files are stored
        """
        super().__init__(symbol, interval)
        self.base_path = base_path or os.path.join('data')
    
    def fetch_data(self, start_date: Union[str, datetime], end_date: Optional[Union[str, datetime]] = None,
                  symbol: Optional[str] = None, interval: Optional[str] = None, path: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data from a CSV file.
        
        Args:
            start_date: Start date for filtering data
            end_date: End date for filtering data (optional, defaults to current time)
            symbol: Trading symbol (optional, overrides the one set in constructor)
            interval: Timeframe interval (optional, overrides the one set in constructor)
            path: Direct path to CSV file (optional, overrides symbol and interval)
            
        Returns:
            DataFrame with fetched data
        """
        # Use provided symbol/interval or fall back to instance variables
        symbol = symbol or self.symbol
        interval = interval or self.interval
        
        start_ts, end_ts = self._validate_dates(start_date, end_date)
        
        # Use provided path or construct based on symbol and interval
        file_path = path if path else self._get_file_path(symbol, interval)
        
        if not os.path.exists(file_path):
            print(f"CSV file not found: {file_path}")
            return pd.DataFrame()
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Check if the timestamp column exists
            if 'timestamp' not in df.columns:
                print(f"CSV file does not have a 'timestamp' column: {file_path}")
                return pd.DataFrame()
            
            # Convert timestamp to datetime
            if df['timestamp'].dtype == object:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Filter by date range
            df = df[(df.index >= start_ts) & (df.index <= end_ts)]
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"CSV file is missing required columns: {missing_columns}")
                # Try to adapt to available columns
                if 'close' not in missing_columns:
                    for col in missing_columns:
                        if col != 'volume':
                            df[col] = df['close']
                        else:
                            df[col] = 0
                else:
                    return pd.DataFrame()
            
            # Keep only the essential columns
            df = df[required_columns]
            
            # Convert numeric columns to float
            df = df.astype(float)
            
            return df
            
        except Exception as e:
            print(f"Error loading CSV file {file_path}: {e}")
            return pd.DataFrame()
    
    def _get_file_path(self, symbol: Optional[str] = None, interval: Optional[str] = None) -> str:
        """
        Construct the file path based on symbol and interval.
        
        Args:
            symbol: Trading symbol (optional, overrides the one set in constructor)
            interval: Timeframe interval (optional, overrides the one set in constructor)
            
        Returns:
            Path to the CSV file
        """
        # Use provided symbol/interval or fall back to instance variables
        symbol = symbol or self.symbol
        interval = interval or self.interval
        
        # Handle different directory structures
        # Try different possible paths including the actual structure used
        possible_paths = [
            os.path.join(self.base_path, symbol, interval, "2018_01_01-now.csv"),
            os.path.join(self.base_path, f"{symbol}_{interval}.csv"),
            os.path.join(self.base_path, symbol, f"{interval}.csv"),
            os.path.join(self.base_path, interval, f"{symbol}.csv")
        ]
        
        # Check if any of the possible paths exist
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Default to the first path if none exist
        return possible_paths[0]