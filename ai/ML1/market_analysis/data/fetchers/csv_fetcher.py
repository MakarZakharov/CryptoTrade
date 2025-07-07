"""
CSV Data Fetcher for market data
"""

import pandas as pd
import os
from typing import Optional
from datetime import datetime


class CSVFetcher:
    """Fetcher for loading market data from CSV files."""
    
    def __init__(self, symbol: str, interval: str, base_path: str = "data/binance"):
        """
        Initialize CSV fetcher.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Timeframe interval (e.g., '1d', '4h', '1h')
            base_path: Base directory path where CSV files are stored
        """
        self.symbol = symbol
        self.interval = interval
        self.base_path = base_path
        
        # Construct the expected file path
        self.file_path = os.path.join(base_path, symbol, interval, "2018_01_01-now.csv")
        
    def fetch_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch market data from CSV file.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Check if file exists
            if not os.path.exists(self.file_path):
                print(f"CSV file not found: {self.file_path}")
                return pd.DataFrame()
            
            # Load CSV data
            df = pd.read_csv(self.file_path)
            
            # Standardize column names (assuming common formats)
            df = self._standardize_columns(df)
            
            # Convert timestamp to datetime index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif len(df.columns) >= 6:  # Assume first column is timestamp if not named
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                df.set_index(df.columns[0], inplace=True)
            
            # Filter by date range if specified
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # Ensure numeric data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with NaN values in essential columns
            df.dropna(subset=['close'], inplace=True)
            
            print(f"Loaded {len(df)} records for {self.symbol} {self.interval}")
            return df
            
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to match expected format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        # Common column name mappings
        column_mapping = {
            # Standard formats
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            
            # Binance format
            'open_time': 'timestamp',
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close',
            'volume': 'volume',
            
            # Alternative formats
            'Date': 'timestamp',
            'Time': 'timestamp',
            'Timestamp': 'timestamp',
            'datetime': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            
            # Yahoo Finance format
            'Adj Close': 'close'
        }
        
        # Apply column mapping
        df.rename(columns=column_mapping, inplace=True)
        
        # If columns are not named, assume standard OHLCV order
        if len(df.columns) >= 6 and df.columns[0] == 0:  # Numeric column names
            expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df.columns = expected_columns[:len(df.columns)]
        
        return df
    
    def get_file_info(self) -> dict:
        """
        Get information about the CSV file.
        
        Returns:
            Dictionary with file information
        """
        info = {
            'symbol': self.symbol,
            'interval': self.interval,
            'file_path': self.file_path,
            'exists': os.path.exists(self.file_path),
            'size_mb': 0,
            'rows': 0
        }
        
        if info['exists']:
            # Get file size
            info['size_mb'] = round(os.path.getsize(self.file_path) / (1024 * 1024), 2)
            
            # Get number of rows (quick estimate)
            try:
                with open(self.file_path, 'r') as f:
                    info['rows'] = sum(1 for line in f) - 1  # Subtract header
            except:
                info['rows'] = 0
        
        return info
    
    @staticmethod
    def list_available_data(base_path: str = "data/binance") -> dict:
        """
        List all available data files.
        
        Args:
            base_path: Base directory to scan
            
        Returns:
            Dictionary with available symbols and intervals
        """
        available_data = {}
        
        if not os.path.exists(base_path):
            return available_data
        
        try:
            # Scan directory structure
            for symbol in os.listdir(base_path):
                symbol_path = os.path.join(base_path, symbol)
                if os.path.isdir(symbol_path):
                    available_data[symbol] = []
                    
                    for interval in os.listdir(symbol_path):
                        interval_path = os.path.join(symbol_path, interval)
                        if os.path.isdir(interval_path):
                            # Check if CSV file exists
                            csv_file = os.path.join(interval_path, "2018_01_01-now.csv")
                            if os.path.exists(csv_file):
                                available_data[symbol].append(interval)
        
        except Exception as e:
            print(f"Error scanning data directory: {e}")
        
        return available_data