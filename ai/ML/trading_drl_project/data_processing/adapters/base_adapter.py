"""
Base Data Adapter

Abstract base class for exchange data adapters.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging


class BaseDataAdapter(ABC):
    """
    Abstract base class for exchange data adapters
    
    Provides common interface for fetching data from different exchanges.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = True,
        rate_limit: int = 100,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base adapter
        
        Args:
            api_key: Exchange API key
            api_secret: Exchange API secret
            testnet: Whether to use testnet
            rate_limit: Rate limit for API calls
            config: Additional configuration
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.rate_limit = rate_limit
        self.config = config or {}
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize client
        self.client = None
        self._initialize_client()
        
    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialize exchange client"""
        pass
    
    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Get available trading symbols"""
        pass
    
    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV data
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '1d')
            since: Start datetime
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book data
        
        Args:
            symbol: Trading pair symbol
            limit: Number of orders to fetch
            
        Returns:
            Order book data
        """
        pass
    
    @abstractmethod
    def get_trades(
        self,
        symbol: str,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get recent trades
        
        Args:
            symbol: Trading pair symbol
            since: Start datetime
            limit: Number of trades to fetch
            
        Returns:
            DataFrame with trade data
        """
        pass
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24h ticker data
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Ticker data
        """
        raise NotImplementedError("Ticker data not implemented for this adapter")
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol is available on exchange
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            True if symbol is valid
        """
        try:
            symbols = self.get_symbols()
            return symbol in symbols
        except Exception as e:
            self.logger.error(f"Error validating symbol {symbol}: {e}")
            return False
    
    def standardize_timeframe(self, timeframe: str) -> str:
        """
        Standardize timeframe format
        
        Args:
            timeframe: Original timeframe
            
        Returns:
            Standardized timeframe
        """
        # Common timeframe mappings
        timeframe_map = {
            '1m': '1m', '1min': '1m',
            '5m': '5m', '5min': '5m',
            '15m': '15m', '15min': '15m',
            '30m': '30m', '30min': '30m',
            '1h': '1h', '1hour': '1h',
            '4h': '4h', '4hour': '4h',
            '1d': '1d', '1day': '1d',
            '1w': '1w', '1week': '1w'
        }
        
        return timeframe_map.get(timeframe.lower(), timeframe)
    
    def standardize_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize OHLCV DataFrame format
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Standardized DataFrame
        """
        # Ensure standard column names
        standard_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if len(data.columns) >= 5:
            data.columns = standard_columns[:len(data.columns)]
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data.index = pd.to_datetime(data['timestamp'])
                data.drop('timestamp', axis=1, inplace=True)
            elif len(data.columns) == 6:  # timestamp as first column
                data.index = pd.to_datetime(data.iloc[:, 0])
                data = data.iloc[:, 1:]
                data.columns = standard_columns
        
        # Convert to numeric
        for col in standard_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove duplicates and sort
        data = data[~data.index.duplicated(keep='last')]
        data = data.sort_index()
        
        # Forward fill missing values
        data = data.fillna(method='ffill')
        
        return data
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        chunk_size: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical data in chunks
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date (default: now)
            chunk_size: Size of each chunk
            
        Returns:
            Complete historical data
        """
        if end_date is None:
            end_date = datetime.now()
        
        all_data = []
        current_date = start_date
        
        while current_date < end_date:
            try:
                chunk_data = self.get_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_date,
                    limit=chunk_size
                )
                
                if chunk_data.empty:
                    break
                
                all_data.append(chunk_data)
                
                # Update current_date to last timestamp + 1 interval
                last_timestamp = chunk_data.index[-1]
                if timeframe.endswith('m'):
                    minutes = int(timeframe[:-1])
                    current_date = last_timestamp + timedelta(minutes=minutes)
                elif timeframe.endswith('h'):
                    hours = int(timeframe[:-1])
                    current_date = last_timestamp + timedelta(hours=hours)
                elif timeframe.endswith('d'):
                    days = int(timeframe[:-1])
                    current_date = last_timestamp + timedelta(days=days)
                else:
                    break
                
                # Rate limiting
                import time
                time.sleep(60 / self.rate_limit)  # Respect rate limit
                
            except Exception as e:
                self.logger.error(f"Error fetching chunk from {current_date}: {e}")
                break
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all chunks
        combined_data = pd.concat(all_data)
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data = combined_data.sort_index()
        
        # Filter by date range
        combined_data = combined_data[
            (combined_data.index >= start_date) & 
            (combined_data.index <= end_date)
        ]
        
        return combined_data
    
    def close(self) -> None:
        """Close adapter and cleanup resources"""
        if hasattr(self.client, 'close'):
            self.client.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()