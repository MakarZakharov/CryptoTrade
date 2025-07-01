"""
Base class for data fetchers.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, Union
from datetime import datetime


class BaseFetcher(ABC):
    """
    Abstract base class for data fetchers.
    
    All data fetchers should inherit from this class and implement the fetch_data method.
    """
    
    def __init__(self, symbol: str, interval: str):
        """
        Initialize the fetcher.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Timeframe interval (e.g., '1d', '4h', '1h', etc.)
        """
        self.symbol = symbol
        self.interval = interval
    
    @abstractmethod
    def fetch_data(self, start_date: Union[str, datetime], end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Fetch data for the specified symbol and timeframe.
        
        Args:
            start_date: Start date for data fetching
            end_date: End date for data fetching (optional, defaults to current time)
            
        Returns:
            DataFrame with fetched data
        """
        pass
    
    def _validate_dates(self, start_date: Union[str, datetime], end_date: Optional[Union[str, datetime]] = None) -> tuple:
        """
        Validate and convert date inputs to datetime objects.
        
        Args:
            start_date: Start date (string or datetime)
            end_date: End date (string, datetime, or None)
            
        Returns:
            Tuple of (start_datetime, end_datetime)
        """
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        
        if end_date is None:
            end_date = pd.Timestamp.now()
        elif isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        
        if start_date > end_date:
            raise ValueError("Start date must be before end date")
        
        return start_date, end_date