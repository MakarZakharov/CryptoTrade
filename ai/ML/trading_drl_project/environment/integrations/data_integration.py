"""
Data Integration

Integration with the main data pipeline for environment data sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

# Relative imports to main data pipeline
try:
    from ...data_processing.core.data_loader import DataLoader
    from ...data_processing.core.indicators import IndicatorCalculator
    from ...data_processing.adapters.base_adapter import BaseDataAdapter
except ImportError:
    # Fallback imports if structure changes
    from data_processing.core.data_loader import DataLoader
    from data_processing.core.indicators import IndicatorCalculator
    from data_processing.adapters.base_adapter import BaseDataAdapter


class EnvironmentDataIntegrator:
    """
    Integrates environment with main data processing pipeline
    
    Provides seamless data flow from data processing components
    to trading environments.
    """
    
    def __init__(
        self,
        data_loader: Optional[DataLoader] = None,
        indicator_calculator: Optional[IndicatorCalculator] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize data integrator
        
        Args:
            data_loader: Data loader instance
            indicator_calculator: Technical indicator calculator
            cache_dir: Directory for caching processed data
        """
        self.data_loader = data_loader or DataLoader()
        self.indicator_calculator = indicator_calculator or IndicatorCalculator()
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Cache for processed data
        self._data_cache = {}
        
    def prepare_environment_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        include_indicators: bool = True,
        cache_key: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for trading environment
        
        Args:
            symbols: List of trading symbols
            timeframe: Data timeframe
            start_date: Start date string
            end_date: End date string
            include_indicators: Whether to include technical indicators
            cache_key: Cache key for storing/retrieving data
            
        Returns:
            Dictionary of symbol -> processed DataFrame
        """
        if cache_key and cache_key in self._data_cache:
            self.logger.info(f"Using cached data for key: {cache_key}")
            return self._data_cache[cache_key]
        
        processed_data = {}
        
        for symbol in symbols:
            self.logger.info(f"Processing data for {symbol}")
            
            # Load raw OHLCV data
            try:
                raw_data = self.data_loader.load_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if raw_data.empty:
                    self.logger.warning(f"No data found for {symbol}")
                    continue
                
                # Add technical indicators if requested
                if include_indicators:
                    enhanced_data = self.indicator_calculator.add_all_indicators(raw_data)
                else:
                    enhanced_data = raw_data.copy()
                
                # Clean and validate data
                cleaned_data = self._clean_data(enhanced_data)
                
                processed_data[symbol] = cleaned_data
                
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Cache the result
        if cache_key:
            self._data_cache[cache_key] = processed_data
            
        return processed_data
    
    def prepare_single_asset_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        window_size: int = 50,
        include_indicators: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data for single asset environment
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_date: Start date string
            end_date: End date string
            window_size: Minimum required history
            include_indicators: Whether to include technical indicators
            
        Returns:
            Processed DataFrame ready for environment
        """
        data_dict = self.prepare_environment_data(
            symbols=[symbol],
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            include_indicators=include_indicators
        )
        
        if symbol not in data_dict:
            raise ValueError(f"No data available for symbol {symbol}")
        
        data = data_dict[symbol]
        
        # Ensure minimum required history
        if len(data) < window_size:
            raise ValueError(
                f"Insufficient data: {len(data)} rows, need at least {window_size}"
            )
        
        return data
    
    def prepare_multi_asset_data(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        alignment_method: str = "inner"
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for multi-asset environment with alignment
        
        Args:
            symbols: List of trading symbols
            timeframe: Data timeframe
            start_date: Start date string
            end_date: End date string
            alignment_method: How to align timestamps ('inner', 'outer', 'forward_fill')
            
        Returns:
            Dictionary of aligned DataFrames
        """
        # Get raw data for all symbols
        raw_data_dict = self.prepare_environment_data(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            include_indicators=True
        )
        
        if not raw_data_dict:
            raise ValueError("No data available for any symbols")
        
        # Align timestamps
        aligned_data = self._align_multi_asset_data(raw_data_dict, alignment_method)
        
        return aligned_data
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        cleaned_data = data.copy()
        
        # Remove duplicates
        cleaned_data = cleaned_data[~cleaned_data.index.duplicated(keep='last')]
        
        # Sort by timestamp
        cleaned_data = cleaned_data.sort_index()
        
        # Forward fill missing values for price data
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in cleaned_data.columns:
                cleaned_data[col] = cleaned_data[col].fillna(method='ffill')
        
        # Fill volume with 0
        if 'volume' in cleaned_data.columns:
            cleaned_data['volume'] = cleaned_data['volume'].fillna(0)
        
        # Forward fill technical indicators
        indicator_cols = [col for col in cleaned_data.columns if col not in price_cols + ['volume']]
        for col in indicator_cols:
            cleaned_data[col] = cleaned_data[col].fillna(method='ffill')
        
        # Remove any remaining NaN rows
        cleaned_data = cleaned_data.dropna()
        
        return cleaned_data
    
    def _align_multi_asset_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        alignment_method: str
    ) -> Dict[str, pd.DataFrame]:
        """Align data across multiple assets"""
        if not data_dict:
            return {}
        
        # Get all unique timestamps
        all_timestamps = set()
        for df in data_dict.values():
            all_timestamps.update(df.index)
        
        all_timestamps = sorted(all_timestamps)
        
        aligned_data = {}
        
        if alignment_method == "inner":
            # Find common timestamps
            common_timestamps = set(data_dict[list(data_dict.keys())[0]].index)
            for df in data_dict.values():
                common_timestamps &= set(df.index)
            
            common_timestamps = sorted(common_timestamps)
            
            for symbol, df in data_dict.items():
                aligned_data[symbol] = df.loc[common_timestamps]
                
        elif alignment_method == "outer":
            # Use all timestamps, fill missing values
            timestamp_index = pd.DatetimeIndex(all_timestamps)
            
            for symbol, df in data_dict.items():
                # Reindex to all timestamps
                reindexed = df.reindex(timestamp_index)
                
                # Forward fill missing values
                aligned_data[symbol] = reindexed.fillna(method='ffill').fillna(method='bfill')
                
        elif alignment_method == "forward_fill":
            # Forward fill each asset independently
            for symbol, df in data_dict.items():
                aligned_data[symbol] = df.fillna(method='ffill')
        
        else:
            raise ValueError(f"Unknown alignment method: {alignment_method}")
        
        return aligned_data
    
    def get_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the data"""
        stats = {
            'total_rows': len(data),
            'date_range': {
                'start': data.index.min(),
                'end': data.index.max()
            },
            'columns': list(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'price_statistics': {}
        }
        
        # Price statistics
        if 'close' in data.columns:
            close_prices = data['close']
            stats['price_statistics'] = {
                'min_price': close_prices.min(),
                'max_price': close_prices.max(),
                'mean_price': close_prices.mean(),
                'volatility': close_prices.pct_change().std(),
                'total_return': (close_prices.iloc[-1] / close_prices.iloc[0] - 1) * 100
            }
        
        return stats
    
    def validate_environment_data(self, data: pd.DataFrame) -> Dict[str, bool]:
        """Validate data for environment use"""
        validation_results = {
            'has_required_columns': all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']),
            'no_missing_prices': not data[['open', 'high', 'low', 'close']].isnull().any().any(),
            'monotonic_index': data.index.is_monotonic_increasing,
            'sufficient_data': len(data) >= 100,
            'valid_price_relationships': True,
            'positive_volume': (data['volume'] >= 0).all() if 'volume' in data.columns else True
        }
        
        # Check price relationships (high >= low, etc.)
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            validation_results['valid_price_relationships'] = (
                (data['high'] >= data['low']).all() and
                (data['high'] >= data['open']).all() and
                (data['high'] >= data['close']).all() and
                (data['low'] <= data['open']).all() and
                (data['low'] <= data['close']).all()
            )
        
        return validation_results
    
    def clear_cache(self) -> None:
        """Clear data cache"""
        self._data_cache.clear()
        self.logger.info("Data cache cleared")