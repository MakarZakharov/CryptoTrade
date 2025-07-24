"""
Advanced feature engineering for STAS_ML v2
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from ..core.config import Config
from ..core.base import Logger


class FeatureEngineer:
    """Advanced feature engineering for cryptocurrency data."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger("FeatureEngineer")
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main feature engineering pipeline."""
        df = data.copy()
        
        # Market regime features
        if self.config.features.use_market_regime:
            df = self._add_market_regime_features(df)
        
        # Momentum features
        if self.config.features.use_momentum_features:
            df = self._add_momentum_features(df)
        
        # Mean reversion features
        if self.config.features.use_mean_reversion_features:
            df = self._add_mean_reversion_features(df)
        
        # Volatility clustering features
        if self.config.features.use_volatility_features:
            df = self._add_volatility_clustering_features(df)
        
        return df
    
    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime identification features."""
        # Trend strength
        for window in [10, 20, 50]:
            sma = df['close'].rolling(window).mean()
            df[f'trend_strength_{window}'] = (df['close'] - sma) / sma
        
        # Market state
        df['bull_market'] = (df['close'] > df['close'].rolling(50).mean()).astype(int)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        # Price momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # ROC (Rate of Change)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        return df
    
    def _add_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mean reversion features."""
        # Z-score
        for window in [10, 20, 50]:
            mean = df['close'].rolling(window).mean()
            std = df['close'].rolling(window).std()
            df[f'zscore_{window}'] = (df['close'] - mean) / std
        
        return df
    
    def _add_volatility_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility clustering features."""
        # Rolling volatility
        returns = df['close'].pct_change()
        for window in [5, 10, 20]:
            df[f'vol_cluster_{window}'] = returns.rolling(window).std()
        
        return df