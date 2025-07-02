"""
Price data processor implementation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Any
from sklearn.preprocessing import MinMaxScaler

from market_analysis.data.processors.base_processor import BaseProcessor
from market_analysis.config import DEFAULT_TRAIN_SPLIT, DEFAULT_VAL_SPLIT


class PriceProcessor(BaseProcessor):
    """
    Processor for price data.
    
    Prepares price data for model training and evaluation.
    """
    
    def __init__(self, window_size: int = 60, feature_columns: list = None, target_column: str = 'close'):
        """
        Initialize the price processor.
        
        Args:
            window_size: Size of the window for sequence data
            feature_columns: List of columns to use as features (default: ['close'])
            target_column: Column to use as target (default: 'close')
        """
        super().__init__(window_size)
        self.feature_columns = feature_columns or ['close']
        self.target_column = target_column
        self.scaler = MinMaxScaler()
    
    def process(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Process the price data for model training and evaluation.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
        """
        # Validate the data
        data = self._validate_data(data)
        
        # Extract features and target
        features = data[self.feature_columns].values
        
        # Scale the features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = self._create_sequences(scaled_features)
        
        # Split the data
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(
            X, y, train_split=DEFAULT_TRAIN_SPLIT, val_split=DEFAULT_VAL_SPLIT
        )
        
        # Reshape for LSTM input (samples, time steps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(self.feature_columns)))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], len(self.feature_columns)))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], len(self.feature_columns)))
        
        return X_train, X_val, X_test, y_train, y_val, y_test, self.scaler
    
    def _create_sequences(self, scaled_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            scaled_data: Scaled feature data
            
        Returns:
            Tuple of (X, y) where X is the input sequences and y is the target values
        """
        X, y = [], []
        
        for i in range(self.window_size, len(scaled_data)):
            # For each feature, take the window_size previous values
            features_sequence = []
            for j in range(len(self.feature_columns)):
                features_sequence.append(scaled_data[i-self.window_size:i, j])
            
            X.append(np.column_stack(features_sequence))
            
            # Target is the current value of the target column
            target_idx = self.feature_columns.index(self.target_column) if self.target_column in self.feature_columns else 0
            y.append(scaled_data[i, target_idx])
        
        return np.array(X), np.array(y)
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = data.copy()
        
        # Add SMA indicators
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Add EMA indicators
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Add RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Add MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Add Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Add volatility
        df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean() * 100
        
        # Add price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Fill NaN values
        df = df.fillna(method='bfill')
        
        return df