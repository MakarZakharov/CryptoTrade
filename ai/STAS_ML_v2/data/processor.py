"""
Advanced data processor for STAS_ML v2

Handles data loading, cleaning, preprocessing, and splitting with robust error handling.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ..core.config import Config
from ..core.base import BasePreprocessor, Logger
from .features import FeatureEngineer
from .indicators import TechnicalIndicators


class DataProcessor(BasePreprocessor):
    """Advanced data processor for cryptocurrency data."""
    
    def __init__(self, config: Config):
        super().__init__(config)
        self.logger = Logger("DataProcessor")
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(config)
        self.technical_indicators = TechnicalIndicators(config)
        
        # Scalers
        self.scaler = None
        self._init_scaler()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.feature_data = None
        self.target_data = None
        
    def _init_scaler(self):
        """Initialize data scaler."""
        scaler_type = getattr(self.config.features, 'scaler_type', 'robust')
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()  # Default
    
    def load_data(self) -> pd.DataFrame:
        """Load data from file with robust error handling."""
        try:
            data_path = Path(self.config.data.data_path)
            
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            # Load data
            data = pd.read_csv(data_path)
            
            # Validate required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Process timestamp
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp')
            data = data.sort_index()
            
            # Filter by date range
            if self.config.data.start_date:
                data = data[data.index >= self.config.data.start_date]
            if self.config.data.end_date:
                data = data[data.index <= self.config.data.end_date]
            
            # Basic validation
            if len(data) < self.config.data.min_data_points:
                raise ValueError(f"Insufficient data: {len(data)} < {self.config.data.min_data_points}")
            
            self.raw_data = data
            
            self.logger.info(f"Loaded {len(data)} data points for {self.config.data.symbol}")
            self.logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data."""
        data_clean = data.copy()
        original_length = len(data_clean)
        
        # Remove invalid price data
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data_clean.columns:
                # Remove negative or zero prices
                data_clean = data_clean[data_clean[col] > 0]
                
                # Remove extreme outliers if configured
                if self.config.data.outlier_removal:
                    z_scores = np.abs((data_clean[col] - data_clean[col].mean()) / data_clean[col].std())
                    data_clean = data_clean[z_scores < self.config.data.outlier_threshold]
        
        # Validate high >= low, high >= open, high >= close
        if all(col in data_clean.columns for col in ['high', 'low', 'open', 'close']):
            valid_mask = (
                (data_clean['high'] >= data_clean['low']) &
                (data_clean['high'] >= data_clean['open']) &
                (data_clean['high'] >= data_clean['close']) &
                (data_clean['low'] <= data_clean['open']) &
                (data_clean['low'] <= data_clean['close'])
            )
            data_clean = data_clean[valid_mask]
        
        # Remove zero volume if present
        if 'volume' in data_clean.columns:
            data_clean = data_clean[data_clean['volume'] > 0]
        
        # Handle missing values
        if self.config.data.fill_missing:
            # Forward fill first, then backward fill
            data_clean = data_clean.fillna(method='ffill').fillna(method='bfill')
        else:
            # Drop rows with missing values
            data_clean = data_clean.dropna()
        
        removed_count = original_length - len(data_clean)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} invalid data points during cleaning")
        
        return data_clean
    
    def add_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic price and volume features."""
        df = data.copy()
        
        # Price changes and returns
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_change'] = df['volume'].pct_change()
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price ratios
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['open_close_ratio'] = (df['close'] - df['open']) / df['open']
        df['high_close_ratio'] = (df['high'] - df['close']) / df['close']
        df['low_close_ratio'] = (df['close'] - df['low']) / df['close']
        
        # Volatility measures
        df['volatility_short'] = df['price_change'].rolling(window=5).std()
        df['volatility_medium'] = df['price_change'].rolling(window=20).std()
        df['volatility_long'] = df['price_change'].rolling(window=50).std()
        
        # Range measures
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        
        return df
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main data processing pipeline."""
        self.logger.info("Processing data...")
        
        # Clean data
        data_clean = self.clean_data(data)
        
        # Add basic features
        data_with_basic = self.add_basic_features(data_clean)
        
        # Add technical indicators
        if self.config.features.use_technical_indicators:
            data_with_indicators = self.technical_indicators.add_all_indicators(data_with_basic)
        else:
            data_with_indicators = data_with_basic
        
        # Advanced feature engineering
        data_with_features = self.feature_engineer.engineer_features(data_with_indicators)
        
        # Store processed data
        self.processed_data = data_with_features
        
        self.logger.info(f"Data processing complete: {len(data_with_features)} rows, {len(data_with_features.columns)} features")
        
        return data_with_features
    
    def create_features_and_targets(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create feature matrix and target vector."""
        self.logger.info("Creating features and targets...")
        
        # Create target variable
        target = self._create_target(data)
        
        # Create feature matrix with lookback window
        features = self._create_feature_matrix(data)
        
        # Align features and targets
        min_len = min(len(features), len(target))
        features = features[:min_len]
        target = target[:min_len]
        
        # Remove NaN values
        valid_mask = ~(np.isnan(features).any(axis=1) | pd.isna(target))
        features = features[valid_mask]
        target = target[valid_mask]
        
        self.logger.info(f"Created {len(features)} samples with {features.shape[1]} features")
        
        # Store for reference
        self.feature_data = features
        self.target_data = target
        
        return features, target
    
    def _create_target(self, data: pd.DataFrame) -> np.ndarray:
        """Create target variable based on configuration."""
        target_type = self.config.model.target_type.value
        horizon = self.config.features.prediction_horizon
        
        if target_type == 'direction':
            # Classification: price direction
            price_change = data['close'].pct_change(periods=horizon).shift(-horizon)
            
            # Use threshold for signal strength
            threshold = getattr(self.config.model, 'signal_threshold', 0.005)
            
            # Create binary target: 1 for up, 0 for down
            target = (price_change > threshold).astype(int)
            
        elif target_type == 'price_change':
            # Regression: actual price change
            target = data['close'].pct_change(periods=horizon).shift(-horizon)
            
        elif target_type == 'volatility':
            # Volatility prediction
            returns = data['close'].pct_change()
            target = returns.rolling(window=horizon).std().shift(-horizon)
            
        else:
            raise ValueError(f"Unsupported target type: {target_type}")
        
        return target.values
    
    def _create_feature_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Create feature matrix with lookback window."""
        # Select numeric columns for features
        feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target-related columns that might leak information
        exclude_cols = ['close']  # We'll include close as a feature through lookback
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        
        # Create lookback features
        lookback_window = self.config.features.lookback_window
        features_list = []
        feature_names = []
        
        for i in range(lookback_window, len(data)):
            # Get window of data
            window_data = data.iloc[i-lookback_window:i][feature_cols]
            
            # Flatten window into feature vector
            window_features = window_data.values.flatten()
            features_list.append(window_features)
            
            # Create feature names (only for first iteration)
            if len(feature_names) == 0:
                for lag in range(lookback_window):
                    for col in feature_cols:
                        feature_names.append(f"{col}_lag_{lag}")
        
        # Store feature names
        self.feature_names = feature_names
        
        return np.array(features_list)
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train/validation/test sets."""
        # Use time series split to preserve temporal order
        train_size = int(len(X) * self.config.data.train_ratio)
        val_size = int(len(X) * self.config.data.val_ratio)
        
        # Split sequentially (important for time series)
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def fit(self, data: pd.DataFrame) -> 'DataProcessor':
        """Fit the preprocessor."""
        self.processed_data = self.process_data(data)
        self.is_fitted = True
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted parameters."""
        if not self.is_fitted:
            raise ValueError("DataProcessor must be fitted before transform")
        
        # Apply same processing steps
        data_clean = self.clean_data(data)
        data_with_basic = self.add_basic_features(data_clean)
        
        if self.config.features.use_technical_indicators:
            data_with_indicators = self.technical_indicators.add_all_indicators(data_with_basic)
        else:
            data_with_indicators = data_with_basic
        
        data_with_features = self.feature_engineer.engineer_features(data_with_indicators)
        
        return data_with_features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.feature_names if self.feature_names else []
    
    def get_preprocessing_params(self) -> Dict[str, Any]:
        """Get preprocessing parameters for saving."""
        return {
            'scaler_params': self.scaler.get_params() if self.scaler else {},
            'feature_names': self.feature_names,
            'config': self.config.to_dict()
        }
    
    def set_preprocessing_params(self, params: Dict[str, Any]):
        """Set preprocessing parameters for loading."""
        self.preprocessing_params = params
        if 'feature_names' in params:
            self.feature_names = params['feature_names']
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of processed data."""
        if self.processed_data is None:
            return {}
        
        summary = {
            'total_rows': len(self.processed_data),
            'total_features': len(self.processed_data.columns),
            'date_range': {
                'start': str(self.processed_data.index[0]),
                'end': str(self.processed_data.index[-1])
            },
            'missing_values': self.processed_data.isnull().sum().sum(),
            'feature_names': self.get_feature_names()[:10]  # First 10 for brevity
        }
        
        return summary