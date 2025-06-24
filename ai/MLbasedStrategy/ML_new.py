import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.linear_model import LogisticRegression
import joblib
import warnings
from typing import Tuple, Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


def calculate_rsi(prices, period=14):
    """Calculate RSI using pandas"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD using pandas"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """Calculate Bollinger Bands using pandas"""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower


def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator using pandas"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    return k_percent, d_percent


def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R using pandas"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range using pandas"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()


def calculate_cci(high, low, close, period=20):
    """Calculate Commodity Channel Index using pandas"""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean()
    )
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return cci


def calculate_mfi(high, low, close, volume, period=14):
    """Calculate Money Flow Index"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    # Positive and negative money flow
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    # Money flow ratio
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mf_ratio = positive_mf / negative_mf
    mfi = 100 - (100 / (1 + mf_ratio))
    
    return mfi


class CryptoMLStrategy:
    """
    Machine Learning based cryptocurrency trading strategy using technical indicators
    to classify future trading signals (Buy, Sell, Hold).
    """
    
    def __init__(self, 
                 rsi_period: int = 14,
                 macd_fast: int = 12, 
                 macd_slow: int = 26, 
                 macd_signal: int = 9,
                 ema_short: int = 12,
                 ema_long: int = 26,
                 bb_period: int = 20,
                 lookforward: int = 5):
        """
        Initialize the ML trading strategy
        
        Args:
            rsi_period: RSI calculation period
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal line period
            ema_short: Short EMA period
            ema_long: Long EMA period
            bb_period: Bollinger Bands period
            lookforward: Days to look forward for signal generation
        """
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.bb_period = bb_period
        self.lookforward = lookforward
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic technical indicators for feature engineering
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        df = data.copy()
        
        # Basic price features
        df['price_change'] = df['close'].pct_change()
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # RSI - Most important indicator
        df['rsi'] = calculate_rsi(df['close'], self.rsi_period)
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['price_to_sma20'] = df['close'] / df['sma_20']
        
        # MACD - Key momentum indicator
        macd, macd_signal, macd_hist = calculate_macd(
            df['close'], 
            fast=self.macd_fast,
            slow=self.macd_slow, 
            signal=self.macd_signal
        )
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(
            df['close'], 
            period=self.bb_period
        )
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # Basic volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # ATR for risk management
        df['atr'] = calculate_atr(
            df['high'], 
            df['low'], 
            df['close']
        )
        
        return df
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on future price movements with dynamic thresholds
        
        Args:
            data: DataFrame with OHLCV and technical indicators
            
        Returns:
            DataFrame with signals
        """
        df = data.copy()
        
        # Calculate future returns
        df['future_return'] = df['close'].shift(-self.lookforward) / df['close'] - 1
        
        # Dynamic thresholds based on volatility
        volatility = df['close'].pct_change().rolling(window=20).std()
        
        # Adaptive thresholds: tighter in low volatility, wider in high volatility
        buy_threshold = volatility * 1.5  # Dynamic based on volatility
        sell_threshold = -volatility * 1.5
        
        # For 7-day lookforward, use percentile-based thresholds
        if self.lookforward >= 7:
            # Use 70th and 30th percentiles for more balanced classes
            returns = df['future_return'].dropna()
            buy_threshold = returns.quantile(0.70)
            sell_threshold = returns.quantile(0.30)
        
        # Generate signals
        conditions = [
            df['future_return'] > buy_threshold,
            df['future_return'] < sell_threshold
        ]
        choices = ['Buy', 'Sell']
        df['signal'] = np.select(conditions, choices, default='Hold')
        
        # Remove rows with NaN future returns
        df = df.dropna(subset=['future_return'])
        
        return df
    
    def prepare_features(self, data: pd.DataFrame, for_prediction: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for machine learning
        
        Args:
            data: DataFrame with indicators and signals
            for_prediction: If True, skip target extraction (for prediction only)
            
        Returns:
            Feature matrix and target vector (None if for_prediction=True)
        """
        # Simple feature columns - only essential indicators
        feature_cols = [
            'price_change',
            'volume_ratio',
            'rsi',
            'macd',
            'macd_signal', 
            'macd_histogram',
            'sma_20',
            'sma_50',
            'price_to_sma20',
            'bb_upper',
            'bb_middle',
            'bb_lower',
            'bb_width',
            'volatility',
            'atr'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in data.columns]
        self.feature_names = available_cols
        
        # Extract features
        X = data[available_cols].values
        
        # Extract target if not for prediction
        if for_prediction:
            y = None
        else:
            y = data['signal'].values
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        return X, y
    
    def create_simple_model(self):
        """Create a simple RandomForest model for fast training"""
        return RandomForestClassifier(
            n_estimators=100,  # Reduced from 300 for faster training
            max_depth=10,      # Limited depth to prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1         # Use all CPU cores for faster training
        )
    
    def train_model(self, data: pd.DataFrame, test_size: float = 0.2, 
                   model_type: str = 'simple', use_grid_search: bool = False) -> Dict[str, Any]:
        """
        Train the machine learning model
        
        Args:
            data: DataFrame with features and signals
            test_size: Proportion of data for testing
            model_type: Type of model ('random_forest', 'gradient_boosting', or 'ensemble')
            use_grid_search: Whether to use hyperparameter optimization
            
        Returns:
            Training results and metrics
        """
        # Prepare features
        X, y = self.prepare_features(data)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Calculate class weights for balanced training
        class_weights = class_weight.compute_class_weight(
            'balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        if model_type == 'simple':
            self.model = self.create_simple_model()
        elif model_type == 'ensemble':
            self.model = self.create_simple_model()  # Use simple model even for ensemble
        elif model_type == 'random_forest':
            if use_grid_search:
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
                base_model = RandomForestClassifier(
                    class_weight=class_weight_dict,
                    random_state=42, 
                    n_jobs=-1
                )
                self.model = GridSearchCV(
                    base_model, 
                    param_grid, 
                    cv=5, 
                    scoring='accuracy', 
                    n_jobs=-1,
                    verbose=1
                )
            else:
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight=class_weight_dict,
                    random_state=42,
                    n_jobs=-1
                )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError("Model type must be 'random_forest', 'gradient_boosting', or 'ensemble'")
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        self.is_trained = True
        
        # Get feature importances
        if hasattr(self.model, 'feature_importances_'):
            feature_importances = self.model.feature_importances_
        elif hasattr(self.model, 'best_estimator_'):
            feature_importances = self.model.best_estimator_.feature_importances_
        else:
            feature_importances = None
        
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(
                self.label_encoder.inverse_transform(y_test),
                self.label_encoder.inverse_transform(y_pred)
            ),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(
                self.feature_names,
                feature_importances
            )) if feature_importances is not None else None
        }
        
        return results
    
    def predict_signals(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict trading signals for new data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Array of predicted signals
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Calculate technical indicators
        data_with_indicators = self.calculate_technical_indicators(data)
        
        # Prepare features (for prediction only, no target needed)
        X, _ = self.prepare_features(data_with_indicators, for_prediction=True)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions_encoded = self.model.predict(X_scaled)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'best_estimator_'):
            importances = self.model.best_estimator_.feature_importances_
        else:
            raise ValueError("Model does not support feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """
        Save trained model to file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'parameters': {
                'rsi_period': self.rsi_period,
                'macd_fast': self.macd_fast,
                'macd_slow': self.macd_slow,
                'macd_signal': self.macd_signal,
                'ema_short': self.ema_short,
                'ema_long': self.ema_long,
                'bb_period': self.bb_period,
                'lookforward': self.lookforward
            }
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from file
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        
        # Load parameters
        params = model_data['parameters']
        self.rsi_period = params['rsi_period']
        self.macd_fast = params['macd_fast']
        self.macd_slow = params['macd_slow']
        self.macd_signal = params['macd_signal']
        self.ema_short = params['ema_short']
        self.ema_long = params['ema_long']
        self.bb_period = params['bb_period']
        self.lookforward = params['lookforward']
        
        self.is_trained = True
        print(f"Model loaded from {filepath}")


def run_ml_backtest(data_path: str, **kwargs) -> Dict[str, Any]:
    """
    Run complete ML trading strategy backtest
    
    Args:
        data_path: Path to the CSV file with OHLCV data
        **kwargs: Additional parameters for the strategy
        
    Returns:
        Dictionary with backtest results
    """
    # Initialize strategy
    strategy = CryptoMLStrategy(**kwargs)
    
    # Load and prepare data
    print("Loading data...")
    data = strategy.load_data(data_path)
    
    print("Calculating technical indicators...")
    data_with_indicators = strategy.calculate_technical_indicators(data)
    
    print("Generating signals...")
    data_with_signals = strategy.generate_signals(data_with_indicators)
    
    print("Training model...")
    results = strategy.train_model(
        data_with_signals, 
        model_type='ensemble',  # Using ensemble model
        use_grid_search=False   # Set to True for hyperparameter optimization
    )
    
    print("\n" + "="*60)
    print("ML TRADING STRATEGY RESULTS")
    print("="*60)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Cross-validation mean: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    if results['feature_importance']:
        print("\nTop 10 Most Important Features:")
        importance_df = pd.DataFrame(list(results['feature_importance'].items()), 
                                   columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False)
        print(importance_df.head(10).to_string(index=False))
    
    return {
        'strategy': strategy,
        'results': results,
        'data': data_with_signals
    }


# Example usage
if __name__ == "__main__":
    # Run ML strategy on BTC data
    # Get the project root directory (go up two levels from current script)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, "data", "binance", "BTCUSDT", "1d", "2018_01_01-now.csv")
    
    # Test with different lookforward periods
    for lookforward in [3, 5, 7]:
        print(f"\n\nTesting with lookforward={lookforward} days...")
        backtest_results = run_ml_backtest(
            data_path=data_path,
            rsi_period=14,
            lookforward=lookforward
        )
        
        strategy = backtest_results['strategy']
        
        # Save the trained model
        strategy.save_model(f"crypto_ml_strategy_lookforward_{lookforward}.joblib")