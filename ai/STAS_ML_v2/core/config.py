"""
Configuration system for STAS_ML v2

Clean, type-safe configuration with validation and environment support.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import json
import yaml
from pathlib import Path


class ModelType(Enum):
    """Supported model types."""
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"
    LINEAR = "linear"


class TargetType(Enum):
    """Target prediction types."""
    DIRECTION = "direction"  # Buy/Sell classification
    PRICE_CHANGE = "price_change"  # Regression
    VOLATILITY = "volatility"  # Volatility prediction


class TimeFrame(Enum):
    """Supported timeframes."""
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"


@dataclass
class DataConfig:
    """Data configuration."""
    symbol: str = "BTCUSDT"
    exchange: str = "binance"  
    timeframe: TimeFrame = TimeFrame.DAY_1
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    data_path: Optional[str] = None
    
    # Data splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Data quality
    min_data_points: int = 1000
    fill_missing: bool = True
    outlier_removal: bool = True
    outlier_threshold: float = 3.0  # Z-score threshold
    
    def __post_init__(self):
        """Validate configuration."""
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 0.001:
            raise ValueError("Data split ratios must sum to 1.0")
        
        if self.data_path is None:
            # Auto-construct data path
            project_root = Path(__file__).parent.parent.parent.parent.parent
            self.data_path = str(
                project_root / "CryptoTrade" / "data" / self.exchange / 
                self.symbol / self.timeframe.value / "2018_01_01-now.csv"
            )


@dataclass  
class FeatureConfig:
    """Feature engineering configuration."""
    # Technical indicators
    use_technical_indicators: bool = True
    
    # Price-based features
    price_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    volume_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Technical indicators with periods
    rsi_periods: List[int] = field(default_factory=lambda: [14, 21])
    macd_config: Dict[str, int] = field(default_factory=lambda: {
        "fast": 12, "slow": 26, "signal": 9
    })
    bollinger_periods: List[int] = field(default_factory=lambda: [20])
    atr_periods: List[int] = field(default_factory=lambda: [14])
    
    # Advanced features
    use_market_regime: bool = True
    use_volatility_features: bool = True
    use_momentum_features: bool = True
    use_mean_reversion_features: bool = True
    
    # Feature selection
    max_features: Optional[int] = 200
    feature_selection_method: str = "importance"  # "importance", "correlation", "mutual_info"
    correlation_threshold: float = 0.95
    
    # Lookback window
    lookback_window: int = 30
    prediction_horizon: int = 1


@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: ModelType = ModelType.XGBOOST
    target_type: TargetType = TargetType.DIRECTION
    
    # XGBoost parameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1
    })
    
    # Random Forest parameters
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1
    })
    
    # LSTM parameters
    lstm_params: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.001,
        "patience": 10
    })
    
    # Ensemble parameters
    ensemble_params: Dict[str, Any] = field(default_factory=lambda: {
        "models": ["xgboost", "random_forest"],
        "voting": "soft",  # "hard" or "soft"
        "weights": None  # Auto-calculated if None
    })


@dataclass
class ValidationConfig:
    """Model validation configuration."""
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "time_series"  # "time_series", "stratified", "kfold"
    
    # Metrics to track
    classification_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "roc_auc"
    ])
    regression_metrics: List[str] = field(default_factory=lambda: [
        "mse", "mae", "rmse", "r2", "mape"
    ])
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    
    # Overfitting detection
    overfitting_threshold: float = 0.1  # Train-Val performance gap


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 10000.0
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    
    # Risk management
    max_position_size: float = 0.95  # 95% of capital
    stop_loss: Optional[float] = 0.03  # 3%
    take_profit: Optional[float] = 0.06  # 6%
    max_holding_period: Optional[int] = 7  # days
    
    # Signal filtering
    min_confidence: float = 0.6
    signal_threshold: float = 0.005  # 0.5% minimum price change
    
    # Performance metrics
    benchmark_symbol: str = "BTCUSDT"  # Buy and hold benchmark


@dataclass
class Config:
    """Main configuration class."""
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    # Experiment settings
    experiment_name: Optional[str] = None
    random_seed: int = 42
    verbose: bool = True
    
    # Paths
    output_dir: str = "experiments"
    model_dir: str = "models"
    log_dir: str = "logs"
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.experiment_name is None:
            # Auto-generate experiment name
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"{self.data.symbol}_{self.data.timeframe.value}_{self.model.model_type.value}_{timestamp}"
        
        # Create directories
        for dir_path in [self.output_dir, self.model_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            features=FeatureConfig(**config_dict.get('features', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            validation=ValidationConfig(**config_dict.get('validation', {})),
            backtest=BacktestConfig(**config_dict.get('backtest', {})),
            experiment_name=config_dict.get('experiment_name'),
            random_seed=config_dict.get('random_seed', 42),
            verbose=config_dict.get('verbose', True),
            output_dir=config_dict.get('output_dir', 'experiments'),
            model_dir=config_dict.get('model_dir', 'models'),
            log_dir=config_dict.get('log_dir', 'logs')
        )
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load config from JSON or YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError("Config file must be JSON or YAML")
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data': {
                'symbol': self.data.symbol,
                'exchange': self.data.exchange,
                'timeframe': self.data.timeframe.value,
                'start_date': self.data.start_date,
                'end_date': self.data.end_date,
                'train_ratio': self.data.train_ratio,
                'val_ratio': self.data.val_ratio,
                'test_ratio': self.data.test_ratio
            },
            'features': {
                'use_technical_indicators': self.features.use_technical_indicators,
                'lookback_window': self.features.lookback_window,
                'prediction_horizon': self.features.prediction_horizon,
                'max_features': self.features.max_features
            },
            'model': {
                'model_type': self.model.model_type.value,
                'target_type': self.model.target_type.value,
                'xgb_params': self.model.xgb_params,
                'rf_params': self.model.rf_params
            },
            'validation': {
                'cv_folds': self.validation.cv_folds,
                'cv_strategy': self.validation.cv_strategy,
                'early_stopping': self.validation.early_stopping
            },
            'backtest': {
                'initial_capital': self.backtest.initial_capital,
                'transaction_cost': self.backtest.transaction_cost,
                'min_confidence': self.backtest.min_confidence
            },
            'experiment_name': self.experiment_name,
            'random_seed': self.random_seed,
            'verbose': self.verbose
        }
    
    def save(self, config_path: str):
        """Save config to file."""
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        if config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.suffix.lower() in ['.yml', '.yaml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        else:
            # Default to JSON
            with open(config_path.with_suffix('.json'), 'w') as f:
                json.dump(config_dict, f, indent=2)
    
    def validate(self):
        """Validate configuration."""
        errors = []
        
        # Validate data config
        if self.data.train_ratio <= 0 or self.data.val_ratio <= 0 or self.data.test_ratio <= 0:
            errors.append("All data split ratios must be positive")
        
        # Validate feature config
        if self.features.lookback_window <= 0:
            errors.append("Lookback window must be positive")
        
        if self.features.prediction_horizon <= 0:
            errors.append("Prediction horizon must be positive")
        
        # Validate model config
        if self.model.model_type not in ModelType:
            errors.append(f"Invalid model type: {self.model.model_type}")
        
        # Validate backtest config
        if self.backtest.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        
        if self.backtest.transaction_cost < 0 or self.backtest.transaction_cost > 1:
            errors.append("Transaction cost must be between 0 and 1")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors))
        
        return True


def create_default_config(symbol: str = "BTCUSDT", 
                         model_type: str = "xgboost",
                         target_type: str = "direction") -> Config:
    """Create a default configuration."""
    config = Config()
    config.data.symbol = symbol
    config.model.model_type = ModelType(model_type)
    config.model.target_type = TargetType(target_type)
    
    return config


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or create default."""
    if config_path and Path(config_path).exists():
        return Config.from_file(config_path)
    else:
        return create_default_config()


if __name__ == "__main__":
    # Test configuration
    config = create_default_config()
    config.validate()
    print("✅ Configuration validation passed")
    
    # Save example config
    config.save("example_config.json")
    print("✅ Example configuration saved")