# STAS_ML v2 - Redesigned ML System for Crypto Trading

## Overview

STAS_ML v2 is a complete redesign of the cryptocurrency trading ML system, focused on:

- **Clean Architecture**: Modular design with clear separation of concerns
- **Robust Data Processing**: Advanced feature engineering and data validation
- **Multiple Models**: Support for XGBoost, Random Forest, LSTM, and Ensemble models
- **Comprehensive Validation**: Cross-validation and rigorous testing
- **Production Ready**: Model versioning, experiment tracking, and deployment capabilities

## Architecture

```
STAS_ML_v2/
├── core/                   # Core business logic
│   ├── config.py          # Configuration management
│   ├── trainer.py         # Main training orchestrator
│   └── base.py            # Abstract base classes
├── data/                   # Data processing
│   ├── processor.py       # Main data processor
│   ├── features.py        # Feature engineering
│   └── indicators.py      # Technical indicators
├── models/                 # ML models
│   ├── xgboost_model.py   # XGBoost implementation
│   ├── rf_model.py        # Random Forest implementation
│   ├── lstm_model.py      # LSTM implementation
│   └── ensemble.py        # Ensemble models
├── validation/             # Model validation
│   └── validator.py       # Cross-validation and metrics
├── backtesting/            # Trading backtesting
│   └── engine.py          # Backtesting engine
└── train.py               # Training script
```

## Key Improvements over v1

### 1. **Modular Architecture**
- Clear separation of data processing, model training, validation, and backtesting
- Abstract base classes ensure consistency across components
- Easy to extend with new models or features

### 2. **Advanced Configuration System**
- Type-safe configuration with validation
- Support for JSON/YAML config files
- Environment-specific configurations

### 3. **Robust Data Processing**
- Comprehensive data cleaning and validation
- Advanced feature engineering pipeline
- Multiple technical indicators with fallback implementations

### 4. **Model Flexibility**
- Multiple model types (XGBoost, Random Forest, LSTM, Ensemble)
- Consistent interface across all models
- Easy model comparison and selection

### 5. **Comprehensive Validation**
- Cross-validation with multiple strategies
- Extensive metrics calculation
- Overfitting detection

### 6. **Production Features**
- Model versioning and serialization
- Experiment tracking and logging
- Comprehensive backtesting framework

## Quick Start

### Basic Training

```bash
# Train XGBoost model on BTCUSDT
python train.py --symbol BTCUSDT --model xgboost --target direction

# Train Random Forest with custom experiment name
python train.py --symbol ETHUSDT --model random_forest --experiment-name eth_rf_v1

# Quick training (skip full validation)
python train.py --symbol BTCUSDT --model xgboost --quick
```

### Advanced Configuration

Create a config file `my_config.json`:

```json
{
  "data": {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "start_date": "2022-01-01",
    "end_date": "2024-12-31"
  },
  "model": {
    "model_type": "ensemble",
    "target_type": "direction"
  },
  "features": {
    "use_technical_indicators": true,
    "lookback_window": 50
  },
  "backtest": {
    "initial_capital": 10000,
    "transaction_cost": 0.001
  }
}
```

```bash
python train.py --config my_config.json
```

### Programmatic Usage

```python
from STAS_ML_v2.core.config import create_default_config
from STAS_ML_v2.core.trainer import ModelTrainer

# Create configuration
config = create_default_config("BTCUSDT", "xgboost", "direction")

# Train model
trainer = ModelTrainer(config)
results = trainer.train_complete_pipeline()

# Get summary
summary = trainer.get_training_summary()
print(f"Test Accuracy: {summary['test_accuracy']:.4f}")
print(f"Backtest Return: {summary['backtest_total_return']:.2%}")
```

## Configuration Options

### Data Configuration
- `symbol`: Trading symbol (e.g., "BTCUSDT")
- `timeframe`: Data timeframe ("1m", "5m", "1h", "1d")
- `start_date`/`end_date`: Date range for training data
- `train_ratio`/`val_ratio`/`test_ratio`: Data split ratios

### Model Configuration
- `model_type`: "xgboost", "random_forest", "lstm", "ensemble", "linear"
- `target_type`: "direction", "price_change", "volatility"
- Model-specific parameters for each model type

### Feature Configuration
- `use_technical_indicators`: Enable technical indicators
- `lookback_window`: Number of past periods to use as features
- `rsi_periods`, `macd_config`, etc.: Technical indicator parameters

### Backtesting Configuration
- `initial_capital`: Starting capital for backtesting
- `transaction_cost`: Transaction cost percentage
- `stop_loss`/`take_profit`: Risk management parameters

## Model Types

### XGBoost (`xgboost`)
- Gradient boosting with excellent performance
- Good for both classification and regression
- Feature importance available

### Random Forest (`random_forest`)
- Ensemble of decision trees
- Robust to overfitting
- Feature importance available

### LSTM (`lstm`)
- Deep learning for sequence modeling
- Captures temporal dependencies
- Requires PyTorch

### Ensemble (`ensemble`)
- Combines multiple models
- Typically provides best performance
- Slower training but robust

### Linear (`linear`)
- Fast and interpretable
- Good baseline model
- Limited capacity for complex patterns

## Validation and Metrics

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (when probabilities available)
- Cross-validation scores

### Regression Metrics
- MSE, MAE, RMSE, R², MAPE
- Cross-validation scores

### Trading Metrics
- Total Return, Annual Return
- Sharpe Ratio, Calmar Ratio
- Maximum Drawdown
- Win Rate, Profit Factor

## Best Practices

### 1. Data Quality
- Ensure clean, validated data
- Use appropriate date ranges
- Check for data gaps or anomalies

### 2. Feature Engineering
- Start with basic features, add complexity gradually
- Use domain knowledge for feature selection
- Monitor feature importance

### 3. Model Selection
- Start with XGBoost or Random Forest
- Try ensemble models for better performance
- Use cross-validation for model comparison

### 4. Validation
- Always use time-series cross-validation for financial data
- Monitor for overfitting (train vs. validation performance)
- Test on out-of-sample data

### 5. Backtesting
- Use realistic transaction costs
- Implement proper risk management
- Consider market conditions during backtest period

## Common Issues and Solutions

### 1. Insufficient Data
- Increase date range
- Lower minimum data points requirement
- Use simpler models

### 2. Poor Performance
- Check data quality
- Increase feature engineering
- Try different model types
- Adjust model parameters

### 3. Overfitting
- Reduce model complexity
- Increase regularization
- Use more validation data
- Implement early stopping

### 4. Long Training Times
- Use fewer features
- Reduce lookback window
- Use simpler models
- Implement parallel processing

## Example Workflows

### Model Comparison
```bash
# Train different models on same data
python train.py --symbol BTCUSDT --model xgboost --experiment-name btc_xgb
python train.py --symbol BTCUSDT --model random_forest --experiment-name btc_rf
python train.py --symbol BTCUSDT --model ensemble --experiment-name btc_ensemble

# Compare results in experiments/ directory
```

### Parameter Tuning
```python
from STAS_ML_v2.core.config import create_default_config
from STAS_ML_v2.core.trainer import ModelTrainer

# Test different lookback windows
for lookback in [20, 30, 50]:
    config = create_default_config("BTCUSDT", "xgboost", "direction")
    config.features.lookback_window = lookback
    config.experiment_name = f"btc_xgb_lookback_{lookback}"
    
    trainer = ModelTrainer(config)
    results = trainer.train_complete_pipeline()
```

## Dependencies

### Required
- pandas, numpy, scikit-learn
- joblib (for model serialization)

### Optional
- xgboost (for XGBoost models)
- torch (for LSTM models)
- talib (for technical indicators)
- yaml (for YAML config files)

## License

MIT License - see LICENSE file for details.