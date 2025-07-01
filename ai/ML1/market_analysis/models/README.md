# Models Module for Market Analysis

This module contains various model implementations for time series prediction in the market analysis system.

## Implemented Models

### Base Models

- **BaseModel**: Abstract base class that defines the interface for all models
- **LSTMModel**: Long Short-Term Memory neural network model
- **GRUModel**: Gated Recurrent Unit neural network model
- **TransformerModel**: Transformer-based model with self-attention mechanisms
- **XGBoostModel**: Gradient boosting tree-based model

### Ensemble Models

- **StackingEnsembleModel**: Combines multiple base models using a meta-model
- **VotingEnsembleModel**: Combines multiple base models using weighted averaging

### Utilities

- **ModelFactory**: Factory class for creating and configuring different types of models

## Usage

### Creating Models

```python
from market_analysis.models import ModelFactory

# Create an LSTM model
lstm_model = ModelFactory.create_model(
    'lstm',
    input_shape=(60, 10),  # (sequence_length, features)
    units=64,
    dropout=0.2,
    learning_rate=0.001
)

# Create an XGBoost model
xgboost_model = ModelFactory.create_model(
    'xgboost',
    max_depth=5,
    n_estimators=100
)

# Create a stacking ensemble model
from sklearn.linear_model import Ridge
ensemble_model = ModelFactory.create_model(
    'stacking_ensemble',
    base_models=[lstm_model, xgboost_model],
    meta_model=Ridge(alpha=0.5)
)
```

### Training Models

```python
# Train a neural network model
lstm_model.train(
    X_train, y_train,
    X_val, y_val,
    batch_size=32,
    epochs=100
)

# Train an XGBoost model
# Note: XGBoost requires flattened input
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
xgboost_model.train(
    X_train_flat, y_train,
    X_val_flat, y_val
)
```

### Making Predictions

```python
# Make predictions with a neural network model
predictions = lstm_model.predict(X_test)

# Make predictions with an XGBoost model
X_test_flat = X_test.reshape(X_test.shape[0], -1)
predictions = xgboost_model.predict(X_test_flat)
```

### Saving and Loading Models

```python
# Save a model
model.save('path/to/model')

# Load a model
model.load('path/to/model')
```

## Examples

See the `examples/model_examples.py` script for complete examples of using the models.

## Next Steps

The following components still need to be implemented to complete the market analysis system:

1. **Training Module**:
   - Implement a Trainer class for more advanced training capabilities
   - Add support for hyperparameter tuning
   - Implement cross-validation

2. **Evaluation Module**:
   - Implement metrics calculation
   - Implement backtesting for trading strategies
   - Implement performance evaluation

3. **Trading Module**:
   - Implement trading strategies
   - Implement portfolio management
   - Implement risk management

4. **Visualization Module**:
   - Implement price charts
   - Implement performance charts
   - Implement indicator visualization

5. **Documentation**:
   - Add docstrings to all classes and methods
   - Create user guides
   - Create API documentation