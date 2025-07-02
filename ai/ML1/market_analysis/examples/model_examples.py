"""
Examples of using different models for market analysis.

This script demonstrates how to create, train, and evaluate different models,
including LSTM, GRU, Transformer, XGBoost, and ensemble models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from parent package
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    LSTMModel, GRUModel, TransformerModel, XGBoostModel,
    StackingEnsembleModel, VotingEnsembleModel, ModelFactory
)
from data.fetchers import BinanceFetcher
from data.processors import PriceProcessor
from data.features import TechnicalIndicators, FeatureSelector


def prepare_data(symbol='BTCUSDT', timeframe='1d', start_date='2020-01-01', end_date=None, window_size=60):
    """
    Prepare data for model training and evaluation.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe for the data
        start_date: Start date for the data
        end_date: End date for the data
        window_size: Window size for sequence data
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    print(f"Fetching data for {symbol}...")
    
    # Use CSV fetcher instead of Binance fetcher to avoid API issues
    # Create a simple DataFrame with random data for testing
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    print(f"Creating synthetic data for testing...")
    
    # Create date range
    end_date_dt = datetime.now() if end_date is None else pd.to_datetime(end_date)
    start_date_dt = pd.to_datetime(start_date)
    date_range = pd.date_range(start=start_date_dt, end=end_date_dt, freq='D')
    
    # Create random price data
    np.random.seed(42)  # For reproducibility
    close_prices = np.random.normal(loc=100, scale=10, size=len(date_range))
    close_prices = np.cumsum(np.random.normal(loc=0, scale=1, size=len(date_range))) + 100
    
    # Ensure prices are positive
    close_prices = np.maximum(close_prices, 1)
    
    # Calculate other OHLCV data
    high_prices = close_prices * np.random.uniform(1.0, 1.05, size=len(date_range))
    low_prices = close_prices * np.random.uniform(0.95, 1.0, size=len(date_range))
    open_prices = low_prices + np.random.uniform(0, 1, size=len(date_range)) * (high_prices - low_prices)
    volumes = np.random.uniform(1000, 10000, size=len(date_range))
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=date_range)
    
    # Add technical indicators first
    indicators = TechnicalIndicators()
    df = indicators.add_all_indicators(df)
    
    # Process data
    processor = PriceProcessor()
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Use the processor to handle the rest of the data preparation
    X_train, X_val, X_test, y_train, y_val, y_test, target_scaler = processor.process(df)
    
    print(f"Data prepared. Shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler


def evaluate_model(model, X_test, y_test, scaler, name=None):
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        scaler: Scaler used to transform the target
        name: Optional name for the model
        
    Returns:
        Dictionary of evaluation metrics
    """
    name = name or model.name
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Reshape arrays to 2D for inverse_transform
    y_test_2d = y_test.reshape(-1, 1)
    y_pred_2d = y_pred.reshape(-1, 1)
    
    # Inverse transform the predictions and actual values
    y_test_inv = scaler.inverse_transform(y_test_2d)
    y_pred_inv = scaler.inverse_transform(y_pred_2d)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    print(f"\nEvaluation for {name}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {
        'name': name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_test': y_test_inv,
        'y_pred': y_pred_inv
    }


def plot_predictions(results, title="Model Predictions"):
    """
    Plot model predictions against actual values.
    
    Args:
        results: List of dictionaries with evaluation results
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot actual values
    plt.plot(results[0]['y_test'], label='Actual', color='black', linewidth=2)
    
    # Plot predictions for each model
    for result in results:
        plt.plot(result['y_pred'], label=f"{result['name']} (RMSE: {result['rmse']:.4f})")
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f"plots/{title.replace(' ', '_').lower()}.png")
    
    plt.show()


def example_individual_models():
    """
    Example of training and evaluating individual models.
    """
    print("\n=== Example: Individual Models ===\n")
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data(
        symbol='BTCUSDT', 
        timeframe='1d',
        window_size=60
    )
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Create models
    models = [
        LSTMModel(input_shape=input_shape, units=64, dropout=0.2),
        GRUModel(input_shape=input_shape, units=64, dropout=0.2),
        TransformerModel(
            input_shape=input_shape, 
            embed_dim=32, 
            num_heads=2, 
            ff_dim=64, 
            num_transformer_blocks=2
        ),
        XGBoostModel(max_depth=5, n_estimators=100)
    ]
    
    # Train and evaluate models
    results = []
    
    for model in models:
        print(f"\nTraining {model.name}...")
        
        # Reshape data for XGBoost
        if isinstance(model, XGBoostModel):
            # Flatten the sequence dimension for XGBoost
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            model.train(X_train_flat, y_train, X_val_flat, y_val, verbose=1)
            result = evaluate_model(model, X_test_flat, y_test, scaler)
        else:
            model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=1)
            result = evaluate_model(model, X_test, y_test, scaler)
        
        results.append(result)
    
    # Plot predictions
    plot_predictions(results, title="Individual Model Predictions")


def example_ensemble_models():
    """
    Example of training and evaluating ensemble models.
    """
    print("\n=== Example: Ensemble Models ===\n")
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data(
        symbol='BTCUSDT', 
        timeframe='1d',
        window_size=60
    )
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Create base models
    lstm_model = LSTMModel(input_shape=input_shape, units=64, dropout=0.2, name="LSTM_Base")
    gru_model = GRUModel(input_shape=input_shape, units=64, dropout=0.2, name="GRU_Base")
    transformer_model = TransformerModel(
        input_shape=input_shape, 
        embed_dim=32, 
        num_heads=2, 
        ff_dim=64, 
        num_transformer_blocks=2,
        name="Transformer_Base"
    )
    
    # Reshape data for XGBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    xgboost_model = XGBoostModel(max_depth=5, n_estimators=100, name="XGBoost_Base")
    
    # Train base models
    print("Training base models...")
    
    lstm_model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=0)
    gru_model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=0)
    transformer_model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=0)
    xgboost_model.train(X_train_flat, y_train, X_val_flat, y_val, verbose=0)
    
    # Create ensemble models
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    
    # Stacking ensemble with linear regression meta-model
    stacking_linear = StackingEnsembleModel(
        base_models=[lstm_model, gru_model, transformer_model, xgboost_model],
        meta_model=Ridge(alpha=0.5),
        name="Stacking_Linear"
    )
    
    # Stacking ensemble with random forest meta-model
    stacking_rf = StackingEnsembleModel(
        base_models=[lstm_model, gru_model, transformer_model, xgboost_model],
        meta_model=RandomForestRegressor(n_estimators=100),
        name="Stacking_RF"
    )
    
    # Voting ensemble with equal weights
    voting_equal = VotingEnsembleModel(
        base_models=[lstm_model, gru_model, transformer_model, xgboost_model],
        name="Voting_Equal"
    )
    
    # Voting ensemble with custom weights
    voting_custom = VotingEnsembleModel(
        base_models=[lstm_model, gru_model, transformer_model, xgboost_model],
        weights=[0.2, 0.2, 0.2, 0.4],  # Give more weight to XGBoost
        name="Voting_Custom"
    )
    
    # Train ensemble models
    print("\nTraining ensemble models...")
    
    # For stacking ensembles, we need to handle the different input shapes
    def prepare_ensemble_data(X, is_flat=False):
        """Prepare data for ensemble models with mixed base models."""
        if is_flat:
            return X
        else:
            return X.reshape(X.shape[0], -1)
    
    # Train stacking ensembles
    stacking_linear.train(
        prepare_ensemble_data(X_train, is_flat=True), 
        y_train,
        prepare_ensemble_data(X_val, is_flat=True), 
        y_val,
        verbose=1
    )
    
    stacking_rf.train(
        prepare_ensemble_data(X_train, is_flat=True), 
        y_train,
        prepare_ensemble_data(X_val, is_flat=True), 
        y_val,
        verbose=1
    )
    
    # For voting ensembles, we need a custom predict method to handle different input shapes
    def custom_predict(model, X):
        """Custom predict method for voting ensembles with mixed base models."""
        predictions = []
        
        for base_model in model.base_models:
            if isinstance(base_model, XGBoostModel):
                # Flatten the sequence dimension for XGBoost
                X_flat = X.reshape(X.shape[0], -1)
                pred = base_model.predict(X_flat)
            else:
                pred = base_model.predict(X)
            
            predictions.append(pred)
        
        # Combine predictions using weighted average
        weighted_preds = np.zeros_like(predictions[0])
        for i, preds in enumerate(predictions):
            weighted_preds += preds * model.weights[i]
        
        return weighted_preds
    
    # Evaluate ensemble models
    results = []
    
    # Evaluate stacking ensembles
    stacking_linear_result = evaluate_model(
        stacking_linear, 
        prepare_ensemble_data(X_test, is_flat=True), 
        y_test, 
        scaler
    )
    results.append(stacking_linear_result)
    
    stacking_rf_result = evaluate_model(
        stacking_rf, 
        prepare_ensemble_data(X_test, is_flat=True), 
        y_test, 
        scaler
    )
    results.append(stacking_rf_result)
    
    # Evaluate voting ensembles using custom predict
    voting_equal_pred = custom_predict(voting_equal, X_test)
    voting_equal_result = {
        'name': voting_equal.name,
        'rmse': np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(voting_equal_pred))),
        'mae': mean_absolute_error(scaler.inverse_transform(y_test), scaler.inverse_transform(voting_equal_pred)),
        'r2': r2_score(scaler.inverse_transform(y_test), scaler.inverse_transform(voting_equal_pred)),
        'y_test': scaler.inverse_transform(y_test),
        'y_pred': scaler.inverse_transform(voting_equal_pred)
    }
    print(f"\nEvaluation for {voting_equal_result['name']}:")
    print(f"RMSE: {voting_equal_result['rmse']:.4f}")
    print(f"MAE: {voting_equal_result['mae']:.4f}")
    print(f"R²: {voting_equal_result['r2']:.4f}")
    results.append(voting_equal_result)
    
    voting_custom_pred = custom_predict(voting_custom, X_test)
    voting_custom_result = {
        'name': voting_custom.name,
        'rmse': np.sqrt(mean_squared_error(scaler.inverse_transform(y_test), scaler.inverse_transform(voting_custom_pred))),
        'mae': mean_absolute_error(scaler.inverse_transform(y_test), scaler.inverse_transform(voting_custom_pred)),
        'r2': r2_score(scaler.inverse_transform(y_test), scaler.inverse_transform(voting_custom_pred)),
        'y_test': scaler.inverse_transform(y_test),
        'y_pred': scaler.inverse_transform(voting_custom_pred)
    }
    print(f"\nEvaluation for {voting_custom_result['name']}:")
    print(f"RMSE: {voting_custom_result['rmse']:.4f}")
    print(f"MAE: {voting_custom_result['mae']:.4f}")
    print(f"R²: {voting_custom_result['r2']:.4f}")
    results.append(voting_custom_result)
    
    # Plot predictions
    plot_predictions(results, title="Ensemble Model Predictions")


def example_model_factory():
    """
    Example of using the model factory to create models.
    """
    print("\n=== Example: Model Factory ===\n")
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_data(
        symbol='BTCUSDT', 
        timeframe='1d',
        window_size=60
    )
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Create models using the factory
    lstm_model = ModelFactory.create_model(
        'lstm',
        input_shape=input_shape,
        units=64,
        dropout=0.2,
        name="LSTM_Factory"
    )
    
    gru_model = ModelFactory.create_model(
        'gru',
        input_shape=input_shape,
        units=64,
        dropout=0.2,
        name="GRU_Factory"
    )
    
    transformer_model = ModelFactory.create_model(
        'transformer',
        input_shape=input_shape,
        embed_dim=32,
        num_heads=2,
        ff_dim=64,
        num_transformer_blocks=2,
        name="Transformer_Factory"
    )
    
    xgboost_model = ModelFactory.create_model(
        'xgboost',
        max_depth=5,
        n_estimators=100,
        name="XGBoost_Factory"
    )
    
    # Train models
    print("Training models created with factory...")
    
    lstm_model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=1)
    gru_model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=1)
    transformer_model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, verbose=1)
    
    # Reshape data for XGBoost
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    xgboost_model.train(X_train_flat, y_train, X_val_flat, y_val, verbose=1)
    
    # Create ensemble model using the factory
    ensemble_config = {
        'ensemble_type': 'stacking',
        'name': 'Stacking_Factory',
        'base_models': [
            {
                'type': 'lstm',
                'input_shape': input_shape,
                'units': 64,
                'dropout': 0.2,
                'name': 'LSTM_Base'
            },
            {
                'type': 'gru',
                'input_shape': input_shape,
                'units': 64,
                'dropout': 0.2,
                'name': 'GRU_Base'
            },
            {
                'type': 'xgboost',
                'max_depth': 5,
                'n_estimators': 100,
                'name': 'XGBoost_Base'
            }
        ],
        'meta_model': {
            'type': 'random_forest',
            'n_estimators': 100
        },
        'use_features_in_meta': False
    }
    
    # This is just to demonstrate the concept - in practice, you would need to handle
    # the different input shapes for the base models as shown in the previous example
    print("\nCreating ensemble model with factory (demonstration only)...")
    ensemble_model = ModelFactory.create_ensemble_from_config(ensemble_config, input_shape)
    print(f"Created ensemble model: {ensemble_model.name}")
    print(f"Base models: {[model.name for model in ensemble_model.base_models]}")
    print(f"Meta-model type: {type(ensemble_model.meta_model).__name__}")


if __name__ == "__main__":
    # Create directories for saving models and plots
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Run examples
    example_individual_models()
    example_ensemble_models()
    example_model_factory()