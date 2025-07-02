"""
Main entry point for the market analysis system.
"""

import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from data.fetchers.binance_fetcher import BinanceFetcher
from data.fetchers.csv_fetcher import CSVFetcher
from data.processors.price_processor import PriceProcessor
from data.features.technical_indicators import TechnicalIndicators
from data.features.feature_selector import FeatureSelector
from models.model_factory import ModelFactory


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cryptocurrency Market Analysis')
    
    # Data parameters
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--timeframe', type=str, default='1d',
                        help='Timeframe interval (default: 1d)')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                        help='Start date for data (default: 2020-01-01)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for data (default: None/current)')
    parser.add_argument('--data_source', type=str, default='binance', choices=['binance', 'csv'],
                        help='Data source (default: binance)')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Path to CSV file if data_source is csv')
    
    # Preprocessing parameters
    parser.add_argument('--window_size', type=int, default=60,
                        help='Window size for sequence data (default: 60)')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Training data split ratio (default: 0.7)')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation data split ratio (default: 0.15)')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='lstm',
                        choices=['lstm', 'gru', 'transformer', 'xgboost', 'stacking', 'voting'],
                        help='Model type to use (default: lstm)')
    parser.add_argument('--ensemble_models', type=str, default='lstm,xgboost',
                        help='Comma-separated list of models for ensemble (default: lstm,xgboost)')
    parser.add_argument('--units', type=int, default=50,
                        help='Number of units in LSTM/GRU layers (default: 50)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (default: 0.2)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    
    # File operations
    parser.add_argument('--save_model', type=str, default=None,
                        help='Path to save the trained model (default: None)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load a pre-trained model (default: None)')
    
    # Execution flags
    parser.add_argument('--no_train', action='store_true',
                        help='Skip training and use loaded model only')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip plotting results')
    
    return parser.parse_args()


def fetch_data(args):
    """Fetch data from the specified source."""
    print(f"📊 Fetching {args.symbol} data from {args.start_date}...")
    
    if args.data_source == 'binance':
        fetcher = BinanceFetcher(symbol=args.symbol, interval=args.timeframe)
        data = fetcher.fetch_data(
            start_date=args.start_date,
            end_date=args.end_date
        )
    elif args.data_source == 'csv':
        if args.csv_path is None:
            raise ValueError("CSV path must be provided when data_source is 'csv'")
        fetcher = CSVFetcher(symbol=args.symbol, interval=args.timeframe)
        data = fetcher.fetch_data(
            start_date=args.start_date,
            end_date=args.end_date,
            path=args.csv_path
        )
    else:
        raise ValueError(f"Unsupported data source: {args.data_source}")
    
    if data is None or len(data) == 0:
        raise ValueError("Failed to fetch data")
    
    print(f"✅ Fetched {len(data)} data points")
    return data


def process_data(data, args):
    """Process the data for model training and evaluation."""
    print("🔄 Processing data...")
    
    # Process price data
    processor = PriceProcessor()
    df = processor.process(data)
    
    # Add technical indicators
    indicators = TechnicalIndicators()
    df = indicators.add_all_indicators(df)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Select features
    selector = FeatureSelector()
    selected_features = selector.select_features(df, target_column='close')
    
    # Prepare features and target
    features = df[selected_features].values
    target = df['close'].values.reshape(-1, 1)
    
    # Scale the data
    from sklearn.preprocessing import MinMaxScaler
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target)
    
    # Create sequences
    X, y = [], []
    for i in range(len(features_scaled) - args.window_size):
        X.append(features_scaled[i:i+args.window_size])
        y.append(target_scaled[i+args.window_size])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split the data
    train_size = int(len(X) * args.train_split)
    val_size = int(len(X) * args.val_split)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    print(f"✅ Data processed. Shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, target_scaler


def create_model(args, input_shape):
    """Create a model based on the specified type."""
    print(f"🏗️ Creating {args.model_type} model...")
    
    if args.model_type in ['lstm', 'gru', 'transformer']:
        # Create neural network model
        return ModelFactory.create_model(
            args.model_type,
            input_shape=input_shape,
            units=args.units,
            dropout=args.dropout,
            learning_rate=args.learning_rate
        )
    elif args.model_type == 'xgboost':
        # Create XGBoost model
        return ModelFactory.create_model(
            'xgboost',
            learning_rate=args.learning_rate
        )
    elif args.model_type in ['stacking', 'voting']:
        # Create ensemble model
        model_names = args.ensemble_models.split(',')
        base_models = []
        
        for model_name in model_names:
            if model_name in ['lstm', 'gru', 'transformer']:
                base_model = ModelFactory.create_model(
                    model_name,
                    input_shape=input_shape,
                    units=args.units,
                    dropout=args.dropout,
                    learning_rate=args.learning_rate,
                    name=f"{model_name.capitalize()}_Base"
                )
            elif model_name == 'xgboost':
                base_model = ModelFactory.create_model(
                    'xgboost',
                    learning_rate=args.learning_rate,
                    name="XGBoost_Base"
                )
            else:
                raise ValueError(f"Unsupported model type for ensemble: {model_name}")
            
            base_models.append(base_model)
        
        if args.model_type == 'stacking':
            from sklearn.linear_model import Ridge
            return ModelFactory.create_model(
                'stacking_ensemble',
                base_models=base_models,
                meta_model=Ridge(alpha=0.5),
                name="Stacking_Ensemble"
            )
        else:  # voting
            return ModelFactory.create_model(
                'voting_ensemble',
                base_models=base_models,
                name="Voting_Ensemble"
            )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")


def train_model(model, X_train, y_train, X_val, y_val, args):
    """Train the model on the provided data."""
    print("🏋️ Training model...")
    
    # Prepare data for XGBoost if needed
    from market_analysis.models.xgboost_model import XGBoostModel
    if isinstance(model, XGBoostModel):
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        # Train XGBoost model
        model.train(
            X_train_flat, y_train,
            X_val_flat, y_val,
            verbose=1
        )
    else:
        # Train neural network model
        model.train(
            X_train, y_train,
            X_val, y_val,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1
        )
    
    print("✅ Model training completed")
    return model


def evaluate_model(model, X_test, y_test, scaler):
    """Evaluate the model on test data."""
    print("📏 Evaluating model...")
    
    # Prepare data for XGBoost if needed
    from market_analysis.models.xgboost_model import XGBoostModel
    if isinstance(model, XGBoostModel):
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        y_pred = model.predict(X_test_flat)
    else:
        y_pred = model.predict(X_test)
    
    # Inverse transform the predictions and actual values
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    print(f"\n=== MODEL EVALUATION ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'y_test': y_test_inv,
        'y_pred': y_pred_inv
    }


def plot_results(evaluation_results, args):
    """Plot the model predictions against actual values."""
    if args.no_plot:
        return
    
    print("📈 Generating visualizations...")
    
    # Create directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(evaluation_results['y_test'], label='Actual', color='black', linewidth=2)
    plt.plot(evaluation_results['y_pred'], label=f"Predicted (RMSE: {evaluation_results['rmse']:.4f})")
    plt.title(f"{args.model_type.capitalize()} Model Predictions for {args.symbol}")
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plot_path = f"plots/{args.model_type}_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path)
    print(f"✅ Plot saved to {plot_path}")
    
    plt.show()


def main():
    """Main function to run the market analysis system."""
    args = parse_arguments()
    
    print(f"🚀 Starting market analysis for {args.symbol}...")
    
    try:
        # Fetch data
        data = fetch_data(args)
        
        # Process data
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = process_data(data, args)
        
        # Get input shape for neural network models
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Create or load model
        if args.load_model:
            print(f"📂 Loading model from {args.load_model}...")
            # Placeholder for model loading
            # In a real implementation, we would load the model here
            model = create_model(args, input_shape)
            model.load(args.load_model)
        else:
            model = create_model(args, input_shape)
        
        # Train model if not skipped
        if not args.no_train and not args.load_model:
            model = train_model(model, X_train, y_train, X_val, y_val, args)
            
            # Save model if requested
            if args.save_model:
                print(f"💾 Saving model to {args.save_model}...")
                model.save(args.save_model)
        
        # Evaluate model
        evaluation_results = evaluate_model(model, X_test, y_test, scaler)
        
        # Plot results
        plot_results(evaluation_results, args)
        
        print("✅ Market analysis completed!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()