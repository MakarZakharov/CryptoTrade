"""
Main entry point for the market analysis system.
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add the CRYPTO_BOT directory to the Python path
crypto_bot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, crypto_bot_dir)

# Now we can import using the full path from the CRYPTO_BOT directory
from ai.ML1.market_analysis.data.fetchers.binance_fetcher import BinanceFetcher
from ai.ML1.market_analysis.data.fetchers.csv_fetcher import CSVFetcher
from ai.ML1.market_analysis.data.processors.price_processor import PriceProcessor
from ai.ML1.market_analysis.data.features.technical_indicators import TechnicalIndicators
from ai.ML1.market_analysis.data.features.feature_selector import FeatureSelector
from ai.ML1.market_analysis.models.model_factory import ModelFactory
from ai.ML1.market_analysis.trading import SimpleStrategy


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cryptocurrency Market Analysis')
    
    # Data parameters
    parser.add_argument('--symbol', type=str, default='BTCUSDC',
                        help='Trading symbol (default: BTCUSDC)')
    parser.add_argument('--timeframe', type=str, default='1d',
                        help='Timeframe interval (default: 1d)')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                        help='Start date for data (default: 2020-01-01)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for data (default: None/current)')
    parser.add_argument('--data_source', type=str, default='csv', choices=['binance', 'csv'],
                        help='Data source (default: csv)')
    parser.add_argument('--csv_path', type=str, default='/home/newuser/CRYPTO_BOT/data/binance/BTCUSDC/1d/2018_12_15-now.csv',
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
    
    # Trading simulation parameters
    parser.add_argument('--run_trading', action='store_true',
                        help='Run trading simulation on test data')
    parser.add_argument('--initial_investment', type=float, default=10000.0,
                        help='Initial investment amount for trading simulation (default: 10000.0)')
    parser.add_argument('--transaction_fee', type=float, default=0.001,
                        help='Transaction fee as a percentage for trading simulation (default: 0.001 = 0.1%)')
    parser.add_argument('--initial_balance', type=float, default=10000,
                        help='Initial balance for backtesting (default: 10000)')
    
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
    
    # Add technical indicators first
    indicators = TechnicalIndicators()
    df = indicators.add_all_indicators(data)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Select features
    selector = FeatureSelector(target_column='close')
    selected_features_df, selected_features = selector.select_features(df)
    
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
    from ai.ML1.market_analysis.models.xgboost_model import XGBoostModel
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
    from ai.ML1.market_analysis.models.xgboost_model import XGBoostModel
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
    
    # Create directory for plots relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
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
    plot_path = os.path.join(plots_dir, f"{args.model_type}_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    print(f"✅ Plot saved to {plot_path}")
    
    plt.show()


def run_trading_simulation(evaluation_results, args) -> Dict[str, Any]:
    """
    Run trading simulation using the model predictions.
    
    Args:
        evaluation_results: Dictionary containing evaluation results
        args: Command line arguments
        
    Returns:
        Dictionary containing trading performance metrics
    """
    if not args.run_trading:
        return None
    
    print("\n🤖 Running trading simulation...")
    
    # Виправлено: використовуємо initial_balance і threshold
    strategy = SimpleStrategy(
        initial_balance=args.initial_balance,
        transaction_fee=args.transaction_fee,
        threshold=getattr(args, 'threshold', 0.0)
    )
    
    actual_prices = evaluation_results['y_test'].flatten()
    predicted_prices = evaluation_results['y_pred'].flatten()
    
    performance = strategy.backtest(actual_prices, predicted_prices)
    
    # Print performance metrics
    print("\n=== TRADING PERFORMANCE ===")
    print(f"Initial Balance: ${performance['initial_investment']:.2f}")
    print(f"Final Balance:   ${performance['final_equity']:.2f}")
    print(f"Absolute Return: ${performance['absolute_return']:.2f}")
    print(f"Percentage Return: {performance['percentage_return']:.2f}%")
    print(f"Number of Trades: {performance['num_trades']}")
    if 'win_rate' in performance:
        print(f"Win Rate: {performance['win_rate']:.2f}%")
    
    # Plot equity curve if not disabled
    if not args.no_plot:
        plot_path = strategy.plot_equity_curve(args.symbol, args.model_type)
        if plot_path:
            print(f"✅ Trading performance plot saved to {plot_path}")
    
    return performance


def walk_forward_training(X, y, args, window_size=500, step_size=100):
    """
    Walk-forward training: навчає модель на відрізках історії та тестує на наступних.
    """
    from sklearn.preprocessing import MinMaxScaler
    n_samples = X.shape[0]
    results = []
    for start in range(0, n_samples - window_size, step_size):
        end = start + window_size
        X_train, y_train = X[start:end], y[start:end]
        X_test, y_test = X[end:end+step_size], y[end:end+step_size]
        if len(X_test) == 0:
            break
        input_shape = (X_train.shape[1], X_train.shape[2])
        scaler = MinMaxScaler()
        y_train_scaled = scaler.fit_transform(y_train)
        y_test_scaled = scaler.transform(y_test)
        model = create_model(args, input_shape)
        model = train_model(model, X_train, y_train_scaled, X_test, y_test_scaled, args)
        eval_result = evaluate_model(model, X_test, y_test_scaled, scaler)
        trading_performance = run_trading_simulation(eval_result, args)
        if trading_performance:
            initial = trading_performance['initial_investment']
            final = trading_performance['final_equity']
            percent = (final / initial) * 100 if initial else 0
            # Просадка (max drawdown)
            equity_curve = trading_performance.get('equity_curve')
            if equity_curve is not None:
                peak = equity_curve[0]
                max_drawdown = 0
                for x in equity_curve:
                    if x > peak:
                        peak = x
                    drawdown = (peak - x) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                max_drawdown_pct = max_drawdown * 100
            else:
                max_drawdown_pct = None
            print(f"Walk {start//step_size+1}: Initial Balance: ${initial:.2f}, Final Balance: ${final:.2f} ({percent:.2f}%)"
                  + (f", Max Drawdown: {max_drawdown_pct:.2f}%" if max_drawdown_pct is not None else ""))
        print(f"Walk {start//step_size+1}: RMSE={eval_result['rmse']:.2f}, MAE={eval_result['mae']:.2f}")
        results.append({'eval': eval_result, 'trading': trading_performance})
    return results


def print_final_balance(walk_results):
    """Вивести кінцевий баланс та статистику останнього кроку walk-forward."""
    for i, res in enumerate(reversed(walk_results), 1):
        t = res['trading']
        if t is None:
            print(f"⚠️ Крок {len(walk_results)-i+1}: trading_performance = None")
        elif 'final_equity' not in t:
            print(f"⚠️ Крок {len(walk_results)-i+1}: trading_performance без final_equity")
        else:
            initial = t['initial_investment']
            final = t['final_equity']
            percent = (final / initial) * 100 if initial else 0
            equity_curve = t.get('equity_curve')
            if equity_curve is not None:
                peak = equity_curve[0]
                max_drawdown = 0
                for x in equity_curve:
                    if x > peak:
                        peak = x
                    drawdown = (peak - x) / peak
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                max_drawdown_pct = max_drawdown * 100
            else:
                max_drawdown_pct = None
            print(f"\n🏁 Кінцевий баланс після walk-forward: ${final:.2f} ({percent:.2f}%)"
                  + (f", Max Drawdown: {max_drawdown_pct:.2f}%" if max_drawdown_pct is not None else ""))
            return
    print("\n❗️ Кінцевий баланс не знайдено.")


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
        
        # Run trading simulation if requested
        if args.run_trading:
            trading_performance = run_trading_simulation(evaluation_results, args)
        
        print("✅ Market analysis completed!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    args = parse_arguments()
    args.initial_balance = 10000  # Гарантуємо початковий баланс 10k$
    data = fetch_data(args)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = process_data(data, args)
    X = np.concatenate([X_train, X_val, X_test], axis=0)
    y = np.concatenate([y_train, y_val, y_test], axis=0)
    walk_results = walk_forward_training(X, y, args, window_size=500, step_size=100)
    print_final_balance(walk_results)
    # main() # <- закоментуйте або видаліть цей рядок, якщо хочете запускати лише walk-forward