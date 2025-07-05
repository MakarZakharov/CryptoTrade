"""
Script for making predictions using a trained model.
"""

import argparse
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, Any

# Add the CRYPTO_BOT directory to the Python path
crypto_bot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, crypto_bot_dir)

from ai.ML1.market_analysis.data.fetchers.binance_fetcher import BinanceFetcher
from ai.ML1.market_analysis.data.fetchers.csv_fetcher import CSVFetcher
from ai.ML1.market_analysis.data.features.technical_indicators import TechnicalIndicators
from ai.ML1.market_analysis.data.features.feature_selector import FeatureSelector
from ai.ML1.market_analysis.models.model_factory import ModelFactory
from ai.ML1.market_analysis.trading import SimpleStrategy


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cryptocurrency Price Prediction')
    
    # Data parameters
    parser.add_argument('--symbol', type=str, default='BTCUSDC',
                        help='Trading symbol (default: BTCUSDC)')
    parser.add_argument('--timeframe', type=str, default='1d',
                        help='Timeframe interval (default: 1d)')
    parser.add_argument('--days', type=int, default=60,
                        help='Number of days of historical data to fetch (default: 60)')
    parser.add_argument('--data_source', type=str, default='csv', choices=['binance', 'csv'],
                        help='Data source (default: csv)')
    parser.add_argument('--csv_path', type=str, default='/home/newuser/CRYPTO_BOT/data/binance/BTCUSDC/1d/2018_01_01-now.csv',
                        help='Path to CSV file if data_source is csv')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='lstm',
                        choices=['lstm', 'gru', 'transformer', 'xgboost', 'stacking', 'voting'],
                        help='Model type (default: lstm)')
    
    # Prediction parameters
    parser.add_argument('--forecast_days', type=int, default=7,
                        help='Number of days to forecast (default: 7)')
    
    # Output parameters
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save the predictions (default: None)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip plotting results')
    
    # Trading simulation parameters
    parser.add_argument('--run_trading', action='store_true',
                        help='Run trading simulation on predicted data')
    parser.add_argument('--initial_investment', type=float, default=10000.0,
                        help='Initial investment amount for trading simulation (default: 10000.0)')
    parser.add_argument('--transaction_fee', type=float, default=0.001,
                        help='Transaction fee as a percentage for trading simulation (default: 0.001 = 0.1%)')
    parser.add_argument('--last_price', type=float, default=None,
                        help='Last known price (if not provided, will use the last price from historical data)')
    
    return parser.parse_args()


def fetch_data(args):
    """Fetch recent data for prediction."""
    print(f"📊 Fetching recent {args.symbol} data from {args.data_source}...")
    
    # Calculate start date based on window size and forecast days
    start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    
    if args.data_source == 'binance':
        fetcher = BinanceFetcher(symbol=args.symbol, interval=args.timeframe)
        data = fetcher.fetch_data(
            start_date=start_date,
            end_date=None  # Current time
        )
    elif args.data_source == 'csv':
        if args.csv_path is None:
            raise ValueError("CSV path must be provided when data_source is 'csv'")
        fetcher = CSVFetcher(symbol=args.symbol, interval=args.timeframe)
        data = fetcher.fetch_data(
            start_date=start_date,
            end_date=None,
            path=args.csv_path
        )
    else:
        raise ValueError(f"Unsupported data source: {args.data_source}")
    
    if data is None or len(data) == 0:
        raise ValueError("Failed to fetch data")
    
    print(f"✅ Fetched {len(data)} data points")
    return data


def prepare_data(data, args):
    """Prepare data for prediction."""
    print("🔄 Preparing data for prediction...")
    
    # Add technical indicators
    indicators = TechnicalIndicators()
    df = indicators.add_all_indicators(data)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Select features (use the same feature selection as in training)
    selector = FeatureSelector(target_column='close')
    selected_df, selected_features = selector.select_features(df)
    
    # Extract features
    features = df[selected_features].values
    
    # Scale the features (note: in a real implementation, you would use the same scaler as in training)
    from sklearn.preprocessing import MinMaxScaler
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    # Fit the target scaler on the 'close' prices to be able to inverse transform later
    target_scaler.fit(df['close'].values.reshape(-1, 1))
    
    features_scaled = feature_scaler.fit_transform(features)
    
    # Create sequences for the model
    X = []
    for i in range(len(features_scaled) - args.days + 1):
        X.append(features_scaled[i:i+args.days])
    
    X = np.array(X)
    
    print(f"✅ Data prepared. Shape: {X.shape}")
    
    return X, target_scaler, df.index[-1]


def load_model(args):
    """Load the trained model."""
    print(f"📂 Loading model from {args.model_path}...")
    
    # Create a dummy model of the same type
    if args.model_type in ['lstm', 'gru', 'transformer']:
        # For neural network models, we need to specify the input shape
        # This is a placeholder, the actual shape will be determined when loading
        model = ModelFactory.create_model(
            args.model_type,
            input_shape=(60, 13),  # Placeholder shape
            units=50,
            dropout=0.2,
            learning_rate=0.001
        )
    elif args.model_type == 'xgboost':
        model = ModelFactory.create_model(
            'xgboost',
            learning_rate=0.001
        )
    else:
        raise ValueError(f"Unsupported model type for prediction: {args.model_type}")
    
    # Load the trained weights
    model.load(args.model_path)
    
    print("✅ Model loaded successfully")
    return model


def make_predictions(model, X, scaler, last_date, args):
    """Make predictions using the loaded model."""
    print("🔮 Making predictions...")
    
    # Get the most recent sequence for prediction
    latest_sequence = X[-1:]
    
    # For XGBoost, we need to flatten the input
    from ai.ML1.market_analysis.models.xgboost_model import XGBoostModel
    if isinstance(model, XGBoostModel):
        latest_sequence = latest_sequence.reshape(latest_sequence.shape[0], -1)
    
    # Make the initial prediction
    predictions = []
    current_sequence = latest_sequence.copy()
    
    # Generate predictions for the specified number of days
    for i in range(args.forecast_days):
        # Make a prediction
        pred = model.predict(current_sequence)
        
        # Store the prediction
        predictions.append(pred[0][0])
        
        # For sequential models, update the sequence for the next prediction
        if not isinstance(model, XGBoostModel):
            # Shift the sequence and add the new prediction
            # This is a simplified approach; in a real implementation, you would need to
            # update all features, not just the target
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = pred[0][0]
    
    # Convert predictions to numpy array
    predictions = np.array(predictions).reshape(-1, 1)
    
    # Inverse transform to get actual prices
    predictions_actual = scaler.inverse_transform(predictions)
    
    # Generate dates for the predictions
    prediction_dates = []
    current_date = last_date
    for i in range(args.forecast_days):
        if args.timeframe == '1d':
            current_date = current_date + timedelta(days=1)
        elif args.timeframe == '4h':
            current_date = current_date + timedelta(hours=4)
        elif args.timeframe == '1h':
            current_date = current_date + timedelta(hours=1)
        else:
            # Default to daily
            current_date = current_date + timedelta(days=1)
        prediction_dates.append(current_date)
    
    # Create a DataFrame with the predictions
    predictions_df = pd.DataFrame({
        'date': prediction_dates,
        'predicted_price': predictions_actual.flatten()
    })
    predictions_df.set_index('date', inplace=True)
    
    print("✅ Predictions completed")
    return predictions_df


def plot_predictions(predictions_df, args):
    """Plot the predictions."""
    if args.no_plot:
        return
    
    print("📈 Generating prediction visualization...")
    
    # Create directory for plots relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(predictions_df.index, predictions_df['predicted_price'], marker='o', linestyle='-', label='Predicted Price')
    plt.title(f"{args.model_type.capitalize()} Model Predictions for {args.symbol}")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, f"prediction_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    print(f"✅ Plot saved to {plot_path}")
    
    plt.show()


def save_predictions(predictions_df, args):
    """Save the predictions to a file."""
    if args.output_file:
        print(f"💾 Saving predictions to {args.output_file}...")
        predictions_df.to_csv(args.output_file)
        print("✅ Predictions saved")


def simulate_trading(predictions_df, args) -> Dict[str, Any]:
    """
    Simulate trading based on predicted prices.
    
    Args:
        predictions_df: DataFrame containing predicted prices
        args: Command line arguments
        
    Returns:
        Dictionary containing trading performance metrics
    """
    if not args.run_trading:
        return None
    
    print("\n🤖 Running trading simulation on predicted prices...")
    
    # Create trading strategy
    strategy = SimpleStrategy(
        initial_investment=args.initial_investment,
        transaction_fee=args.transaction_fee
    )
    
    # Get the last known price (either provided or from historical data)
    last_price = args.last_price
    if last_price is None:
        print("Warning: No last price provided. Using the first predicted price as reference.")
        last_price = predictions_df['predicted_price'].iloc[0] * 0.99  # Slightly lower to trigger a buy
    
    # Create a list of prices for simulation
    # Start with the last known price, then add all predicted prices
    prices = [last_price] + predictions_df['predicted_price'].tolist()
    
    # Create a list of "predictions" (shifted by 1)
    # For the first price, use the first prediction to make a decision
    predictions = predictions_df['predicted_price'].tolist() + [predictions_df['predicted_price'].iloc[-1]]
    
    # Run backtest
    performance = strategy.backtest(np.array(prices), np.array(predictions))
    
    # Print performance metrics
    print("\n=== POTENTIAL TRADING PERFORMANCE ===")
    print(f"Initial Investment: ${performance['initial_investment']:.2f}")
    print(f"Final Equity: ${performance['final_equity']:.2f}")
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


def main():
    """Main function to run the prediction."""
    args = parse_arguments()
    
    print(f"🚀 Starting price prediction for {args.symbol}...")
    
    try:
        # Fetch recent data
        data = fetch_data(args)
        
        # Prepare data for prediction
        X, scaler, last_date = prepare_data(data, args)
        
        # Load the trained model
        model = load_model(args)
        
        # Make predictions
        predictions_df = make_predictions(model, X, scaler, last_date, args)
        
        # Display predictions
        print("\n=== PRICE PREDICTIONS ===")
        print(predictions_df)
        
        # Plot predictions
        plot_predictions(predictions_df, args)
        
        # Save predictions
        save_predictions(predictions_df, args)
        
        # Run trading simulation if requested
        if args.run_trading:
            trading_performance = simulate_trading(predictions_df, args)
        
        print("✅ Prediction completed!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()