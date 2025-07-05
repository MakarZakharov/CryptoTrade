"""
Test script for evaluating a trained model and running trading simulation with initial and final balance output.
"""

import argparse
from datetime import datetime
import os
import sys
import numpy as np

# Add the CRYPTO_BOT directory to the Python path
crypto_bot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, crypto_bot_dir)

from ai.ML1.market_analysis.main import fetch_data, process_data, evaluate_model, plot_results
from ai.ML1.market_analysis.models.model_factory import ModelFactory
from ai.ML1.market_analysis.trading import SimpleStrategy


def parse_test_arguments():
    parser = argparse.ArgumentParser(description='Test trained model and run trading simulation')
    parser.add_argument('--symbol', type=str, default='BTCUSDC', help='Trading symbol (default: BTCUSDC)')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe interval (default: 1d)')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to CSV file with data')
    parser.add_argument('--model_type', type=str, default='lstm', help='Model type (default: lstm)')
    parser.add_argument('--load_model', type=str, required=True, help='Path to trained model (.h5)')
    parser.add_argument('--initial_balance', type=float, default=10000, help='Initial balance for backtesting (default: 10000)')
    parser.add_argument('--transaction_fee', type=float, default=0.001, help='Transaction fee (default: 0.001)')
    parser.add_argument('--no_plot', action='store_true', help='Skip plotting results')
    return parser.parse_args()


def main():
    args = parse_test_arguments()
    print(f"\n🚀 Testing model {args.load_model} on {args.symbol}...")

    # Prepare args for fetch_data/process_data
    class DummyArgs:
        pass
    dummy_args = DummyArgs()
    dummy_args.symbol = args.symbol
    dummy_args.timeframe = args.timeframe
    dummy_args.data_source = 'csv'
    dummy_args.csv_path = args.csv_path
    dummy_args.start_date = '2020-01-01'
    dummy_args.end_date = None
    dummy_args.window_size = 60
    dummy_args.train_split = 0.7
    dummy_args.val_split = 0.15

    # Fetch and process data
    data = fetch_data(dummy_args)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = process_data(data, dummy_args)
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Load model
    model = ModelFactory.create_model(args.model_type, input_shape=input_shape)
    model.load(args.load_model)

    # Evaluate model
    evaluation_results = evaluate_model(model, X_test, y_test, scaler)
    plot_results(evaluation_results, args)

    # Trading simulation
    print("\n🤖 Running trading simulation with initial balance...")
    strategy = SimpleStrategy(initial_balance=args.initial_balance, transaction_fee=args.transaction_fee)
    actual_prices = evaluation_results['y_test'].flatten()
    predicted_prices = evaluation_results['y_pred'].flatten()
    performance = strategy.backtest(actual_prices, predicted_prices)

    print(f"\n=== BALANCE REPORT ===")
    print(f"Initial balance: ${args.initial_balance:.2f}")
    print(f"Final balance:   ${performance['final_equity']:.2f}")
    print(f"Absolute return: ${performance['absolute_return']:.2f}")
    print(f"Percentage return: {performance['percentage_return']:.2f}%")
    print(f"Number of trades: {performance['num_trades']}")
    if 'win_rate' in performance:
        print(f"Win Rate: {performance['win_rate']:.2f}%")

    if not args.no_plot:
        plot_path = strategy.plot_equity_curve(args.symbol, args.model_type)
        if plot_path:
            print(f"✅ Trading performance plot saved to {plot_path}")

    print("\n✅ Model test and trading simulation completed!")


if __name__ == "__main__":
    main()
