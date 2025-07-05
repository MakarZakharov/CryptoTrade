"""
Example script demonstrating trading simulation with different models.
"""

import os
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the CRYPTO_BOT directory to the Python path
crypto_bot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.insert(0, crypto_bot_dir)


def run_model_with_trading(symbol, model_type, initial_investment=10000.0):
    """
    Run a model with trading simulation and return the performance metrics.
    
    Args:
        symbol: Trading symbol (e.g., BTCUSDC)
        model_type: Model type (lstm, gru, transformer, xgboost)
        initial_investment: Initial investment amount
        
    Returns:
        Dictionary containing performance metrics
    """
    print(f"\n=== Running {model_type.upper()} model for {symbol} with ${initial_investment:.2f} investment ===")
    
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    model_path = os.path.join(project_dir, 'models', f"{model_type}_{symbol.lower()}")
    
    # Train the model if it doesn't exist
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Training a new model...")
        train_cmd = [
            "python", os.path.join(project_dir, "main.py"),
            "--symbol", symbol,
            "--model_type", model_type,
            "--save_model", model_path,
            "--run_trading",
            "--initial_investment", str(initial_investment)
        ]
        subprocess.run(train_cmd)
    
    # Run prediction with trading simulation
    predict_cmd = [
        "python", os.path.join(project_dir, "predict.py"),
        "--symbol", symbol,
        "--model_path", model_path,
        "--model_type", model_type,
        "--run_trading",
        "--initial_investment", str(initial_investment),
        "--forecast_days", "14"
    ]
    subprocess.run(predict_cmd)
    
    # For demonstration purposes, we'll return some dummy metrics
    # In a real implementation, you would parse the output or save/load the metrics
    return {
        'model_type': model_type,
        'symbol': symbol,
        'initial_investment': initial_investment,
        'final_equity': initial_investment * (1 + (0.05 if model_type == 'lstm' else 0.03)),
        'percentage_return': 5.0 if model_type == 'lstm' else 3.0
    }


def compare_models(symbol, models, initial_investment=10000.0):
    """
    Compare multiple models for trading performance.
    
    Args:
        symbol: Trading symbol (e.g., BTCUSDC)
        models: List of model types to compare
        initial_investment: Initial investment amount
    """
    results = []
    
    for model_type in models:
        performance = run_model_with_trading(symbol, model_type, initial_investment)
        results.append(performance)
    
    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    
    # Print comparison table
    print("\n=== MODEL COMPARISON ===")
    print(df[['model_type', 'symbol', 'initial_investment', 'final_equity', 'percentage_return']])
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df['model_type'], df['percentage_return'])
    plt.title(f"Trading Performance Comparison for {symbol}")
    plt.xlabel("Model Type")
    plt.ylabel("Return (%)")
    plt.grid(axis='y')
    
    # Save the plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(os.path.dirname(script_dir), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"model_comparison_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")
    
    plt.show()


def main():
    """Main function to run the example."""
    # Define parameters
    symbol = "BTCUSDC"
    models = ["lstm", "gru", "xgboost"]
    initial_investment = 10000.0
    
    # Compare models
    compare_models(symbol, models, initial_investment)


if __name__ == "__main__":
    main()