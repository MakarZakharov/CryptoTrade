"""
Simple trading strategy implementation for market analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import os
from datetime import datetime

class SimpleStrategy:
    """
    Simple trading strategy based on price predictions.
    
    This strategy buys when the predicted price is higher than the current price
    and sells when the predicted price is lower than the current price.
    """
    
    def __init__(self, initial_balance: float = 10000.0, transaction_fee: float = 0.001, threshold: float = 0.0):
        """
        Initialize the simple trading strategy.

        Args:
            initial_balance: Initial balance for backtesting (default: 10000)
            transaction_fee: Transaction fee as a percentage (default: 0.1%)
            threshold: Minimal price difference to trigger trade (default: 0.0)
        """
        self.initial_investment = initial_balance
        self.transaction_fee = transaction_fee
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset the strategy state."""
        self.balance = self.initial_investment
        self.asset_amount = 0.0
        self.in_position = False
        self.trades = []
        self.equity_history = []
    
    def calculate_fee(self, amount: float) -> float:
        """
        Calculate the transaction fee.
        
        Args:
            amount: Transaction amount
            
        Returns:
            Fee amount
        """
        return amount * self.transaction_fee
    
    def buy(self, price: float, timestamp=None):
        """
        Execute a buy order.
        
        Args:
            price: Current price
            timestamp: Optional timestamp for the trade
        """
        if self.in_position:
            return
        
        # Calculate the amount of asset to buy (use all available balance)
        amount_to_spend = self.balance
        fee = self.calculate_fee(amount_to_spend)
        amount_to_spend -= fee
        
        # Calculate the amount of asset bought
        self.asset_amount = amount_to_spend / price
        
        # Update balance
        self.balance = 0
        
        # Update position status
        self.in_position = True
        
        # Record the trade
        self.trades.append({
            'timestamp': timestamp,
            'action': 'buy',
            'price': price,
            'amount': self.asset_amount,
            'fee': fee,
            'balance': self.balance,
            'asset_value': self.asset_amount * price,
            'total_equity': self.balance + (self.asset_amount * price)
        })
    
    def sell(self, price: float, timestamp=None):
        """
        Execute a sell order.
        
        Args:
            price: Current price
            timestamp: Optional timestamp for the trade
        """
        if not self.in_position:
            return
        
        # Calculate the amount received from selling
        amount_received = self.asset_amount * price
        fee = self.calculate_fee(amount_received)
        amount_received -= fee
        
        # Update balance and asset amount
        self.balance = amount_received
        self.asset_amount = 0
        
        # Update position status
        self.in_position = False
        
        # Record the trade
        self.trades.append({
            'timestamp': timestamp,
            'action': 'sell',
            'price': price,
            'amount': self.asset_amount,
            'fee': fee,
            'balance': self.balance,
            'asset_value': 0,
            'total_equity': self.balance
        })
    
    def update_equity(self, price: float, timestamp=None):
        """
        Update the equity history.
        
        Args:
            price: Current price
            timestamp: Optional timestamp for the update
        """
        equity = self.balance + (self.asset_amount * price)
        self.equity_history.append({
            'timestamp': timestamp,
            'price': price,
            'balance': self.balance,
            'asset_amount': self.asset_amount,
            'asset_value': self.asset_amount * price,
            'total_equity': equity
        })
    
    def backtest(self, prices: np.ndarray, predictions: np.ndarray, timestamps=None) -> Dict:
        """
        Backtest the strategy on historical data.
        
        Args:
            prices: Array of historical prices
            predictions: Array of price predictions (shifted by 1 period)
            timestamps: Optional array of timestamps
            
        Returns:
            Dictionary containing backtest results
        """
        # Reset the strategy state
        self.reset()
        
        # Ensure predictions are aligned with prices (prediction at t is for price at t+1)
        if len(predictions) < len(prices) - 1:
            raise ValueError("Not enough predictions for the given prices")
        
        # Use only the relevant part of the predictions
        predictions = predictions[:len(prices)-1]
        
        # Iterate through the prices and predictions
        for i in range(len(prices) - 1):
            current_price = prices[i]
            next_price = prices[i+1]
            predicted_next_price = predictions[i][0] if predictions[i].ndim > 0 else predictions[i]
            timestamp = timestamps[i] if timestamps is not None else i
            
            # Update equity history
            self.update_equity(current_price, timestamp)
            
            # Покращена логіка для частіших трейдів
            if predicted_next_price > current_price + self.threshold and not self.in_position:
                # Predicted price is higher, buy
                self.buy(current_price, timestamp)
            elif predicted_next_price < current_price - self.threshold and self.in_position:
                # Predicted price is lower, sell
                self.sell(current_price, timestamp)
        
        # Final update with the last price
        last_price = prices[-1]
        last_timestamp = timestamps[-1] if timestamps is not None else len(prices) - 1
        self.update_equity(last_price, last_timestamp)
        
        # If still in position, sell at the last price
        if self.in_position:
            self.sell(last_price, last_timestamp)
        
        # Calculate performance metrics
        return self.calculate_performance()
    
    def calculate_performance(self) -> Dict:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.equity_history:
            return {
                'initial_investment': self.initial_investment,
                'final_equity': self.initial_investment,
                'absolute_return': 0,
                'percentage_return': 0,
                'num_trades': 0
            }
        
        # Calculate returns
        initial_equity = self.initial_investment
        final_equity = self.equity_history[-1]['total_equity']
        absolute_return = final_equity - initial_equity
        percentage_return = (absolute_return / initial_equity) * 100
        
        # Calculate number of trades
        num_trades = len(self.trades)
        
        # Calculate win rate
        if num_trades > 0:
            profitable_trades = 0
            for i in range(0, num_trades, 2):
                if i + 1 < num_trades:  # Ensure we have a complete buy-sell pair
                    buy_trade = self.trades[i]
                    sell_trade = self.trades[i+1]
                    if sell_trade['balance'] > buy_trade['asset_value']:
                        profitable_trades += 1
            
            win_rate = (profitable_trades / (num_trades // 2)) * 100 if num_trades > 1 else 0
        else:
            win_rate = 0
        
        return {
            'initial_investment': initial_equity,
            'final_equity': final_equity,
            'absolute_return': absolute_return,
            'percentage_return': percentage_return,
            'num_trades': num_trades,
            'win_rate': win_rate
        }
    
    def plot_equity_curve(self, symbol: str = None, model_type: str = None) -> str:
        """
        Plot the equity curve.
        
        Args:
            symbol: Trading symbol
            model_type: Model type
            
        Returns:
            Path to the saved plot
        """
        if not self.equity_history:
            return None
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Extract data
        timestamps = [entry['timestamp'] for entry in self.equity_history]
        equity = [entry['total_equity'] for entry in self.equity_history]
        
        # Plot equity curve
        plt.plot(timestamps, equity, label='Total Equity')
        
        # Add horizontal line for initial investment
        plt.axhline(y=self.initial_investment, color='r', linestyle='--', label='Initial Investment')
        
        # Add markers for trades
        for trade in self.trades:
            if trade['action'] == 'buy':
                plt.scatter(trade['timestamp'], trade['total_equity'], color='g', marker='^', s=100)
            else:  # sell
                plt.scatter(trade['timestamp'], trade['total_equity'], color='r', marker='v', s=100)
        
        # Add title and labels
        title = f"Trading Performance"
        if symbol:
            title += f" - {symbol}"
        if model_type:
            title += f" ({model_type})"
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.legend()
        
        # Create directory for plots relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(script_dir))), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Save the plot
        plot_name = f"trading_performance"
        if symbol:
            plot_name += f"_{symbol}"
        if model_type:
            plot_name += f"_{model_type}"
        plot_name += f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plot_path = os.path.join(plots_dir, plot_name)
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path