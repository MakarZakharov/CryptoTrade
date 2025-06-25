import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.analyzers as btanalyzers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import os
import joblib
from datetime import datetime


class MLStrategy(bt.Strategy):
    """Fully autonomous backtrader strategy using ML predictions"""
    
    params = (
        ('predictions', None),  # ML predictions array
        ('prediction_dates', None),  # Dates for predictions
        ('prediction_probs', None),  # Prediction probabilities for confidence
        ('autonomous', True),  # Fully autonomous mode
    )
    
    def __init__(self):
        self.predictions = self.params.predictions
        self.prediction_dates = self.params.prediction_dates
        self.prediction_probs = self.params.prediction_probs
        self.order = None
        self.orders = []  # Track multiple orders
        self.position_tracker = {}  # Track multiple positions
        
        # Strategy state
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.recent_returns = []
        self.max_positions = 5  # Dynamic, can be adjusted
        
        # Create prediction lookup
        self.prediction_dict = {}
        self.confidence_dict = {}
        if self.predictions is not None and self.prediction_dates is not None:
            for date, pred in zip(self.prediction_dates, self.predictions):
                self.prediction_dict[date.date()] = pred
                
        # Market indicators
        self.sma20 = bt.indicators.SMA(self.datas[0].close, period=20)
        self.sma50 = bt.indicators.SMA(self.datas[0].close, period=50)
        self.rsi = bt.indicators.RSI(self.datas[0].close, period=14)
        self.atr = bt.indicators.ATR(self.datas[0], period=14)
        self.volatility = bt.indicators.StdDev(self.datas[0].close, period=20)
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, '
                        f'Comm: {order.executed.comm:.2f}')
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, '
                        f'Cost: {order.executed.value:.2f}, '
                        f'Comm: {order.executed.comm:.2f}')
        
        self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        
        self.log(f'TRADE PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
        
        # Track consecutive wins/losses for autonomous risk management
        if trade.pnlcomm > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
        
        # Store recent returns for analysis
        self.recent_returns.append(trade.pnlcomm)
        if len(self.recent_returns) > 20:
            self.recent_returns.pop(0)
    
    def next(self):
        # Check if we have a pending order
        if self.order:
            return
        
        # Get current date
        current_date = self.datas[0].datetime.date(0)
        
        # Get prediction for current date
        prediction = self.prediction_dict.get(current_date, 'Hold')
        
        # Autonomous decision making
        if not self.position:  # No position
            if prediction == 'Buy':
                # Fully autonomous position sizing based on market conditions
                current_portfolio_value = self.broker.getvalue()
                available_cash = self.broker.getcash()
                
                # Use all available capital for each trade
                position_value = available_cash * 0.99  # Use 99% to account for rounding/slippage
                size = position_value / self.datas[0].close[0]
                
                # Execute trade if size is meaningful
                if size > 0.0001:  # Minimum BTC amount
                    self.order = self.buy(size=size)
                    self.log(f'BUY CREATE, Price: {self.datas[0].close[0]:.2f}, Size: {size:.6f}')
        
        else:  # We have a position
            if prediction == 'Sell':
                self.order = self.close()
                self.log(f'SELL CREATE, Price: {self.datas[0].close[0]:.2f}')
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')


class PandasData(btfeeds.PandasData):
    """Custom Pandas data feed for backtrader"""
    params = (
        ('datetime', 'timestamp'),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('openinterest', None),
    )


class MLBacktraderBacktest:
    """Backtrader-based backtesting system for ML strategy"""
    
    def __init__(self, initial_capital: float = 10000.0, commission: float = 0.001):
        """
        Initialize backtrader backtester
        
        Args:
            initial_capital: Starting capital
            commission: Trading commission (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.cerebro = None
        self.results = None
    
    def run_backtest(self, data: pd.DataFrame, predictions: np.ndarray) -> Dict[str, Any]:
        """
        Run backtest using backtrader
        
        Args:
            data: DataFrame with OHLCV data
            predictions: Array of ML predictions
            
        Returns:
            Dictionary with backtest results
        """
        # Initialize Cerebro engine
        self.cerebro = bt.Cerebro()
        
        # Set initial capital
        self.cerebro.broker.setcash(self.initial_capital)
        
        # Set commission
        self.cerebro.broker.setcommission(commission=self.commission)
        
        # Prepare data
        data_copy = data.copy()
        data_copy.reset_index(inplace=True)
        
        # Create data feed
        data_feed = PandasData(dataname=data_copy)
        self.cerebro.adddata(data_feed)
        
        # Add strategy with predictions
        self.cerebro.addstrategy(
            MLStrategy,
            predictions=predictions,
            prediction_dates=data_copy['timestamp']
        )
        
        # Add analyzers
        self.cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(btanalyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(btanalyzers.Returns, _name='returns')
        self.cerebro.addanalyzer(btanalyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(btanalyzers.SQN, _name='sqn')
        
        # Run backtest
        print('Starting Portfolio Value: %.2f' % self.cerebro.broker.getvalue())
        self.results = self.cerebro.run()
        print('Final Portfolio Value: %.2f' % self.cerebro.broker.getvalue())
        
        # Extract results
        strat = self.results[0]
        
        # Calculate metrics
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Get analyzer results
        sharpe = strat.analyzers.sharpe.get_analysis()
        drawdown = strat.analyzers.drawdown.get_analysis()
        returns = strat.analyzers.returns.get_analysis()
        trades = strat.analyzers.trades.get_analysis()
        sqn = strat.analyzers.sqn.get_analysis()
        
        # Calculate buy and hold return
        buy_hold_return = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        
        # Compile results
        metrics = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return * 100,
            'sharpe_ratio': sharpe.get('sharperatio', 0) if sharpe else 0,
            'max_drawdown': drawdown.get('max', {}).get('drawdown', 0) if drawdown else 0,
            'total_trades': trades.get('total', {}).get('total', 0) if trades else 0,
            'won_trades': trades.get('won', {}).get('total', 0) if trades else 0,
            'lost_trades': trades.get('lost', {}).get('total', 0) if trades else 0,
            'win_rate': (trades.get('won', {}).get('total', 0) / trades.get('total', {}).get('total', 1) * 100) if trades and trades.get('total', {}).get('total', 0) > 0 else 0,
            'avg_trade': returns.get('ravg', 0) * 100 if returns else 0,
            'sqn': sqn.get('sqn', 0) if sqn else 0,
            'buy_hold_return': buy_hold_return * 100,
            'outperformance': (total_return - buy_hold_return) * 100
        }
        
        return metrics
    
    def plot_results(self, save_path: str = None):
        """
        Plot backtest results using backtrader's built-in plotting
        
        Args:
            save_path: Path to save the plot
        """
        if self.cerebro is None:
            raise ValueError("Must run backtest before plotting")
        
        # Create figure
        fig = plt.figure(figsize=(12, 8))
        
        # Plot using backtrader
        self.cerebro.plot(
            style='candlestick',
            barup='green',
            bardown='red',
            volume=False,
            fig=fig
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate detailed backtest report"""
        
        sqn_interpretation = ""
        if metrics['sqn'] < 1.6:
            sqn_interpretation = "Poor"
        elif metrics['sqn'] < 1.9:
            sqn_interpretation = "Below Average"
        elif metrics['sqn'] < 2.4:
            sqn_interpretation = "Average"
        elif metrics['sqn'] < 2.9:
            sqn_interpretation = "Good"
        elif metrics['sqn'] < 5.0:
            sqn_interpretation = "Excellent"
        else:
            sqn_interpretation = "Superb"
        
        report = f"""
BACKTRADER ML STRATEGY BACKTEST REPORT
=====================================

Performance Summary:
-------------------
Initial Capital:     ${metrics['initial_capital']:,.2f}
Final Portfolio:     ${metrics['final_value']:,.2f}
Total Return:        {metrics['total_return']:.2f}%
Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}
Max Drawdown:        {metrics['max_drawdown']:.2f}%
SQN Score:           {metrics['sqn']:.2f} ({sqn_interpretation})

Trading Statistics:
------------------
Total Trades:        {metrics['total_trades']}
Winning Trades:      {metrics['won_trades']}
Losing Trades:       {metrics['lost_trades']}
Win Rate:            {metrics['win_rate']:.2f}%
Avg Trade Return:    {metrics['avg_trade']:.2f}%

Comparison:
-----------
Buy & Hold Return:   {metrics['buy_hold_return']:.2f}%
Strategy Outperform: {metrics['outperformance']:.2f}%

System Quality Number (SQN) Interpretation:
- 1.6-1.9: Poor
- 2.0-2.4: Average
- 2.5-2.9: Good
- 3.0-5.0: Excellent
- 5.0+: Superb
"""
        return report


def run_full_backtest(model_path: str, data_path: str, pair: str, timeframe: str, lookforward: int = 7):
    """
    Run complete backtest with trained ML model using backtrader
    
    Args:
        model_path: Path to saved ML model
        data_path: Path to OHLCV data
        pair: Cryptocurrency pair (e.g., 'BTCUSDT')
        timeframe: Timeframe (e.g., '1d', '4h', '1h')
        lookforward: Lookforward period used in model
    """
    from ML_new import CryptoMLStrategy
    
    # Load trained model
    strategy = CryptoMLStrategy(lookforward=lookforward)
    strategy.load_model(model_path)
    
    # Load data
    data = pd.read_csv(data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    
    # Generate predictions for entire dataset
    print(f"Generating predictions for {pair} {timeframe}...")
    predictions = strategy.predict_signals(data)
    
    # Run backtest using backtrader
    print(f"Running backtest for {pair} {timeframe}...")
    backtester = MLBacktraderBacktest(initial_capital=10000, commission=0.001)
    results = backtester.run_backtest(data, predictions)
    
    # Add pair and timeframe to results
    results['pair'] = pair
    results['timeframe'] = timeframe
    
    return backtester, results


def run_multi_backtest(model_path: str, data_dir: str, lookforward: int = 7):
    """
    Run backtests on multiple cryptocurrency pairs and timeframes
    
    Args:
        model_path: Path to saved ML model
        data_dir: Base directory containing data
        lookforward: Lookforward period used in model
    """
    # Define pairs and timeframes to test
    pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT', 'SOLUSDT']
    timeframes = ['1d', '4h', '1h']
    
    all_results = []
    
    for pair in pairs:
        for timeframe in timeframes:
            # Construct data path
            data_path = os.path.join(data_dir, "binance", pair, timeframe, "2018_01_01-now.csv")
            
            # Check if file exists
            if not os.path.exists(data_path):
                print(f"Data file not found for {pair} {timeframe}, skipping...")
                continue
            
            try:
                print(f"\n{'='*60}")
                print(f"BACKTESTING {pair} on {timeframe}")
                print('='*60)
                
                # Run backtest
                _, results = run_full_backtest(model_path, data_path, pair, timeframe, lookforward)
                all_results.append(results)
                
            except Exception as e:
                print(f"Error backtesting {pair} {timeframe}: {str(e)}")
                continue
    
    # Aggregate and display results
    print_aggregated_results(all_results)
    
    return all_results


def print_aggregated_results(results_list: List[Dict[str, Any]]):
    """Print aggregated results from multiple backtests"""
    
    print("\n" + "="*80)
    print("AGGREGATED BACKTEST RESULTS - ALL PAIRS AND TIMEFRAMES")
    print("="*80)
    
    # Create summary table
    print("\n{:<12} {:<8} {:>15} {:>12} {:>15}".format(
        "Pair", "TF", "Total Return %", "Win Rate %", "Max DD %"
    ))
    print("-" * 80)
    
    total_returns = []
    win_rates = []
    max_drawdowns = []
    
    for result in results_list:
        pair = result.get('pair', 'Unknown')
        timeframe = result.get('timeframe', 'Unknown')
        total_return = result['total_return']
        win_rate = result['win_rate']
        max_dd = abs(result['max_drawdown'])
        
        total_returns.append(total_return)
        win_rates.append(win_rate)
        max_drawdowns.append(max_dd)
        
        print("{:<12} {:<8} {:>15.2f} {:>12.2f} {:>15.2f}".format(
            pair, timeframe, total_return, win_rate, max_dd
        ))
    
    # Print averages
    print("-" * 80)
    print("{:<12} {:<8} {:>15.2f} {:>12.2f} {:>15.2f}".format(
        "AVERAGE", "", 
        np.mean(total_returns), 
        np.mean(win_rates), 
        np.mean(max_drawdowns)
    ))
    
    # Best and worst performers
    best_idx = np.argmax(total_returns)
    worst_idx = np.argmin(total_returns)
    
    print(f"\nBest Performer: {results_list[best_idx]['pair']} {results_list[best_idx]['timeframe']} "
          f"({results_list[best_idx]['total_return']:.2f}%)")
    print(f"Worst Performer: {results_list[worst_idx]['pair']} {results_list[worst_idx]['timeframe']} "
          f"({results_list[worst_idx]['total_return']:.2f}%)")


# Example usage
if __name__ == "__main__":
    # Path to your trained model
    model_path = "crypto_ml_strategy_lookforward_7.joblib"
    
    # Path to data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, "data")
    
    # Run backtests on multiple pairs and timeframes
    all_results = run_multi_backtest(
        model_path=model_path,
        data_dir=data_dir,
        lookforward=7
    )