import backtrader as bt
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the path to handle imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from CryptoTrade.strategies.TestStrategies.first_strategy import SmaCrossStrategy
except ImportError as e:
    print(f"Warning: Could not import SmaCrossStrategy: {e}")
    print("Please ensure the strategy file exists at the correct path")
    SmaCrossStrategy = None


def add_data_validation(cerebro, data):
    """Add data validation and debugging info."""
    try:
        cerebro.adddata(data)

        # Add some debugging for data validation
        print(f"Data period: Check your CSV date range")
        print("First few lines should show OHLC data loading...")

        # Add analyzers for better statistics
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        # Try to get some basic info about the data
        print(f"Data feed added successfully")
        return True
    except Exception as e:
        print(f"Error adding data feed: {e}")
        return False
    """
    Create a simple test strategy if the main strategy is not available.
    This helps debug data loading issues.
    """

    class SimpleTestStrategy(bt.Strategy):
        params = (
            ('fast_period', 10),
            ('slow_period', 50),
        )

        def __init__(self):
            self.trades = []
            self.fast_ma = bt.indicators.SimpleMovingAverage(
                self.data.close, period=self.params.fast_period
            )
            self.slow_ma = bt.indicators.SimpleMovingAverage(
                self.data.close, period=self.params.slow_period
            )
            self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
            print(f"Strategy initialized with {len(self.data)} data points")

        def next(self):
            if not self.position:  # Not in market
                if self.crossover > 0:  # Fast MA crosses above slow MA
                    size = int(self.broker.getcash() / self.data.close[0] * 0.95)
                    if size > 0:
                        self.buy(size=size)
                        print(f"BUY signal at {self.data.datetime.date(0)}: "
                              f"Price={self.data.close[0]:.2f}, Size={size}")

            else:  # In market
                if self.crossover < 0:  # Fast MA crosses below slow MA
                    self.sell(size=self.position.size)
                    print(f"SELL signal at {self.data.datetime.date(0)}: "
                          f"Price={self.data.close[0]:.2f}")

        def notify_trade(self, trade):
            if trade.isclosed:
                profit = trade.pnl
                self.trades.append(profit)
                print(f"Trade closed: Profit=${profit:.2f}")

    return SimpleTestStrategy


def max_drawdown(values):
    """
    Calculate maximum drawdown from a list of portfolio values.

    Args:
        values (list): List of portfolio values over time

    Returns:
        float: Maximum drawdown as a decimal (0.1 = 10%)
    """
    if not values or len(values) < 2:
        return 0.0

    max_val = values[0]
    drawdown = 0.0

    for v in values:
        if v > max_val:
            max_val = v
        if max_val > 0:  # Avoid division by zero
            dd = (max_val - v) / max_val
            if dd > drawdown:
                drawdown = dd

    return drawdown


def validate_data_file(datafile):
    """
    Validate that the data file exists and is readable.

    Args:
        datafile (str): Path to the CSV data file

    Returns:
        bool: True if file is valid, False otherwise
    """
    if not os.path.exists(datafile):
        print(f"Error: Data file not found: {datafile}")
        return False

    if not os.path.isfile(datafile):
        print(f"Error: Path is not a file: {datafile}")
        return False

    if not os.access(datafile, os.R_OK):
        print(f"Error: Cannot read file: {datafile}")
        return False

    return True


def run_backtest(
        strategy_class,
        datafile,
        cash=1000,
        fast_period=10,
        slow_period=50,
        commission=0.00075,
        plot_results=True
):
    """
    Run a backtest with the given parameters.

    Args:
        strategy_class: The strategy class to use
        datafile (str): Path to CSV data file
        cash (float): Starting cash amount
        fast_period (int): Fast moving average period
        slow_period (int): Slow moving average period
        commission (float): Commission rate (0.00075 = 0.075%)
        plot_results (bool): Whether to plot results

    Returns:
        dict: Results dictionary with key metrics
    """
    print("=" * 50)
    print("BACKTEST CONFIGURATION")
    print("=" * 50)
    print(f"Current working directory: {os.getcwd()}")
    print(f"Data file: {datafile}")
    print(f"Starting capital: ${cash:,.2f}")
    print(f"Fast MA period: {fast_period}")
    print(f"Slow MA period: {slow_period}")
    print(f"Commission rate: {commission * 100:.3f}%")

    # Validate inputs
    if strategy_class is None:
        print("Error: Strategy class is None. Cannot run backtest.")
        return None

    if not validate_data_file(datafile):
        return None

    if fast_period >= slow_period:
        print("Warning: Fast period should be less than slow period")

    try:
        # Initialize Cerebro
        cerebro = bt.Cerebro()
        cerebro.broker.set_cash(cash)
        cerebro.broker.setcommission(commission=commission)

        # Add strategy
        cerebro.addstrategy(
            strategy_class,
            fast_period=fast_period,
            slow_period=slow_period
        )

        # Load data - updated for your CSV format
        data = bt.feeds.GenericCSVData(
            dataname=datafile,
            dtformat='%Y-%m-%dT%H:%M:%S',
            datetime=0,  # timestamp column
            open=1,  # open column
            high=2,  # high column
            low=3,  # low column
            close=4,  # close column
            volume=5,  # volume column
            openinterest=-1,  # not present, use -1
            header=1,  # skip header row
            separator=','  # explicitly set separator
        )
        cerebro.adddata(data)

        # Run backtest
        print("\n" + "=" * 50)
        print("RUNNING BACKTEST...")
        print("=" * 50)

        results = cerebro.run()
        strat = results[0]

        # Calculate results
        final_value = cerebro.broker.getvalue()
        total_profit = final_value - cash
        roi_percent = (total_profit / cash) * 100

        # Print basic results
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)
        print(f"Starting Capital: ${cash:,.2f}")
        print(f"Final Capital: ${final_value:,.2f}")
        print(f"Total Profit: ${total_profit:,.2f}")
        print(f"Return on Investment: {roi_percent:.2f}%")

        # Safely get analyzer results
        try:
            trades_analyzer = strat.analyzers.trades.get_analysis()
            drawdown_analyzer = strat.analyzers.drawdown.get_analysis()

            print(f"Trades analysis: {trades_analyzer}")
            print(f"Drawdown analysis: {drawdown_analyzer}")

            # Extract trade data safely
            total_trades = trades_analyzer.get('total', {}).get('total', 0) if trades_analyzer else 0
            won_trades = trades_analyzer.get('won', {}).get('total', 0) if trades_analyzer else 0
            lost_trades = trades_analyzer.get('lost', {}).get('total', 0) if trades_analyzer else 0

        except Exception as e:
            print(f"Error getting analyzer results: {e}")
            total_trades = 0
            won_trades = 0
            lost_trades = 0

        print(f"\nTotal Trades: {total_trades}")

        results_dict = {
            'starting_capital': cash,
            'final_capital': final_value,
            'total_profit': total_profit,
            'roi_percent': roi_percent,
            'total_trades': total_trades,
            'max_drawdown_percent': 0.0
        }

        if total_trades > 0:
            win_rate = (won_trades / total_trades) * 100 if total_trades > 0 else 0

            print(f"Winning Trades: {won_trades} ({win_rate:.1f}%)")
            print(f"Losing Trades: {lost_trades}")

            # Get additional trade statistics if available
            try:
                if 'won' in trades_analyzer and 'pnl' in trades_analyzer['won']:
                    avg_win = trades_analyzer['won']['pnl'].get('average', 0)
                    max_win = trades_analyzer['won']['pnl'].get('max', 0)
                    print(f"Average Winning Trade: ${avg_win:.2f}")
                    print(f"Best Trade: ${max_win:.2f}")

                if 'lost' in trades_analyzer and 'pnl' in trades_analyzer['lost']:
                    avg_loss = trades_analyzer['lost']['pnl'].get('average', 0)
                    max_loss = trades_analyzer['lost']['pnl'].get('max', 0)
                    print(f"Average Losing Trade: ${avg_loss:.2f}")
                    print(f"Worst Trade: ${max_loss:.2f}")

                # Get drawdown information
                if drawdown_analyzer:
                    max_dd_percent = drawdown_analyzer.get('max', {}).get('drawdown', 0) * 100
                    print(f"Maximum Drawdown: {max_dd_percent:.2f}%")
                    results_dict['max_drawdown_percent'] = max_dd_percent

            except Exception as e:
                print(f"Error getting detailed trade statistics: {e}")

            # Update results dict
            results_dict.update({
                'winning_trades': won_trades,
                'losing_trades': lost_trades,
                'win_rate_percent': win_rate,
            })
        else:
            print("No trades were executed!")
            print("\nPossible reasons:")
            print("1. Strategy conditions never triggered")
            print("2. Not enough data for moving averages to calculate")
            print("3. Data format issues")
            print("4. Strategy logic problems")
            print("5. All signals occurred before moving averages were ready")

            # Check if strategy has custom tracking
            if hasattr(strat, 'trade_count'):
                print(f"Strategy trade counter: {strat.trade_count}")
            if hasattr(strat, 'data_points'):
                print(f"Data points processed: {strat.data_points}")
            if hasattr(strat, 'trades') and isinstance(strat.trades, list):
                print(f"Strategy custom trades list: {len(strat.trades)} items")

        # Plot results if requested
        if plot_results:
            try:
                print("\nGenerating plot...")
                cerebro.plot(style='candlestick')
            except Exception as e:
                print(f"Warning: Could not generate plot: {e}")

        return results_dict

    except FileNotFoundError:
        print(f"Error: Could not find data file: {datafile}")
        return None
    except Exception as e:
        print(f"Error during backtest: {e}")
        return None


def main():
    """Main function to run the backtest."""
    # Configuration
    datafile = "../../data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv"

    # Alternative datafile paths to try
    alternative_paths = [
        "data/binance/BTCUSDT/1d/2018_01_01-2025_01_01.csv",
        "./2018_01_01-2025_01_01.csv",
        "BTCUSDT_1d.csv"
    ]

    # Try to find a valid data file
    valid_datafile = None
    if validate_data_file(datafile):
        valid_datafile = datafile
    else:
        print(f"Primary data file not found: {datafile}")
        print("Trying alternative paths...")
        for alt_path in alternative_paths:
            if validate_data_file(alt_path):
                valid_datafile = alt_path
                print(f"Found alternative data file: {alt_path}")
                break

    if not valid_datafile:
        print("Error: No valid data file found. Please check the file path.")
        print("Expected CSV format: datetime,open,high,low,close,volume")
        return

    # Use test strategy if main strategy not available
    strategy_to_use = SmaCrossStrategy if SmaCrossStrategy is not None else SimpleTestStrategy
    strategy_name = "SmaCrossStrategy" if SmaCrossStrategy is not None else "SimpleTestStrategy"

    print(f"Using strategy: {strategy_name}")

    # Run backtest
    results = run_backtest(
        strategy_class=strategy_to_use,
        datafile=valid_datafile,
        cash=1000,
        fast_period=10,
        slow_period=50,
        commission=0.00075,
        plot_results=True
    )

    if results:
        print("\n" + "=" * 50)
        print("BACKTEST COMPLETED SUCCESSFULLY")
        print("=" * 50)
    else:
        print("Backtest failed to complete")


if __name__ == "__main__":
    main()