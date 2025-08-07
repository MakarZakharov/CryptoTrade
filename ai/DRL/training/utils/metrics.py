"""
Professional Trading Metrics Calculator
Implements comprehensive financial performance metrics for DRL trading evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None
from scipy import stats
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TradeAnalysis:
    """Container for individual trade analysis"""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'BUY' or 'SELL'
    pnl: float
    return_pct: float
    duration: int
    commission: float = 0.0


@dataclass 
class PerformanceMetrics:
    """Container for comprehensive performance metrics"""
    # Return metrics
    total_return: float
    annualized_return: float
    cumulative_return: float
    
    # Risk metrics
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Trading metrics
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    total_trades: int
    avg_trade_duration: float
    
    # Additional metrics
    kelly_criterion: float
    expectancy: float
    recovery_factor: float
    payoff_ratio: float
    
    # Benchmark comparison
    beta: float = 0.0
    alpha: float = 0.0
    correlation_with_benchmark: float = 0.0


class ProfessionalMetricsCalculator:
    """
    Professional-grade metrics calculator for trading strategies
    
    Implements industry-standard financial metrics used by hedge funds
    and institutional trading firms
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 trading_days_per_year: int = 365,
                 intervals_per_day: int = 96):  # 15min intervals
        """
        Initialize calculator with market parameters
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
            trading_days_per_year: Trading days per year for crypto (365)
            intervals_per_day: Number of 15min intervals per day (96)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.intervals_per_year = trading_days_per_year * intervals_per_day
        self.intervals_per_day = intervals_per_day
        
    def calculate_comprehensive_metrics(self,
                                      portfolio_values: List[float],
                                      trades: Optional[List[Any]] = None,
                                      benchmark_returns: Optional[List[float]] = None,
                                      initial_capital: float = 100000.0) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            portfolio_values: Time series of portfolio values
            trades: List of trade objects (optional)
            benchmark_returns: Benchmark returns for comparison (optional)
            initial_capital: Initial portfolio value
            
        Returns:
            PerformanceMetrics object with all calculated metrics
        """
        if len(portfolio_values) < 2:
            return self._create_empty_metrics()
        
        # Convert to numpy arrays
        portfolio_values = np.array(portfolio_values)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return self._create_empty_metrics()
        
        # Basic return metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annualized_return = (1 + total_return) ** (self.intervals_per_year / len(returns)) - 1
        cumulative_return = portfolio_values[-1] / portfolio_values[0] - 1
        
        # Risk metrics
        volatility = np.std(returns) * np.sqrt(self.intervals_per_year)
        max_dd, max_dd_duration = self._calculate_max_drawdown_and_duration(portfolio_values)
        var_95 = np.percentile(returns, 5)
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else 0.0
        
        # Risk-adjusted metrics
        excess_returns = returns - (self.risk_free_rate / self.intervals_per_year)
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(self.intervals_per_year) if np.std(returns) > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(self.intervals_per_year) if downside_std > 0 else 0.0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_dd) if abs(max_dd) > 0 else 0.0
        
        # Information ratio (simplified without benchmark)
        information_ratio = sharpe_ratio  # Simplified when no benchmark
        
        # Trading-specific metrics
        trade_metrics = self._calculate_trade_metrics(trades) if trades else self._create_empty_trade_metrics()
        
        # Additional metrics
        kelly_criterion = self._calculate_kelly_criterion(returns)
        expectancy = trade_metrics['expectancy']
        recovery_factor = abs(total_return / max_dd) if abs(max_dd) > 0 else 0.0
        payoff_ratio = trade_metrics['payoff_ratio']
        
        # Benchmark comparison
        beta, alpha, correlation = self._calculate_benchmark_metrics(returns, benchmark_returns)
        
        return PerformanceMetrics(
            # Return metrics
            total_return=total_return,
            annualized_return=annualized_return, 
            cumulative_return=cumulative_return,
            
            # Risk metrics
            volatility=volatility,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            var_95=var_95,
            cvar_95=cvar_95,
            
            # Risk-adjusted metrics
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            
            # Trading metrics
            win_rate=trade_metrics['win_rate'],
            profit_factor=trade_metrics['profit_factor'],
            avg_win=trade_metrics['avg_win'],
            avg_loss=trade_metrics['avg_loss'],
            largest_win=trade_metrics['largest_win'],
            largest_loss=trade_metrics['largest_loss'],
            total_trades=trade_metrics['total_trades'],
            avg_trade_duration=trade_metrics['avg_duration'],
            
            # Additional metrics
            kelly_criterion=kelly_criterion,
            expectancy=expectancy,
            recovery_factor=recovery_factor,
            payoff_ratio=payoff_ratio,
            
            # Benchmark comparison
            beta=beta,
            alpha=alpha,
            correlation_with_benchmark=correlation
        )
    
    def _calculate_max_drawdown_and_duration(self, portfolio_values: np.ndarray) -> Tuple[float, int]:
        """Calculate maximum drawdown and its duration"""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calculate duration
        max_dd_duration = 0
        current_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                current_duration = 0
                
        return max_drawdown, max_dd_duration
    
    def _calculate_trade_metrics(self, trades: List[Any]) -> Dict[str, float]:
        """Calculate detailed trading metrics"""
        if not trades:
            return self._create_empty_trade_metrics()
        
        # Extract PnL values
        pnls = []
        durations = []
        
        for trade in trades:
            if hasattr(trade, 'realized_pnl'):
                pnls.append(trade.realized_pnl)
            elif hasattr(trade, 'pnl'):
                pnls.append(trade.pnl)
            
            if hasattr(trade, 'execution_time'):
                durations.append(trade.execution_time)
            elif hasattr(trade, 'duration'):
                durations.append(trade.duration)
            else:
                durations.append(1)  # Default duration
        
        if not pnls:
            return self._create_empty_trade_metrics()
        
        pnls = np.array(pnls)
        winning_trades = pnls[pnls > 0]
        losing_trades = pnls[pnls < 0]
        
        # Basic metrics
        total_trades = len(pnls)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        
        # PnL metrics
        total_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0.0
        total_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        # Average metrics
        avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0.0
        avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0.0
        
        # Extreme values
        largest_win = np.max(winning_trades) if len(winning_trades) > 0 else 0.0
        largest_loss = np.min(losing_trades) if len(losing_trades) > 0 else 0.0
        
        # Duration
        avg_duration = np.mean(durations) if durations else 0.0
        
        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        
        # Payoff ratio
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_duration': avg_duration,
            'expectancy': expectancy,
            'payoff_ratio': payoff_ratio
        }
    
    def _create_empty_trade_metrics(self) -> Dict[str, float]:
        """Create empty trade metrics dictionary"""
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_duration': 0.0,
            'expectancy': 0.0,
            'payoff_ratio': 0.0
        }
    
    def _calculate_kelly_criterion(self, returns: np.ndarray) -> float:
        """Calculate Kelly Criterion for optimal position sizing"""
        if len(returns) == 0:
            return 0.0
        
        # Simple Kelly calculation
        mean_return = np.mean(returns)
        variance = np.var(returns)
        
        if variance > 0:
            kelly = mean_return / variance
            return np.clip(kelly, 0, 1)  # Clip between 0 and 1
        return 0.0
    
    def _calculate_benchmark_metrics(self, 
                                   returns: np.ndarray, 
                                   benchmark_returns: Optional[List[float]]) -> Tuple[float, float, float]:
        """Calculate beta, alpha, and correlation with benchmark"""
        if benchmark_returns is None or len(benchmark_returns) == 0:
            return 0.0, 0.0, 0.0
        
        benchmark_returns = np.array(benchmark_returns)
        
        # Align lengths
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]
        
        if min_len < 2:
            return 0.0, 0.0, 0.0
        
        # Correlation
        correlation = np.corrcoef(returns, benchmark_returns)[0, 1] if np.std(returns) > 0 and np.std(benchmark_returns) > 0 else 0.0
        
        # Beta (sensitivity to benchmark)
        if np.var(benchmark_returns) > 0:
            beta = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        else:
            beta = 0.0
        
        # Alpha (excess return after adjusting for beta)
        mean_return = np.mean(returns) * self.intervals_per_year
        mean_benchmark = np.mean(benchmark_returns) * self.intervals_per_year
        alpha = mean_return - (self.risk_free_rate + beta * (mean_benchmark - self.risk_free_rate))
        
        return beta, alpha, correlation
    
    def _create_empty_metrics(self) -> PerformanceMetrics:
        """Create empty metrics object"""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            cumulative_return=0.0,
            volatility=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            var_95=0.0,
            cvar_95=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            total_trades=0,
            avg_trade_duration=0.0,
            kelly_criterion=0.0,
            expectancy=0.0,
            recovery_factor=0.0,
            payoff_ratio=0.0
        )


class BenchmarkCalculator:
    """Calculate benchmark strategies for comparison"""
    
    @staticmethod
    def calculate_buy_and_hold(prices: List[float], initial_capital: float = 100000.0) -> List[float]:
        """Calculate buy and hold strategy returns"""
        if len(prices) < 2:
            return [initial_capital]
        
        prices = np.array(prices)
        initial_price = prices[0]
        
        # Buy and hold portfolio value
        portfolio_values = []
        for price in prices:
            portfolio_value = initial_capital * (price / initial_price)
            portfolio_values.append(portfolio_value)
        
        return portfolio_values
    
    @staticmethod
    def calculate_random_trading(prices: List[float], 
                               initial_capital: float = 100000.0,
                               trade_frequency: float = 0.1,
                               seed: int = 42) -> List[float]:
        """Calculate random trading strategy"""
        np.random.seed(seed)
        
        if len(prices) < 2:
            return [initial_capital]
        
        prices = np.array(prices)
        portfolio_values = [initial_capital]
        cash = initial_capital
        position = 0.0
        
        for i in range(1, len(prices)):
            current_price = prices[i]
            
            # Random trading decision
            if np.random.random() < trade_frequency:
                if position == 0 and cash > current_price * 10:  # Buy
                    shares_to_buy = cash * 0.5 / current_price
                    position += shares_to_buy
                    cash -= shares_to_buy * current_price
                elif position > 0:  # Sell
                    cash += position * current_price * 0.5
                    position *= 0.5
            
            portfolio_value = cash + position * current_price
            portfolio_values.append(portfolio_value)
        
        return portfolio_values
    
    @staticmethod
    def calculate_simple_ma_crossover(prices: List[float],
                                    initial_capital: float = 100000.0,
                                    short_window: int = 20,
                                    long_window: int = 50) -> List[float]:
        """Calculate simple moving average crossover strategy"""
        if len(prices) < max(short_window, long_window):
            return BenchmarkCalculator.calculate_buy_and_hold(prices, initial_capital)
        
        prices = np.array(prices)
        portfolio_values = [initial_capital] * long_window
        cash = initial_capital
        position = 0.0
        
        for i in range(long_window, len(prices)):
            current_price = prices[i]
            
            # Calculate moving averages
            short_ma = np.mean(prices[i-short_window:i])
            long_ma = np.mean(prices[i-long_window:i])
            
            # Trading logic
            if short_ma > long_ma and position == 0 and cash > current_price * 10:
                # Buy signal
                shares_to_buy = cash * 0.95 / current_price
                position = shares_to_buy
                cash = cash * 0.05  # Keep 5% cash
            elif short_ma < long_ma and position > 0:
                # Sell signal
                cash = position * current_price * 0.999  # Account for slippage
                position = 0.0
            
            portfolio_value = cash + position * current_price
            portfolio_values.append(portfolio_value)
        
        return portfolio_values


class MetricsVisualizer:
    """Create professional visualizations for trading metrics"""
    
    @staticmethod
    def create_performance_dashboard(metrics: PerformanceMetrics,
                                   portfolio_values: List[float],
                                   benchmark_values: Optional[List[float]] = None,
                                   title: str = "Trading Performance Dashboard") -> plt.Figure:
        """Create comprehensive performance dashboard"""
        fig = plt.figure(figsize=(16, 12))
        
        # Layout: 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Equity curve
        ax1 = fig.add_subplot(gs[0, :2])
        steps = range(len(portfolio_values))
        ax1.plot(steps, portfolio_values, label='Strategy', linewidth=2, color='blue')
        
        if benchmark_values and len(benchmark_values) == len(portfolio_values):
            ax1.plot(steps, benchmark_values, label='Buy & Hold', linewidth=1, color='gray', alpha=0.7)
        
        ax1.set_title('Equity Curve')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[1, :2])
        portfolio_values_arr = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values_arr)
        drawdown = (portfolio_values_arr - peak) / peak * 100
        
        ax2.fill_between(steps, drawdown, 0, alpha=0.3, color='red')
        ax2.plot(steps, drawdown, color='red', linewidth=1)
        ax2.set_title(f'Drawdown (Max: {metrics.max_drawdown:.2%})')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Metrics summary
        ax3 = fig.add_subplot(gs[:, 2])
        ax3.axis('off')
        
        metrics_text = f"""
PERFORMANCE METRICS

Return Metrics:
• Total Return: {metrics.total_return:.2%}
• Annualized Return: {metrics.annualized_return:.2%}
• Volatility: {metrics.volatility:.2%}

Risk Metrics:
• Max Drawdown: {metrics.max_drawdown:.2%}
• VaR (95%): {metrics.var_95:.4f}
• CVaR (95%): {metrics.cvar_95:.4f}

Risk-Adjusted:
• Sharpe Ratio: {metrics.sharpe_ratio:.3f}
• Sortino Ratio: {metrics.sortino_ratio:.3f}
• Calmar Ratio: {metrics.calmar_ratio:.3f}

Trading Metrics:
• Win Rate: {metrics.win_rate:.2%}
• Profit Factor: {metrics.profit_factor:.2f}
• Total Trades: {metrics.total_trades}
• Expectancy: {metrics.expectancy:.4f}

Additional:
• Kelly Criterion: {metrics.kelly_criterion:.3f}
• Recovery Factor: {metrics.recovery_factor:.2f}
        """.strip()
        
        ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes, 
                verticalalignment='top', fontfamily='monospace', fontsize=10)
        
        # 4. Returns distribution
        ax4 = fig.add_subplot(gs[2, 0])
        if len(portfolio_values) > 1:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            returns = returns[~np.isnan(returns)]
            ax4.hist(returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax4.axvline(np.mean(returns), color='red', linestyle='--', label=f'Mean: {np.mean(returns):.4f}')
            ax4.set_title('Returns Distribution')
            ax4.set_xlabel('Returns')
            ax4.set_ylabel('Frequency')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # 5. Monthly returns heatmap (if enough data)
        ax5 = fig.add_subplot(gs[2, 1])
        if len(portfolio_values) > 30:  # At least 30 data points
            # Simplified monthly returns visualization
            monthly_returns = []
            chunk_size = max(1, len(portfolio_values) // 12)  # Approximate months
            
            for i in range(0, len(portfolio_values) - chunk_size, chunk_size):
                start_val = portfolio_values[i]
                end_val = portfolio_values[i + chunk_size]
                monthly_return = (end_val - start_val) / start_val
                monthly_returns.append(monthly_return)
            
            if monthly_returns:
                months = range(len(monthly_returns))
                colors = ['red' if ret < 0 else 'green' for ret in monthly_returns]
                bars = ax5.bar(months, monthly_returns, color=colors, alpha=0.7)
                ax5.set_title('Period Returns')
                ax5.set_xlabel('Period')
                ax5.set_ylabel('Return')
                ax5.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        return fig
    
    @staticmethod
    def create_trades_analysis(trades: List[Any]) -> plt.Figure:
        """Create detailed trades analysis visualization"""
        if not trades:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No trades available for analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Trade Analysis')
            return fig
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract trade data
        pnls = []
        durations = []
        for trade in trades:
            if hasattr(trade, 'realized_pnl'):
                pnls.append(trade.realized_pnl)
            if hasattr(trade, 'execution_time'):
                durations.append(trade.execution_time)
            elif hasattr(trade, 'duration'):
                durations.append(trade.duration)
        
        if pnls:
            # 1. PnL distribution
            ax1.hist(pnls, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(0, color='red', linestyle='--', label='Breakeven')
            ax1.set_title('Trade PnL Distribution')
            ax1.set_xlabel('PnL')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Cumulative PnL
            cumulative_pnl = np.cumsum(pnls)
            ax2.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='green')
            ax2.set_title('Cumulative PnL')
            ax2.set_xlabel('Trade Number')
            ax2.set_ylabel('Cumulative PnL')
            ax2.grid(True, alpha=0.3)
            
            # 3. Winning vs Losing trades
            winning_trades = [pnl for pnl in pnls if pnl > 0]
            losing_trades = [pnl for pnl in pnls if pnl < 0]
            
            categories = ['Winning', 'Losing']
            counts = [len(winning_trades), len(losing_trades)]
            colors = ['green', 'red']
            
            ax3.bar(categories, counts, color=colors, alpha=0.7)
            ax3.set_title('Win/Loss Count')
            ax3.set_ylabel('Number of Trades')
            
            # Add percentages
            total_trades = len(pnls)
            for i, count in enumerate(counts):
                percentage = count / total_trades * 100
                ax3.text(i, count + 0.1, f'{percentage:.1f}%', ha='center')
            
            ax3.grid(True, alpha=0.3)
        
        # 4. Trade duration analysis
        if durations:
            ax4.hist(durations, bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax4.set_title('Trade Duration Distribution')
            ax4.set_xlabel('Duration')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig