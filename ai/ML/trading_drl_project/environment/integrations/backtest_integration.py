"""
Backtest Integration

Integration with existing backtesting framework for enhanced testing capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging

try:
    # Try to import from main backtesting module
    from ...testing.core.backtester import Backtester
    from ...testing.core.metrics import MetricsCalculator
except ImportError:
    # Fallback or mock classes if backtesting module not available
    class Backtester:
        def run_backtest(self, *args, **kwargs):
            raise NotImplementedError("Backtesting module not available")
    
    class MetricsCalculator:
        def calculate_metrics(self, *args, **kwargs):
            raise NotImplementedError("Metrics module not available")


class EnvironmentBacktestIntegrator:
    """
    Integrates DRL environment with traditional backtesting framework
    
    Allows running DRL agents through existing backtest infrastructure
    and comparing performance with traditional strategies.
    """
    
    def __init__(
        self,
        backtester: Optional[Backtester] = None,
        metrics_calculator: Optional[MetricsCalculator] = None
    ):
        """
        Initialize backtest integrator
        
        Args:
            backtester: Backtester instance
            metrics_calculator: Metrics calculator instance
        """
        self.backtester = backtester or Backtester()
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        
        self.logger = logging.getLogger(__name__)
        
    def run_drl_backtest(
        self,
        agent,
        environment,
        start_date: str,
        end_date: str,
        initial_balance: float = 10000.0,
        benchmark_symbol: str = "BTC/USDT"
    ) -> Dict[str, Any]:
        """
        Run DRL agent backtest using environment
        
        Args:
            agent: Trained DRL agent
            environment: Trading environment
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_balance: Initial portfolio balance
            benchmark_symbol: Symbol for benchmark comparison
            
        Returns:
            Backtest results and metrics
        """
        self.logger.info(f"Running DRL backtest from {start_date} to {end_date}")
        
        # Reset environment
        obs, info = environment.reset()
        
        # Initialize tracking
        portfolio_values = []
        actions_taken = []
        rewards = []
        timestamps = []
        
        done = False
        step = 0
        
        while not done:
            # Get agent action
            action, _states = agent.predict(obs, deterministic=True)
            
            # Execute action in environment
            obs, reward, done, truncated, info = environment.step(action)
            
            # Record data
            portfolio_values.append(info.get('portfolio_value', initial_balance))
            actions_taken.append(action)
            rewards.append(reward)
            
            # Get timestamp (if available)
            if 'timestamp' in info:
                timestamps.append(info['timestamp'])
            else:
                timestamps.append(step)
            
            step += 1
            
            if step % 1000 == 0:
                self.logger.info(f"Processed {step} steps")
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'timestamp': timestamps,
            'portfolio_value': portfolio_values,
            'action': actions_taken,
            'reward': rewards
        })
        
        if isinstance(timestamps[0], (int, float)):
            # Convert step numbers to approximate dates
            results_df['timestamp'] = pd.date_range(
                start=start_date,
                periods=len(results_df),
                freq='H'  # Assume hourly data
            )
        
        results_df.set_index('timestamp', inplace=True)
        
        # Calculate performance metrics
        metrics = self._calculate_drl_metrics(results_df, initial_balance, benchmark_symbol)
        
        return {
            'results': results_df,
            'metrics': metrics,
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] / initial_balance - 1) * 100,
            'total_steps': step
        }
    
    def compare_with_benchmark(
        self,
        drl_results: Dict[str, Any],
        benchmark_strategy: str = "buy_and_hold",
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Compare DRL performance with benchmark strategy
        
        Args:
            drl_results: Results from DRL backtest
            benchmark_strategy: Type of benchmark strategy
            benchmark_data: Market data for benchmark
            
        Returns:
            Comparison results
        """
        if benchmark_data is None:
            self.logger.warning("No benchmark data provided, skipping comparison")
            return {}
        
        # Calculate benchmark performance
        if benchmark_strategy == "buy_and_hold":
            benchmark_metrics = self._calculate_buy_hold_metrics(benchmark_data)
        else:
            self.logger.warning(f"Unknown benchmark strategy: {benchmark_strategy}")
            return {}
        
        # Compare metrics
        drl_metrics = drl_results['metrics']
        comparison = {
            'drl_metrics': drl_metrics,
            'benchmark_metrics': benchmark_metrics,
            'outperformance': {
                'total_return': drl_metrics.get('total_return', 0) - benchmark_metrics.get('total_return', 0),
                'sharpe_ratio': drl_metrics.get('sharpe_ratio', 0) - benchmark_metrics.get('sharpe_ratio', 0),
                'max_drawdown': benchmark_metrics.get('max_drawdown', 0) - drl_metrics.get('max_drawdown', 0),  # Lower is better
            }
        }
        
        return comparison
    
    def run_walk_forward_analysis(
        self,
        agent_factory,
        environment_factory,
        data: pd.DataFrame,
        train_window: int = 1000,
        test_window: int = 200,
        step_size: int = 100
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis for DRL agent
        
        Args:
            agent_factory: Function to create new agent instances
            environment_factory: Function to create environment instances
            data: Full dataset for analysis
            train_window: Training window size
            test_window: Testing window size
            step_size: Step size between windows
            
        Returns:
            Walk-forward analysis results
        """
        self.logger.info("Starting walk-forward analysis")
        
        results = []
        start_idx = 0
        
        while start_idx + train_window + test_window <= len(data):
            self.logger.info(f"Processing window starting at index {start_idx}")
            
            # Split data
            train_end = start_idx + train_window
            test_end = train_end + test_window
            
            train_data = data.iloc[start_idx:train_end]
            test_data = data.iloc[train_end:test_end]
            
            try:
                # Create and train agent
                agent = agent_factory()
                train_env = environment_factory(train_data)
                
                # Training would happen here (simplified)
                # agent.learn(total_timesteps=10000, env=train_env)
                
                # Test on out-of-sample data
                test_env = environment_factory(test_data)
                test_results = self.run_drl_backtest(
                    agent=agent,
                    environment=test_env,
                    start_date=test_data.index[0].strftime('%Y-%m-%d'),
                    end_date=test_data.index[-1].strftime('%Y-%m-%d')
                )
                
                results.append({
                    'start_idx': start_idx,
                    'train_period': (train_data.index[0], train_data.index[-1]),
                    'test_period': (test_data.index[0], test_data.index[-1]),
                    'metrics': test_results['metrics'],
                    'total_return': test_results['total_return']
                })
                
            except Exception as e:
                self.logger.error(f"Error in window {start_idx}: {e}")
                continue
            
            start_idx += step_size
        
        # Aggregate results
        if results:
            aggregated_metrics = self._aggregate_walk_forward_results(results)
        else:
            aggregated_metrics = {}
        
        return {
            'individual_results': results,
            'aggregated_metrics': aggregated_metrics,
            'total_windows': len(results)
        }
    
    def _calculate_drl_metrics(
        self,
        results_df: pd.DataFrame,
        initial_balance: float,
        benchmark_symbol: str
    ) -> Dict[str, float]:
        """Calculate performance metrics for DRL results"""
        portfolio_values = results_df['portfolio_value']
        
        # Calculate returns
        returns = portfolio_values.pct_change().dropna()
        
        metrics = {
            'total_return': (portfolio_values.iloc[-1] / initial_balance - 1) * 100,
            'annualized_return': self._annualize_return(portfolio_values),
            'volatility': returns.std() * np.sqrt(252) * 100,  # Annualized
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'win_rate': (returns > 0).mean() * 100,
            'profit_factor': self._calculate_profit_factor(returns),
            'calmar_ratio': self._calculate_calmar_ratio(portfolio_values)
        }
        
        return metrics
    
    def _calculate_buy_hold_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate buy and hold benchmark metrics"""
        if 'close' not in data.columns:
            return {}
        
        prices = data['close']
        returns = prices.pct_change().dropna()
        
        metrics = {
            'total_return': (prices.iloc[-1] / prices.iloc[0] - 1) * 100,
            'annualized_return': self._annualize_return(prices),
            'volatility': returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(prices),
            'win_rate': (returns > 0).mean() * 100
        }
        
        return metrics
    
    def _annualize_return(self, values: pd.Series) -> float:
        """Calculate annualized return"""
        if len(values) < 2:
            return 0.0
        
        days = (values.index[-1] - values.index[0]).days
        if days == 0:
            return 0.0
        
        total_return = values.iloc[-1] / values.iloc[0]
        annualized = (total_return ** (365.25 / days)) - 1
        
        return annualized * 100
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        return drawdown.min() * 100
    
    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor"""
        gains = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return np.inf if gains > 0 else 1.0
        
        return gains / losses
    
    def _calculate_calmar_ratio(self, values: pd.Series) -> float:
        """Calculate Calmar ratio"""
        annualized_return = self._annualize_return(values)
        max_drawdown = abs(self._calculate_max_drawdown(values))
        
        if max_drawdown == 0:
            return np.inf if annualized_return > 0 else 0.0
        
        return annualized_return / max_drawdown
    
    def _aggregate_walk_forward_results(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate walk-forward analysis results"""
        if not results:
            return {}
        
        returns = [r['total_return'] for r in results]
        
        aggregated = {
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'win_rate': (np.array(returns) > 0).mean() * 100,
            'best_period': max(returns),
            'worst_period': min(returns),
            'consistency': 1 - (np.std(returns) / (abs(np.mean(returns)) + 1e-8))
        }
        
        return aggregated