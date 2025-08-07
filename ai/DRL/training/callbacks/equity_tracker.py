"""
Professional Equity Tracking Callback for DRL Trading
Implements comprehensive monitoring and logging for financial metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

try:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Figure
    import torch
    SB3_AVAILABLE = True
except ImportError:
    BaseCallback = object
    Figure = object
    SB3_AVAILABLE = False


class ProfessionalEquityCallback(BaseCallback):
    """
    Professional callback for tracking equity curve and trading metrics
    
    Features:
    - Real-time equity curve tracking
    - Comprehensive trading metrics (Sharpe, drawdown, win rate, etc.)
    - W&B and TensorBoard integration
    - Automatic model checkpointing on best performance
    - Risk management alerts
    """
    
    def __init__(self, 
                 eval_env,
                 eval_freq: int = 10000,
                 n_eval_episodes: int = 5,
                 log_path: str = "./logs",
                 save_best_model: bool = True,
                 use_wandb: bool = True,
                 verbose: int = 1):
        super().__init__(verbose)
        
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = Path(log_path)
        self.save_best_model = save_best_model
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # Create directories
        self.log_path.mkdir(parents=True, exist_ok=True)
        (self.log_path / "equity_curves").mkdir(exist_ok=True)
        (self.log_path / "checkpoints").mkdir(exist_ok=True)
        
        # Tracking variables
        self.equity_history = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.trading_metrics_history = []
        self.best_return = -np.inf
        self.best_sharpe = -np.inf
        self.evaluation_count = 0
        
        # Risk management thresholds
        self.max_drawdown_threshold = 0.2  # 20%
        self.min_sharpe_threshold = 0.5
        
        print(f"✅ ProfessionalEquityCallback initialized")
        print(f"   Log path: {self.log_path}")
        print(f"   Evaluation frequency: {self.eval_freq}")
        print(f"   W&B integration: {self.use_wandb}")
        
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate_and_log()
    
    def _evaluate_and_log(self):
        """Comprehensive evaluation and logging"""
        print(f"\n📊 Evaluation at timestep {self.num_timesteps}")
        
        # Run evaluation episodes
        eval_results = self._run_evaluation_episodes()
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(eval_results)
        
        # Log to console
        self._log_to_console(metrics)
        
        # Log to W&B/TensorBoard
        self._log_to_trackers(metrics, eval_results)
        
        # Save best model
        if self.save_best_model:
            self._save_best_model(metrics)
        
        # Risk management alerts
        self._check_risk_alerts(metrics)
        
        # Save detailed results
        self._save_evaluation_results(eval_results, metrics)
        
        self.evaluation_count += 1
    
    def _run_evaluation_episodes(self) -> Dict[str, Any]:
        """Run evaluation episodes and collect detailed data"""
        episode_returns = []
        episode_lengths = []
        portfolio_values = []
        actions_taken = []
        trades_history = []
        
        for episode in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            episode_return = 0
            episode_length = 0
            episode_portfolio = [info.get('portfolio_value', 100000)]
            episode_actions = []
            
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                episode_return += reward
                episode_length += 1
                episode_portfolio.append(info.get('portfolio_value', episode_portfolio[-1]))
                episode_actions.append(action)
                
                if terminated or truncated:
                    break
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            portfolio_values.append(episode_portfolio)
            actions_taken.append(episode_actions)
            
            # Collect trading history if available
            if hasattr(self.eval_env, 'trades_history'):
                trades_history.extend(self.eval_env.trades_history)
        
        return {
            'returns': episode_returns,
            'lengths': episode_lengths,
            'portfolio_values': portfolio_values,
            'actions': actions_taken,
            'trades': trades_history,
            'num_episodes': self.n_eval_episodes
        }
    
    def _calculate_comprehensive_metrics(self, eval_results: Dict) -> Dict[str, float]:
        """Calculate comprehensive trading metrics"""
        returns = np.array(eval_results['returns'])
        lengths = np.array(eval_results['lengths'])
        
        # Basic metrics
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        mean_length = np.mean(lengths)
        
        # Portfolio analysis
        all_portfolio_values = []
        for pv in eval_results['portfolio_values']:
            all_portfolio_values.extend(pv)
        
        portfolio_returns = np.diff(all_portfolio_values) / all_portfolio_values[:-1]
        portfolio_returns = portfolio_returns[~np.isnan(portfolio_returns)]
        
        # Sharpe Ratio (annualized, assuming 15min intervals)
        if len(portfolio_returns) > 0 and np.std(portfolio_returns) > 0:
            sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252 * 24 * 4)  # 15min intervals
        else:
            sharpe_ratio = 0.0
        
        # Maximum Drawdown
        max_drawdown = self._calculate_max_drawdown(all_portfolio_values)
        
        # Calmar Ratio
        total_return = (all_portfolio_values[-1] - all_portfolio_values[0]) / all_portfolio_values[0]
        calmar_ratio = total_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0.0
        
        # Trading-specific metrics
        trades = eval_results.get('trades', [])
        win_rate, profit_factor, avg_trade_duration = self._analyze_trades(trades)
        
        # Action analysis
        action_distribution = self._analyze_actions(eval_results['actions'])
        
        # Volatility
        volatility = np.std(portfolio_returns) * np.sqrt(252 * 24 * 4) if len(portfolio_returns) > 0 else 0.0
        
        return {
            'mean_episode_reward': mean_return,
            'std_episode_reward': std_return,
            'mean_episode_length': mean_length,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'volatility': volatility,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'total_trades': len(trades),
            'final_portfolio_value': all_portfolio_values[-1] if all_portfolio_values else 100000,
            **action_distribution
        }
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0.0
            
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return np.min(drawdown)
    
    def _analyze_trades(self, trades: List) -> tuple:
        """Analyze trading activity"""
        if not trades:
            return 0.0, 0.0, 0.0
        
        # Win rate
        profitable_trades = sum(1 for trade in trades if getattr(trade, 'realized_pnl', 0) > 0)
        win_rate = profitable_trades / len(trades) if trades else 0.0
        
        # Profit factor
        gross_profit = sum(max(0, getattr(trade, 'realized_pnl', 0)) for trade in trades)
        gross_loss = abs(sum(min(0, getattr(trade, 'realized_pnl', 0)) for trade in trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Average trade duration (simplified)
        avg_duration = np.mean([getattr(trade, 'execution_time', 1) for trade in trades])
        
        return win_rate, profit_factor, avg_duration
    
    def _analyze_actions(self, all_actions: List[List]) -> Dict[str, float]:
        """Analyze action distribution"""
        if not all_actions or not all_actions[0]:
            return {'action_hold_pct': 100.0, 'action_buy_pct': 0.0, 'action_sell_pct': 0.0}
        
        # Flatten all actions
        flat_actions = []
        for episode_actions in all_actions:
            flat_actions.extend(episode_actions)
        
        if not flat_actions:
            return {'action_hold_pct': 100.0, 'action_buy_pct': 0.0, 'action_sell_pct': 0.0}
        
        # Convert to numpy array
        actions = np.array(flat_actions)
        
        # Analyze first dimension (trade signal: -1 to 1)
        if len(actions.shape) > 1:
            trade_signals = actions[:, 0]
        else:
            trade_signals = actions
        
        # Classify actions
        buy_threshold = 0.1
        sell_threshold = -0.1
        
        buy_actions = np.sum(trade_signals > buy_threshold)
        sell_actions = np.sum(trade_signals < sell_threshold)
        hold_actions = len(trade_signals) - buy_actions - sell_actions
        
        total_actions = len(trade_signals)
        
        return {
            'action_buy_pct': (buy_actions / total_actions) * 100,
            'action_sell_pct': (sell_actions / total_actions) * 100,
            'action_hold_pct': (hold_actions / total_actions) * 100
        }
    
    def _log_to_console(self, metrics: Dict[str, float]):
        """Log metrics to console"""
        print(f"📈 Portfolio Value: ${metrics['final_portfolio_value']:,.2f}")
        print(f"📊 Total Return: {metrics['total_return']:.2%}")
        print(f"⚡ Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"📉 Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"🎯 Win Rate: {metrics['win_rate']:.2%}")
        print(f"💼 Total Trades: {metrics['total_trades']}")
        print(f"🎲 Actions - Buy: {metrics['action_buy_pct']:.1f}% | Sell: {metrics['action_sell_pct']:.1f}% | Hold: {metrics['action_hold_pct']:.1f}%")
    
    def _log_to_trackers(self, metrics: Dict[str, float], eval_results: Dict):
        """Log to W&B and TensorBoard"""
        # Log to W&B
        if self.use_wandb and wandb.run is not None:
            wandb.log({
                'timesteps': self.num_timesteps,
                'eval/mean_reward': metrics['mean_episode_reward'],
                'eval/total_return': metrics['total_return'],
                'eval/sharpe_ratio': metrics['sharpe_ratio'],
                'eval/max_drawdown': metrics['max_drawdown'],
                'eval/calmar_ratio': metrics['calmar_ratio'],
                'eval/volatility': metrics['volatility'],
                'eval/win_rate': metrics['win_rate'],
                'eval/profit_factor': metrics['profit_factor'],
                'eval/total_trades': metrics['total_trades'],
                'eval/portfolio_value': metrics['final_portfolio_value'],
                'actions/buy_pct': metrics['action_buy_pct'],
                'actions/sell_pct': metrics['action_sell_pct'],
                'actions/hold_pct': metrics['action_hold_pct']
            })
            
            # Log equity curve plot
            if eval_results['portfolio_values']:
                fig = self._create_equity_curve_plot(eval_results['portfolio_values'])
                wandb.log({"eval/equity_curve": wandb.Image(fig)})
                plt.close(fig)
        
        # Log to TensorBoard (SB3 logger)
        if self.logger is not None:
            self.logger.record('eval/mean_reward', metrics['mean_episode_reward'])
            self.logger.record('eval/total_return', metrics['total_return'])
            self.logger.record('eval/sharpe_ratio', metrics['sharpe_ratio'])
            self.logger.record('eval/max_drawdown', metrics['max_drawdown'])
            self.logger.record('eval/win_rate', metrics['win_rate'])
            self.logger.record('eval/total_trades', metrics['total_trades'])
            
    def _create_equity_curve_plot(self, portfolio_values_list: List[List[float]]) -> plt.Figure:
        """Create equity curve visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot all episode curves
        for i, portfolio_values in enumerate(portfolio_values_list):
            steps = range(len(portfolio_values))
            ax1.plot(steps, portfolio_values, alpha=0.7, label=f'Episode {i+1}')
        
        # Average curve
        if portfolio_values_list:
            max_len = max(len(pv) for pv in portfolio_values_list)
            avg_curve = []
            
            for step in range(max_len):
                values_at_step = []
                for pv in portfolio_values_list:
                    if step < len(pv):
                        values_at_step.append(pv[step])
                if values_at_step:
                    avg_curve.append(np.mean(values_at_step))
            
            ax1.plot(range(len(avg_curve)), avg_curve, 'k--', linewidth=2, label='Average')
        
        ax1.set_title('Portfolio Equity Curves')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown plot
        if portfolio_values_list:
            all_values = []
            for pv in portfolio_values_list:
                all_values.extend(pv)
            
            if all_values:
                values = np.array(all_values)
                peak = np.maximum.accumulate(values)
                drawdown = (values - peak) / peak * 100
                
                ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
                ax2.plot(range(len(drawdown)), drawdown, color='red', linewidth=1)
                ax2.set_title('Drawdown (%)')
                ax2.set_xlabel('Steps')
                ax2.set_ylabel('Drawdown (%)')
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _save_best_model(self, metrics: Dict[str, float]):
        """Save model if it achieves best performance"""
        current_return = metrics['total_return']
        current_sharpe = metrics['sharpe_ratio']
        
        # Save if better total return
        if current_return > self.best_return:
            self.best_return = current_return
            model_path = self.log_path / "checkpoints" / f"best_return_model.zip"
            self.model.save(model_path)
            print(f"💾 New best return model saved: {current_return:.2%}")
        
        # Save if better Sharpe ratio
        if current_sharpe > self.best_sharpe:
            self.best_sharpe = current_sharpe
            model_path = self.log_path / "checkpoints" / f"best_sharpe_model.zip"
            self.model.save(model_path)
            print(f"💾 New best Sharpe model saved: {current_sharpe:.3f}")
    
    def _check_risk_alerts(self, metrics: Dict[str, float]):
        """Check for risk management alerts"""
        if metrics['max_drawdown'] < -self.max_drawdown_threshold:
            print(f"🚨 HIGH DRAWDOWN ALERT: {metrics['max_drawdown']:.2%}")
        
        if metrics['sharpe_ratio'] < self.min_sharpe_threshold:
            print(f"⚠️ LOW SHARPE RATIO: {metrics['sharpe_ratio']:.3f}")
    
    def _save_evaluation_results(self, eval_results: Dict, metrics: Dict[str, float]):
        """Save detailed evaluation results"""
        results = {
            'timestep': self.num_timesteps,
            'evaluation_count': self.evaluation_count,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'eval_episodes': eval_results['num_episodes']
        }
        
        # Save to JSON
        results_file = self.log_path / f"eval_results_{self.num_timesteps}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Update history
        self.trading_metrics_history.append(results)
        
        # Save comprehensive history
        history_file = self.log_path / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.trading_metrics_history, f, indent=2, default=str)


class TradingMetricsCallback(BaseCallback):
    """Lightweight callback for continuous metrics tracking"""
    
    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval
        
    def _on_step(self) -> bool:
        # Log basic training metrics every N steps
        if self.num_timesteps % self.log_interval == 0:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                # Get current episode info from environment if available
                if hasattr(self.locals.get('infos', [{}])[0], 'portfolio_value'):
                    info = self.locals['infos'][0]
                    self.model.logger.record('train/portfolio_value', info.get('portfolio_value', 0))
                    self.model.logger.record('train/balance', info.get('balance', 0))
                    self.model.logger.record('train/total_trades', info.get('total_trades', 0))
        
        return True