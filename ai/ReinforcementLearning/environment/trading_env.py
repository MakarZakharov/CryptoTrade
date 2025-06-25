"""
Lightweight Trading Environment for Reinforcement Learning
Optimized for performance on limited hardware
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from ..features.feature_extractor import FeatureExtractor
from ..config import *


class TradingEnvironment(gym.Env):
    """
    Trading environment optimized for crypto trading with RL
    Supports multiple timeframes and comprehensive indicators
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 timeframe: str = '1h',
                 initial_balance: float = INITIAL_BALANCE,
                 trading_fees: float = TRADING_FEES,
                 max_position_size: float = MAX_POSITION_SIZE,
                 reward_config: Dict = None):
        """
        Initialize trading environment
        
        Args:
            data: Historical OHLCV data
            timeframe: Trading timeframe
            initial_balance: Starting balance
            trading_fees: Trading fee percentage
            max_position_size: Maximum position size as fraction of balance
            reward_config: Reward calculation configuration
        """
        super().__init__()
        
        self.data = data
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.trading_fees = trading_fees
        self.max_position_size = max_position_size
        self.reward_config = reward_config or REWARD_CONFIG
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor({
            'INDICATORS_CONFIG': INDICATORS_CONFIG,
            'STATE_WINDOW': STATE_WINDOW,
            'FEATURE_SCALING': FEATURE_SCALING
        })
        
        # Extract all features
        self.features = self.feature_extractor.extract_features(data, timeframe)
        self.n_features = self.features.shape[1]
        
        # Environment parameters
        self.current_step = 0
        self.max_steps = len(data) - STATE_WINDOW - 1
        
        # Account state
        self.balance = initial_balance
        self.position = 0  # Current position in base currency
        self.position_price = 0  # Average entry price
        self.trades = []
        self.trade_history = []
        
        # Performance tracking
        self.peak_balance = initial_balance
        self.drawdown = 0
        self.total_profit = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Action space: [action_type, position_size]
        # action_type: 0=hold, 1=buy, 2=sell
        # position_size: 0.0 to 1.0 (fraction of available balance/position)
        self.action_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([2, 1]), 
            dtype=np.float32
        )
        
        # Observation space: features + account state
        account_state_size = 7  # balance, position, profit, drawdown, etc.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(STATE_WINDOW, self.n_features + account_state_size),
            dtype=np.float32
        )
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = STATE_WINDOW
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.trades = []
        self.trade_history = []
        
        self.peak_balance = self.initial_balance
        self.drawdown = 0
        self.total_profit = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step within the environment
        
        Args:
            action: [action_type, position_size]
            
        Returns:
            observation, reward, done, info
        """
        # Parse action
        action_type = int(action[0])
        position_size = float(action[1])
        
        # Get current price
        current_price = self.data['close'].iloc[self.current_step]
        
        # Store previous portfolio value
        prev_portfolio_value = self._get_portfolio_value()
        
        # Execute action
        if action_type == 1:  # Buy
            self._execute_buy(current_price, position_size)
        elif action_type == 2:  # Sell
            self._execute_sell(current_price, position_size)
        # action_type == 0 is hold, do nothing
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(prev_portfolio_value)
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        # Get observation
        observation = self._get_observation()
        
        # Additional info
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self._get_portfolio_value(),
            'drawdown': self.drawdown,
            'total_profit': self.total_profit,
            'trades': len(self.trade_history),
            'win_rate': self._get_win_rate()
        }
        
        return observation, reward, done, info
    
    def _execute_buy(self, price: float, position_size: float):
        """Execute buy order"""
        if self.balance <= 0:
            return
        
        # Calculate order size
        max_buy_value = self.balance * self.max_position_size
        buy_value = max_buy_value * position_size
        
        # Apply fees
        fee = buy_value * self.trading_fees
        buy_value_after_fee = buy_value - fee
        
        if buy_value_after_fee <= 0:
            return
        
        # Calculate position size
        buy_amount = buy_value_after_fee / price
        
        # Update position
        if self.position > 0:
            # Average up
            total_value = self.position * self.position_price + buy_value_after_fee
            self.position += buy_amount
            self.position_price = total_value / self.position
        else:
            self.position = buy_amount
            self.position_price = price
        
        # Update balance
        self.balance -= (buy_value_after_fee + fee)
        
        # Record trade
        self.trades.append({
            'step': self.current_step,
            'type': 'buy',
            'price': price,
            'amount': buy_amount,
            'value': buy_value_after_fee,
            'fee': fee
        })
    
    def _execute_sell(self, price: float, position_size: float):
        """Execute sell order"""
        if self.position <= 0:
            return
        
        # Calculate sell amount
        sell_amount = self.position * position_size
        
        if sell_amount <= 0:
            return
        
        # Calculate sell value
        sell_value = sell_amount * price
        fee = sell_value * self.trading_fees
        sell_value_after_fee = sell_value - fee
        
        # Calculate profit/loss
        cost_basis = sell_amount * self.position_price
        profit = sell_value_after_fee - cost_basis
        
        # Update totals
        self.total_profit += profit
        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update position
        self.position -= sell_amount
        if self.position <= 0:
            self.position = 0
            self.position_price = 0
        
        # Update balance
        self.balance += sell_value_after_fee
        
        # Update peak and drawdown
        portfolio_value = self._get_portfolio_value()
        if portfolio_value > self.peak_balance:
            self.peak_balance = portfolio_value
        self.drawdown = (self.peak_balance - portfolio_value) / self.peak_balance
        
        # Record trade
        trade = {
            'step': self.current_step,
            'type': 'sell',
            'price': price,
            'amount': sell_amount,
            'value': sell_value_after_fee,
            'fee': fee,
            'profit': profit
        }
        self.trades.append(trade)
        self.trade_history.append(trade)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Get feature window
        start_idx = self.current_step - STATE_WINDOW
        end_idx = self.current_step
        
        feature_window = self.features[start_idx:end_idx]
        
        # Get account state
        portfolio_value = self._get_portfolio_value()
        
        account_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            self.position * self.data['close'].iloc[self.current_step] / self.initial_balance,  # Normalized position value
            self.total_profit / self.initial_balance,  # Normalized profit
            self.drawdown,  # Current drawdown
            self._get_win_rate(),  # Win rate
            len(self.trades) / 100,  # Normalized trade count
            self._get_sharpe_ratio()  # Sharpe ratio
        ])
        
        # Tile account state to match window size
        account_state_window = np.tile(account_state, (STATE_WINDOW, 1))
        
        # Combine features and account state
        observation = np.concatenate([feature_window, account_state_window], axis=1)
        
        return observation.astype(np.float32)
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        position_value = 0
        if self.position > 0:
            current_price = self.data['close'].iloc[self.current_step]
            position_value = self.position * current_price
        
        return self.balance + position_value
    
    def _calculate_reward(self, prev_portfolio_value: float) -> float:
        """
        Calculate reward based on profit and risk metrics
        Optimized for maximum profit with minimal drawdowns
        """
        current_portfolio_value = self._get_portfolio_value()
        
        # Profit component
        profit = current_portfolio_value - prev_portfolio_value
        profit_reward = profit / self.initial_balance * self.reward_config['profit_weight']
        
        # Drawdown penalty
        drawdown_penalty = -self.drawdown * self.reward_config['drawdown_penalty']
        
        # Sharpe ratio bonus
        sharpe_bonus = self._get_sharpe_ratio() * self.reward_config['sharpe_bonus']
        
        # Trade cost penalty
        if len(self.trades) > 0 and self.trades[-1]['step'] == self.current_step:
            trade_penalty = -self.trades[-1]['fee'] / self.initial_balance * self.reward_config['trade_cost_penalty']
        else:
            trade_penalty = 0
        
        # Holding penalty (to encourage trading when profitable)
        if self.position == 0 and len(self.trades) == 0:
            holding_penalty = -self.reward_config['holding_penalty']
        else:
            holding_penalty = 0
        
        # Win rate bonus
        win_rate_bonus = self._get_win_rate() * self.reward_config['win_rate_bonus']
        
        # Total reward
        reward = (
            profit_reward + 
            drawdown_penalty + 
            sharpe_bonus + 
            trade_penalty + 
            holding_penalty + 
            win_rate_bonus
        )
        
        # Normalize reward
        if NORMALIZE_REWARDS:
            reward = np.tanh(reward)  # Squash to [-1, 1]
        
        return float(reward)
    
    def _get_win_rate(self) -> float:
        """Calculate win rate"""
        total_trades = self.winning_trades + self.losing_trades
        if total_trades == 0:
            return 0.5  # Default to 50%
        return self.winning_trades / total_trades
    
    def _get_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.trade_history) < 2:
            return 0.0
        
        # Calculate returns from trades
        returns = []
        for trade in self.trade_history:
            if 'profit' in trade:
                returns.append(trade['profit'] / self.initial_balance)
        
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        
        # Annualized Sharpe ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Assume 252 trading days per year
        sharpe = np.sqrt(252) * mean_return / std_return
        
        return float(np.clip(sharpe, -3, 3))  # Clip extreme values
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Position: {self.position:.4f}")
            print(f"Portfolio Value: ${self._get_portfolio_value():.2f}")
            print(f"Drawdown: {self.drawdown:.2%}")
            print(f"Total Profit: ${self.total_profit:.2f}")
            print(f"Win Rate: {self._get_win_rate():.2%}")
            print(f"Sharpe Ratio: {self._get_sharpe_ratio():.2f}")
            print("-" * 50)
    
    def get_metrics(self) -> Dict:
        """Get performance metrics"""
        portfolio_value = self._get_portfolio_value()
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        return {
            'total_return': total_return,
            'portfolio_value': portfolio_value,
            'total_profit': self.total_profit,
            'max_drawdown': self.drawdown,
            'win_rate': self._get_win_rate(),
            'sharpe_ratio': self._get_sharpe_ratio(),
            'total_trades': len(self.trade_history),
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }