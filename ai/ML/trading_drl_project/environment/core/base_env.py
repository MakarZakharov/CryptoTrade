"""
Base Environment Abstract Class

Abstract base class for all trading environments.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Space


class BaseTradingEnv(gym.Env, ABC):
    """
    Abstract base class for all trading environments
    
    Provides common functionality and interface for trading environments.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base trading environment
        
        Args:
            data: Market data (OHLCV)
            initial_balance: Starting portfolio balance
            commission: Trading commission rate
            config: Environment configuration
        """
        super().__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.commission = commission
        self.config = config or {}
        
        # Environment state
        self.current_step = 0
        self.done = False
        self.info = {}
        
        # Performance tracking
        self.episode_returns = []
        self.episode_trades = []
        self.episode_metrics = {}
        
    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        pass
    
    @abstractmethod
    def _take_action(self, action: Any) -> Dict[str, Any]:
        """Execute trading action"""
        pass
    
    @abstractmethod
    def _calculate_reward(self, action_result: Dict[str, Any]) -> float:
        """Calculate reward for current step"""
        pass
    
    @abstractmethod
    def _is_done(self) -> bool:
        """Check if episode is finished"""
        pass
    
    @abstractmethod
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information"""
        pass
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.done = False
        
        # Reset episode tracking
        self.episode_returns = []
        self.episode_trades = []
        self.episode_metrics = {}
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        if self.done:
            raise RuntimeError("Episode finished. Call reset() to start new episode.")
        
        # Take action
        action_result = self._take_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(action_result)
        self.episode_returns.append(reward)
        
        # Update step
        self.current_step += 1
        
        # Check if done
        self.done = self._is_done()
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Update episode metrics
        self._update_episode_metrics(action_result, reward)
        
        return observation, reward, self.done, False, info
    
    def _update_episode_metrics(self, action_result: Dict[str, Any], reward: float) -> None:
        """Update episode performance metrics"""
        self.episode_metrics.update({
            'total_return': sum(self.episode_returns),
            'avg_return': np.mean(self.episode_returns) if self.episode_returns else 0,
            'volatility': np.std(self.episode_returns) if len(self.episode_returns) > 1 else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown(),
            'total_trades': len(self.episode_trades)
        })
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio for episode"""
        if len(self.episode_returns) < 2:
            return 0.0
        
        returns = np.array(self.episode_returns)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown for episode"""
        if not self.episode_returns:
            return 0.0
        
        cumulative_returns = np.cumsum(self.episode_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / (peak + 1e-8)
        
        return np.max(drawdown)
    
    def get_episode_metrics(self) -> Dict[str, Any]:
        """Get current episode metrics"""
        return self.episode_metrics.copy()
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Total Return: {sum(self.episode_returns):.4f}")
            print(f"Sharpe Ratio: {self.episode_metrics.get('sharpe_ratio', 0):.4f}")
            print("-" * 30)
        
        return None
    
    def close(self) -> None:
        """Clean up environment resources"""
        pass