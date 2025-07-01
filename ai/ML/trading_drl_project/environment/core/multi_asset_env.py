"""
Multi-Asset Trading Environment

Trading environment that supports multiple assets simultaneously.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from gymnasium.spaces import Box, Discrete
import logging

from .base_env import BaseTradingEnv
from .portfolio_manager import PortfolioManager
from .market_simulator import MarketSimulator
from .reward_functions import get_reward_function


class MultiAssetTradingEnv(BaseTradingEnv):
    """
    Multi-asset trading environment
    
    Supports simultaneous trading of multiple cryptocurrency pairs
    with portfolio-level risk management and optimization.
    """
    
    def __init__(
        self,
        data_dict: Dict[str, pd.DataFrame],
        initial_balance: float = 10000.0,
        commission: float = 0.001,
        window_size: int = 50,
        action_type: str = "continuous",  # "discrete" or "continuous"
        reward_function: str = "portfolio_sharpe",
        max_position_pct: float = 0.3,
        min_position_pct: float = 0.01,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-asset trading environment
        
        Args:
            data_dict: Dictionary of asset_name -> OHLCV DataFrame
            initial_balance: Starting portfolio balance
            commission: Trading commission rate
            window_size: Observation window size
            action_type: Type of action space ("discrete" or "continuous")
            reward_function: Reward function type
            max_position_pct: Maximum position size as percentage of portfolio
            min_position_pct: Minimum position size as percentage of portfolio
            config: Environment configuration
        """
        # Validate and align data
        self.asset_names = list(data_dict.keys())
        self.num_assets = len(self.asset_names)
        
        if self.num_assets == 0:
            raise ValueError("At least one asset must be provided")
        
        # Align all data to common timestamps
        aligned_data = self._align_data(data_dict)
        
        # Initialize base class with combined data
        super().__init__(
            data=aligned_data,
            initial_balance=initial_balance,
            commission=commission,
            config=config
        )
        
        self.window_size = window_size
        self.action_type = action_type
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        
        # Initialize components
        self.portfolio = PortfolioManager(
            initial_balance=initial_balance,
            commission_rate=commission,
            assets=self.asset_names
        )
        
        self.market_sims = {
            asset: MarketSimulator(aligned_data[asset])
            for asset in self.asset_names
        }
        
        self.reward_fn = get_reward_function(reward_function)
        
        # Define action and observation spaces
        self._setup_spaces()
        
        self.logger = logging.getLogger(__name__)
        
    def _align_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align all asset data to common timestamps"""
        # Find common date range
        start_dates = [df.index.min() for df in data_dict.values()]
        end_dates = [df.index.max() for df in data_dict.values()]
        
        common_start = max(start_dates)
        common_end = min(end_dates)
        
        # Align all dataframes
        aligned_data = pd.DataFrame()
        
        for asset, df in data_dict.items():
            asset_data = df.loc[common_start:common_end].copy()
            
            # Add asset prefix to columns
            asset_data.columns = [f"{asset}_{col}" for col in asset_data.columns]
            
            if aligned_data.empty:
                aligned_data = asset_data
            else:
                aligned_data = aligned_data.join(asset_data, how='inner')
        
        # Forward fill missing values
        aligned_data = aligned_data.fillna(method='ffill').dropna()
        
        return aligned_data
    
    def _setup_spaces(self) -> None:
        """Setup action and observation spaces"""
        # Action space
        if self.action_type == "discrete":
            # 3^num_assets possible combinations (hold, buy, sell for each asset)
            self.action_space = Discrete(3 ** self.num_assets)
        else:
            # Continuous actions: [-1, 1] for each asset (-1=sell all, 1=buy max, 0=hold)
            self.action_space = Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_assets,),
                dtype=np.float32
            )
        
        # Observation space
        # Features per asset: OHLCV (5) + technical indicators (10) + portfolio state (3)
        features_per_asset = 18
        total_features = features_per_asset * self.num_assets + 5  # +5 for portfolio-level features
        
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, total_features),
            dtype=np.float32
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation for all assets"""
        observations = []
        
        for asset in self.asset_names:
            # Get market data window for this asset
            asset_obs = self.market_sims[asset].get_observation_window(
                self.current_step,
                self.window_size
            )
            
            # Add portfolio state for this asset
            position_pct = self.portfolio.get_asset_position_pct(asset)
            position_value = self.portfolio.get_asset_value(asset)
            unrealized_pnl = self.portfolio.get_asset_unrealized_pnl(asset)
            
            portfolio_features = np.array([position_pct, position_value, unrealized_pnl])
            portfolio_features = np.repeat(
                portfolio_features.reshape(1, -1),
                self.window_size,
                axis=0
            )
            
            # Combine market and portfolio features
            combined_obs = np.concatenate([asset_obs, portfolio_features], axis=1)
            observations.append(combined_obs)
        
        # Concatenate all asset observations
        full_observation = np.concatenate(observations, axis=1)
        
        # Add portfolio-level features
        portfolio_features = self._get_portfolio_features()
        portfolio_features = np.repeat(
            portfolio_features.reshape(1, -1),
            self.window_size,
            axis=0
        )
        
        full_observation = np.concatenate([full_observation, portfolio_features], axis=1)
        
        return full_observation.astype(np.float32)
    
    def _get_portfolio_features(self) -> np.ndarray:
        """Get portfolio-level features"""
        total_value = self.portfolio.get_total_value()
        cash_pct = self.portfolio.cash / total_value if total_value > 0 else 1.0
        total_pnl = self.portfolio.get_total_pnl()
        sharpe_ratio = self.episode_metrics.get('sharpe_ratio', 0)
        max_drawdown = self.episode_metrics.get('max_drawdown', 0)
        
        return np.array([total_value, cash_pct, total_pnl, sharpe_ratio, max_drawdown])
    
    def _take_action(self, action: Any) -> Dict[str, Any]:
        """Execute trading action across multiple assets"""
        if self.action_type == "discrete":
            # Convert discrete action to individual asset actions
            asset_actions = self._decode_discrete_action(action)
        else:
            # Use continuous actions directly
            asset_actions = action
        
        # Execute actions for each asset
        action_results = {}
        total_transactions = 0
        
        for i, asset in enumerate(self.asset_names):
            current_price = self.market_sims[asset].get_current_price()
            asset_action = asset_actions[i]
            
            # Determine action type and size
            if abs(asset_action) < 0.1:  # Hold threshold
                result = self.portfolio.hold_asset(asset, current_price)
            elif asset_action > 0:  # Buy
                target_pct = min(asset_action * self.max_position_pct, self.max_position_pct)
                result = self.portfolio.buy_asset(asset, current_price, target_pct)
            else:  # Sell
                sell_pct = min(abs(asset_action), 1.0)
                result = self.portfolio.sell_asset(asset, current_price, sell_pct)
            
            action_results[asset] = result
            if result.get('success', False):
                total_transactions += 1
        
        # Update market simulators
        for asset in self.asset_names:
            self.market_sims[asset].step()
        
        return {
            'asset_results': action_results,
            'total_transactions': total_transactions,
            'portfolio_value': self.portfolio.get_total_value()
        }
    
    def _decode_discrete_action(self, action: int) -> List[int]:
        """Convert discrete action to individual asset actions"""
        asset_actions = []
        remaining = action
        
        for _ in range(self.num_assets):
            asset_action = remaining % 3  # 0=hold, 1=buy, 2=sell
            asset_actions.append(asset_action - 1)  # Convert to [-1, 0, 1]
            remaining //= 3
        
        return asset_actions
    
    def _calculate_reward(self, action_result: Dict[str, Any]) -> float:
        """Calculate portfolio-level reward"""
        return self.reward_fn.calculate(self.portfolio, action_result)
    
    def _is_done(self) -> bool:
        """Check if episode is finished"""
        # Episode ends if:
        # 1. We've reached the end of data
        # 2. Portfolio value drops below threshold
        # 3. Maximum steps reached
        
        max_steps = len(self.data) - self.window_size - 1
        portfolio_value = self.portfolio.get_total_value()
        min_value_threshold = self.initial_balance * 0.1  # 90% loss threshold
        
        return (
            self.current_step >= max_steps or
            portfolio_value <= min_value_threshold or
            any(sim.is_finished() for sim in self.market_sims.values())
        )
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information"""
        info = {
            'step': self.current_step,
            'portfolio_value': self.portfolio.get_total_value(),
            'cash_balance': self.portfolio.cash,
            'asset_positions': {
                asset: self.portfolio.get_asset_position_pct(asset)
                for asset in self.asset_names
            },
            'asset_values': {
                asset: self.portfolio.get_asset_value(asset)
                for asset in self.asset_names
            },
            'total_trades': self.portfolio.total_trades,
            'commission_paid': self.portfolio.total_commission
        }
        
        # Add current prices
        info['current_prices'] = {
            asset: self.market_sims[asset].get_current_price()
            for asset in self.asset_names
        }
        
        return info
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        observation, info = super().reset(seed=seed)
        
        # Reset portfolio and market simulators
        self.portfolio.reset(self.initial_balance, self.asset_names)
        for sim in self.market_sims.values():
            sim.reset()
        
        return observation, info