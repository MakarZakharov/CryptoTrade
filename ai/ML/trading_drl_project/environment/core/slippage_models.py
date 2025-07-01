"""
Slippage Models

Models transaction costs and slippage for realistic trading simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging


@dataclass
class SlippageResult:
    """Result of slippage calculation"""
    original_price: float
    executed_price: float
    slippage_bps: float
    total_cost_bps: float
    market_impact: float


class BaseSlippageModel(ABC):
    """Base class for slippage models"""
    
    @abstractmethod
    def calculate_slippage(
        self,
        order_size: float,
        price: float,
        volume: float,
        volatility: float,
        side: str  # 'buy' or 'sell'
    ) -> SlippageResult:
        """Calculate slippage for given order"""
        pass


class LinearSlippageModel(BaseSlippageModel):
    """
    Linear slippage model
    
    Simple model where slippage increases linearly with order size.
    """
    
    def __init__(
        self,
        base_spread_bps: float = 5.0,
        impact_coefficient: float = 0.1,
        volatility_multiplier: float = 2.0
    ):
        """
        Initialize linear slippage model
        
        Args:
            base_spread_bps: Base bid-ask spread in basis points
            impact_coefficient: Market impact coefficient
            volatility_multiplier: Volatility impact multiplier
        """
        self.base_spread_bps = base_spread_bps
        self.impact_coefficient = impact_coefficient
        self.volatility_multiplier = volatility_multiplier
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_slippage(
        self,
        order_size: float,
        price: float,
        volume: float,
        volatility: float,
        side: str
    ) -> SlippageResult:
        """Calculate linear slippage"""
        # Normalize order size by average volume
        relative_size = order_size / (volume + 1e-8)
        
        # Base spread cost
        spread_cost = self.base_spread_bps / 2  # Half spread
        
        # Market impact (linear in size)
        market_impact = self.impact_coefficient * relative_size * 10000  # Convert to bps
        
        # Volatility impact
        volatility_impact = volatility * self.volatility_multiplier * 10000
        
        # Total slippage
        total_slippage_bps = spread_cost + market_impact + volatility_impact
        
        # Apply direction
        direction = 1 if side == 'buy' else -1
        slippage_price_change = price * (total_slippage_bps / 10000) * direction
        
        executed_price = price + slippage_price_change
        
        return SlippageResult(
            original_price=price,
            executed_price=executed_price,
            slippage_bps=total_slippage_bps,
            total_cost_bps=total_slippage_bps,
            market_impact=market_impact
        )


class SquareRootSlippageModel(BaseSlippageModel):
    """
    Square root slippage model
    
    More realistic model where market impact follows square root of order size.
    """
    
    def __init__(
        self,
        base_spread_bps: float = 5.0,
        impact_coefficient: float = 0.3,
        volatility_multiplier: float = 1.5,
        temporary_impact_decay: float = 0.5
    ):
        """
        Initialize square root slippage model
        
        Args:
            base_spread_bps: Base bid-ask spread in basis points
            impact_coefficient: Market impact coefficient
            volatility_multiplier: Volatility impact multiplier
            temporary_impact_decay: Decay rate for temporary impact
        """
        self.base_spread_bps = base_spread_bps
        self.impact_coefficient = impact_coefficient
        self.volatility_multiplier = volatility_multiplier
        self.temporary_impact_decay = temporary_impact_decay
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_slippage(
        self,
        order_size: float,
        price: float,
        volume: float,
        volatility: float,
        side: str
    ) -> SlippageResult:
        """Calculate square root slippage"""
        # Normalize order size by average volume
        relative_size = order_size / (volume + 1e-8)
        
        # Base spread cost
        spread_cost = self.base_spread_bps / 2
        
        # Market impact (square root law)
        market_impact = self.impact_coefficient * np.sqrt(relative_size) * 10000
        
        # Volatility impact
        volatility_impact = volatility * self.volatility_multiplier * 10000
        
        # Temporary impact (decays over time)
        temporary_impact = market_impact * self.temporary_impact_decay
        
        # Permanent impact
        permanent_impact = market_impact * (1 - self.temporary_impact_decay)
        
        # Total slippage
        total_slippage_bps = spread_cost + temporary_impact + permanent_impact + volatility_impact
        
        # Apply direction
        direction = 1 if side == 'buy' else -1
        slippage_price_change = price * (total_slippage_bps / 10000) * direction
        
        executed_price = price + slippage_price_change
        
        return SlippageResult(
            original_price=price,
            executed_price=executed_price,
            slippage_bps=total_slippage_bps,
            total_cost_bps=total_slippage_bps,
            market_impact=market_impact
        )


class AdaptiveSlippageModel(BaseSlippageModel):
    """
    Adaptive slippage model
    
    Adjusts slippage based on current market conditions and regime.
    """
    
    def __init__(
        self,
        base_model: BaseSlippageModel,
        regime_multipliers: Optional[Dict[str, float]] = None,
        time_of_day_multipliers: Optional[Dict[int, float]] = None
    ):
        """
        Initialize adaptive slippage model
        
        Args:
            base_model: Base slippage model to adapt
            regime_multipliers: Multipliers for different market regimes
            time_of_day_multipliers: Multipliers for different hours
        """
        self.base_model = base_model
        
        self.regime_multipliers = regime_multipliers or {
            'bull': 0.8,
            'bear': 1.2,
            'sideways': 1.0,
            'volatile': 1.5
        }
        
        self.time_of_day_multipliers = time_of_day_multipliers or {
            # Asian session
            0: 1.3, 1: 1.3, 2: 1.3, 3: 1.3, 4: 1.3, 5: 1.3, 6: 1.3, 7: 1.3,
            # European session
            8: 1.0, 9: 0.8, 10: 0.8, 11: 0.8, 12: 0.8, 13: 0.8, 14: 0.8, 15: 1.0,
            # US session  
            16: 0.7, 17: 0.7, 18: 0.7, 19: 0.7, 20: 0.7, 21: 0.8, 22: 1.0, 23: 1.2
        }
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_slippage(
        self,
        order_size: float,
        price: float,
        volume: float,
        volatility: float,
        side: str,
        market_regime: str = 'sideways',
        timestamp: Optional[pd.Timestamp] = None
    ) -> SlippageResult:
        """Calculate adaptive slippage"""
        # Get base slippage
        base_result = self.base_model.calculate_slippage(
            order_size, price, volume, volatility, side
        )
        
        # Apply regime adjustment
        regime_multiplier = self.regime_multipliers.get(market_regime, 1.0)
        
        # Apply time-of-day adjustment
        hour_multiplier = 1.0
        if timestamp is not None:
            hour = timestamp.hour
            hour_multiplier = self.time_of_day_multipliers.get(hour, 1.0)
        
        # Combined multiplier
        total_multiplier = regime_multiplier * hour_multiplier
        
        # Adjust slippage
        adjusted_slippage_bps = base_result.slippage_bps * total_multiplier
        adjusted_market_impact = base_result.market_impact * total_multiplier
        
        # Recalculate executed price
        direction = 1 if side == 'buy' else -1
        slippage_price_change = price * (adjusted_slippage_bps / 10000) * direction
        executed_price = price + slippage_price_change
        
        return SlippageResult(
            original_price=price,
            executed_price=executed_price,
            slippage_bps=adjusted_slippage_bps,
            total_cost_bps=adjusted_slippage_bps,
            market_impact=adjusted_market_impact
        )


class VolumeProfileSlippageModel(BaseSlippageModel):
    """
    Volume profile based slippage model
    
    Uses historical volume profile to estimate slippage more accurately.
    """
    
    def __init__(
        self,
        volume_profile: Optional[pd.DataFrame] = None,
        depth_decay: float = 0.1,
        impact_recovery: float = 0.3
    ):
        """
        Initialize volume profile slippage model
        
        Args:
            volume_profile: Historical volume at different price levels
            depth_decay: Rate at which order book depth decays
            impact_recovery: Rate at which impact recovers
        """
        self.volume_profile = volume_profile
        self.depth_decay = depth_decay
        self.impact_recovery = impact_recovery
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_slippage(
        self,
        order_size: float,
        price: float,
        volume: float,
        volatility: float,
        side: str
    ) -> SlippageResult:
        """Calculate volume profile based slippage"""
        # Estimate order book depth
        if self.volume_profile is not None:
            # Use volume profile to estimate depth
            depth_estimate = self._estimate_depth_from_profile(price, volume)
        else:
            # Fallback to simple depth estimate
            depth_estimate = volume * 0.1  # Assume 10% of volume available at best price
        
        # Calculate how much of order book is consumed
        levels_consumed = 0
        remaining_size = order_size
        total_cost = 0
        current_price = price
        
        while remaining_size > 0 and levels_consumed < 10:  # Max 10 levels
            # Depth at current level
            level_depth = depth_estimate * np.exp(-levels_consumed * self.depth_decay)
            
            # Amount executed at this level
            executed_at_level = min(remaining_size, level_depth)
            
            # Price impact at this level
            price_impact = levels_consumed * 0.001 * price  # 0.1% per level
            if side == 'buy':
                execution_price = current_price + price_impact
            else:
                execution_price = current_price - price_impact
            
            # Accumulate cost
            total_cost += executed_at_level * execution_price
            remaining_size -= executed_at_level
            levels_consumed += 1
        
        # Calculate average execution price
        if order_size > 0:
            avg_execution_price = total_cost / order_size
        else:
            avg_execution_price = price
        
        # Calculate slippage in basis points
        slippage_bps = abs(avg_execution_price - price) / price * 10000
        
        return SlippageResult(
            original_price=price,
            executed_price=avg_execution_price,
            slippage_bps=slippage_bps,
            total_cost_bps=slippage_bps,
            market_impact=slippage_bps * 0.7  # Assume 70% is market impact
        )
    
    def _estimate_depth_from_profile(self, price: float, volume: float) -> float:
        """Estimate order book depth from volume profile"""
        if self.volume_profile is None:
            return volume * 0.1
        
        # Find nearest price level in profile
        price_levels = self.volume_profile.index
        nearest_idx = np.argmin(np.abs(price_levels - price))
        nearest_volume = self.volume_profile.iloc[nearest_idx].values[0]
        
        # Scale by recent volume
        scaling_factor = volume / (self.volume_profile['volume'].mean() + 1e-8)
        estimated_depth = nearest_volume * scaling_factor * 0.1
        
        return max(estimated_depth, volume * 0.05)  # Minimum 5% of volume


class SlippageModelFactory:
    """Factory for creating slippage models"""
    
    @staticmethod
    def create_model(
        model_type: str,
        **kwargs
    ) -> BaseSlippageModel:
        """
        Create slippage model of specified type
        
        Args:
            model_type: Type of model ('linear', 'sqrt', 'adaptive', 'volume_profile')
            **kwargs: Model-specific parameters
            
        Returns:
            Slippage model instance
        """
        models = {
            'linear': LinearSlippageModel,
            'sqrt': SquareRootSlippageModel,
            'adaptive': AdaptiveSlippageModel,
            'volume_profile': VolumeProfileSlippageModel
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return models[model_type](**kwargs)