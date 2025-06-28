"""
Market Dynamics Modeling

Models various market dynamics and regime changes for realistic trading simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging


class MarketRegime(Enum):
    """Market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class MarketState:
    """Current market state"""
    regime: MarketRegime
    volatility: float
    trend_strength: float
    momentum: float
    liquidity: float
    timestamp: pd.Timestamp


class MarketDynamicsModel:
    """
    Market dynamics model for realistic trading simulation
    
    Simulates various market conditions including regime changes,
    volatility clustering, and momentum effects.
    """
    
    def __init__(
        self,
        regime_persistence: float = 0.95,
        volatility_clustering: float = 0.8,
        momentum_halflife: int = 20,
        liquidity_impact: float = 0.1
    ):
        """
        Initialize market dynamics model
        
        Args:
            regime_persistence: Probability of staying in same regime
            volatility_clustering: Volatility clustering coefficient
            momentum_halflife: Half-life of momentum decay (in periods)
            liquidity_impact: Impact of liquidity on price movements
        """
        self.regime_persistence = regime_persistence
        self.volatility_clustering = volatility_clustering
        self.momentum_halflife = momentum_halflife
        self.liquidity_impact = liquidity_impact
        
        self.logger = logging.getLogger(__name__)
        
        # Current state
        self.current_regime = MarketRegime.SIDEWAYS
        self.current_volatility = 0.02
        self.current_momentum = 0.0
        self.current_liquidity = 1.0
        
        # Regime transition probabilities
        self.regime_transitions = self._initialize_regime_transitions()
        
    def _initialize_regime_transitions(self) -> Dict[MarketRegime, Dict[MarketRegime, float]]:
        """Initialize regime transition probability matrix"""
        return {
            MarketRegime.BULL: {
                MarketRegime.BULL: 0.85,
                MarketRegime.SIDEWAYS: 0.10,
                MarketRegime.BEAR: 0.03,
                MarketRegime.VOLATILE: 0.02
            },
            MarketRegime.BEAR: {
                MarketRegime.BEAR: 0.80,
                MarketRegime.SIDEWAYS: 0.12,
                MarketRegime.BULL: 0.05,
                MarketRegime.VOLATILE: 0.03
            },
            MarketRegime.SIDEWAYS: {
                MarketRegime.SIDEWAYS: 0.70,
                MarketRegime.BULL: 0.15,
                MarketRegime.BEAR: 0.10,
                MarketRegime.VOLATILE: 0.05
            },
            MarketRegime.VOLATILE: {
                MarketRegime.VOLATILE: 0.60,
                MarketRegime.SIDEWAYS: 0.25,
                MarketRegime.BULL: 0.08,
                MarketRegime.BEAR: 0.07
            }
        }
    
    def simulate_price_impact(
        self, 
        base_return: float, 
        trade_size: float, 
        market_state: MarketState
    ) -> float:
        """
        Simulate market impact on price returns
        
        Args:
            base_return: Base return without market effects
            trade_size: Size of trade (normalized)
            market_state: Current market state
            
        Returns:
            Adjusted return with market impact
        """
        # Volatility impact
        volatility_noise = np.random.normal(0, market_state.volatility)
        
        # Momentum impact
        momentum_impact = market_state.momentum * 0.1
        
        # Liquidity impact
        liquidity_impact = trade_size * self.liquidity_impact / market_state.liquidity
        
        # Regime-specific adjustments
        regime_adjustments = {
            MarketRegime.BULL: 0.1,
            MarketRegime.BEAR: -0.1,
            MarketRegime.SIDEWAYS: 0.0,
            MarketRegime.VOLATILE: np.random.normal(0, 0.2)
        }
        
        regime_impact = regime_adjustments.get(market_state.regime, 0.0)
        
        # Combine all effects
        adjusted_return = (
            base_return + 
            volatility_noise + 
            momentum_impact + 
            liquidity_impact + 
            regime_impact
        )
        
        return adjusted_return