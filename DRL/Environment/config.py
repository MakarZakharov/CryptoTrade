"""
Configuration module for CryptoTradingEnv.

Defines default parameters and configuration schema for the trading environment.
All parameters are customizable via config dict passed to environment constructor.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class EnvConfig:
    """
    Complete configuration specification for CryptoTradingEnv.

    All parameters have sensible defaults and can be overridden via constructor.
    """

    # ========== DATA INPUT ==========
    data_path: Optional[str] = None  # Path to parquet file or DataFrame
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"  # 15m, 1h, 4h, 1d
    start_date: Optional[str] = None  # Format: 'YYYY-MM-DD', None = use all data
    end_date: Optional[str] = None

    # ========== OBSERVATION SPACE ==========
    window_size: int = 50  # Number of historical bars in observation
    observation_mode: str = "vector"  # "vector" or "dict"

    # Technical indicators to include (list of indicator names)
    indicators: List[str] = field(default_factory=lambda: [
        "ema_10", "ema_30", "rsi_14", "macd", "atr_14", "bb_upper", "bb_lower"
    ])

    # Normalization method: "zscore", "minmax", "none"
    normalization: str = "zscore"
    normalization_window: int = 200  # Rolling window for normalization stats

    # Include order book features (if available in data)
    use_orderbook: bool = False
    orderbook_depth: int = 5  # Top-k levels

    # ========== ACTION SPACE ==========
    action_mode: str = "continuous"  # "discrete" or "continuous"

    # Discrete action space: {0: hold, 1: buy, 2: sell, 3: exit}
    discrete_actions: int = 4

    # Continuous action space: [-1, 1] where -1=full short, 0=flat, 1=full long
    continuous_bounds: tuple = (-1.0, 1.0)

    # ========== POSITION CONSTRAINTS ==========
    max_position_pct: float = 1.0  # Max position as fraction of portfolio value
    min_trade_size: float = 0.01  # Minimum trade size (in base currency)
    lot_size: float = 0.001  # Minimum increment for position sizing
    leverage: float = 1.0  # Leverage multiplier (1.0 = no leverage)
    max_leverage: float = 3.0

    # ========== EXECUTION MODEL ==========
    execution_type: str = "market"  # "market" or "limit"

    # Slippage model: callable or "linear"
    # Linear: slippage = slippage_coef * trade_size_pct * volatility
    slippage_model: str = "linear"
    slippage_coef: float = 0.0001

    # Market impact: price_shift = impact_coef * (trade_vol / avg_daily_vol)
    market_impact: bool = True
    impact_coef: float = 0.0005

    # Latency simulation (steps delay)
    latency_steps: int = 0  # 0 = immediate execution
    latency_random: bool = False  # Random latency between 0 and latency_steps

    # ========== TRANSACTION COSTS ==========
    fee_taker: float = 0.00075  # 0.075% taker fee (market orders)
    fee_maker: float = 0.00025  # 0.025% maker fee (limit orders)

    # Funding rate for perpetual futures (per 8 hours, annualized ~= 3*365*rate)
    funding_rate: float = 0.0001  # 0.01% per 8h
    funding_interval_steps: int = 8  # Apply every N steps (for hourly data)

    # Spread: bid_ask_spread as fraction of price
    bid_ask_spread: float = 0.0001  # 0.01% spread

    # Borrowing cost for shorts (annual rate)
    borrow_rate_annual: float = 0.05  # 5% annual

    # ========== MARGIN & RISK ==========
    use_margin: bool = False
    initial_margin_req: float = 0.1  # 10% initial margin (allows 10x leverage)
    maintenance_margin_req: float = 0.05  # 5% maintenance margin
    liquidation_penalty: float = 0.5  # 50% of remaining equity lost on liquidation

    # ========== REWARD FUNCTION ==========
    # Options: "nav_delta", "nav_delta_minus_tx", "risk_adjusted", "sharpe", "sparse"
    reward_type: str = "nav_delta_minus_tx"

    # Risk adjustment parameter (lambda)
    risk_lambda: float = 0.01  # Penalty coefficient for volatility/drawdown

    # Transaction penalty (eta * abs(trade_size))
    transaction_penalty: float = 0.0  # Additional penalty beyond actual costs

    # Position penalty (xi * abs(position))
    position_penalty: float = 0.0  # Discourage large positions

    # Reward scaling factor
    reward_scale: float = 1.0

    # ========== EPISODE CONTROL ==========
    max_episode_steps: int = 1000  # Maximum steps per episode
    warmup_period: int = 50  # Initial steps to build indicators (no trading)

    # Termination conditions
    terminate_on_bankruptcy: bool = True
    bankruptcy_threshold: float = 0.1  # NAV < initial_balance * threshold

    terminate_on_margin_call: bool = True

    # ========== INITIAL CONDITIONS ==========
    initial_balance: float = 10000.0  # Starting capital in quote currency (USDT)
    initial_position: float = 0.0  # Starting position in base currency (BTC)

    # ========== RANDOMIZATION ==========
    randomize_start: bool = True  # Random episode start index
    randomize_fees: bool = False  # Randomize fee rates per episode
    randomize_slippage: bool = False
    randomize_latency: bool = False

    # Randomization ranges (uniform sampling)
    fee_range: tuple = (0.0001, 0.001)  # Min/max fee rates
    slippage_range: tuple = (0.00005, 0.0005)
    latency_range: tuple = (0, 3)  # Steps

    # ========== TRAIN/EVAL MODE ==========
    mode: str = "train"  # "train" or "eval"
    eval_deterministic: bool = True  # Disable randomization in eval mode

    # ========== LOGGING & METRICS ==========
    log_trades: bool = True
    log_frequency: int = 1  # Log every N steps
    verbose: int = 0  # 0=silent, 1=episode summary, 2=step details

    # Metrics to compute at episode end
    compute_metrics: bool = True

    # ========== PERFORMANCE ==========
    stream_data: bool = False  # Stream from disk vs load all in memory
    vectorized: bool = False  # Support vectorized environments (future)

    # ========== REPRODUCIBILITY ==========
    seed: Optional[int] = None

    # ========== CUSTOM FUNCTIONS ==========
    # Allow custom reward/slippage/impact functions
    custom_reward_fn: Optional[Callable] = None
    custom_slippage_fn: Optional[Callable] = None
    custom_impact_fn: Optional[Callable] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.window_size > 0, "window_size must be positive"
        assert self.timeframe in ["15m", "1h", "4h", "1d"], f"Invalid timeframe: {self.timeframe}"
        assert self.observation_mode in ["vector", "dict"], f"Invalid observation_mode: {self.observation_mode}"
        assert self.action_mode in ["discrete", "continuous"], f"Invalid action_mode: {self.action_mode}"
        assert self.normalization in ["zscore", "minmax", "none"], f"Invalid normalization: {self.normalization}"
        assert self.reward_type in [
            "nav_delta", "nav_delta_minus_tx", "risk_adjusted", "sharpe", "sparse"
        ], f"Invalid reward_type: {self.reward_type}"
        assert 0 <= self.max_position_pct <= 10, "max_position_pct must be in [0, 10]"
        assert self.leverage >= 1.0, "leverage must be >= 1.0"
        assert self.initial_balance > 0, "initial_balance must be positive"
        assert self.mode in ["train", "eval"], f"Invalid mode: {self.mode}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnvConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


# Default configurations for common use cases
DEFAULT_CONFIG = EnvConfig()

MINIMAL_CONFIG = EnvConfig(
    window_size=20,
    indicators=["ema_10", "rsi_14"],
    max_episode_steps=500,
    verbose=1
)

HIGH_FREQUENCY_CONFIG = EnvConfig(
    timeframe="15m",
    window_size=100,
    fee_taker=0.001,  # Higher fees for HFT
    slippage_coef=0.0005,
    latency_steps=1,  # 15min latency
    max_episode_steps=2000
)

CONSERVATIVE_CONFIG = EnvConfig(
    leverage=1.0,
    max_position_pct=0.5,
    risk_lambda=0.05,  # Strong risk penalty
    position_penalty=0.001,
    transaction_penalty=0.001
)

AGGRESSIVE_CONFIG = EnvConfig(
    leverage=3.0,
    max_position_pct=1.0,
    use_margin=True,
    risk_lambda=0.0,
    fee_taker=0.00075
)


def get_config(preset: str = "default", **kwargs) -> EnvConfig:
    """
    Get environment configuration with optional overrides.

    Args:
        preset: Configuration preset name
            - "default": Balanced configuration
            - "minimal": Simple setup for quick testing
            - "high_frequency": HFT-style configuration
            - "conservative": Low risk, low leverage
            - "aggressive": High leverage, less risk penalty
        **kwargs: Override any config parameters

    Returns:
        EnvConfig instance

    Example:
        >>> config = get_config("conservative", window_size=100, seed=42)
    """
    presets = {
        "default": DEFAULT_CONFIG,
        "minimal": MINIMAL_CONFIG,
        "high_frequency": HIGH_FREQUENCY_CONFIG,
        "conservative": CONSERVATIVE_CONFIG,
        "aggressive": AGGRESSIVE_CONFIG
    }

    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

    base_config = presets[preset]
    config_dict = base_config.to_dict()
    config_dict.update(kwargs)

    return EnvConfig.from_dict(config_dict)
