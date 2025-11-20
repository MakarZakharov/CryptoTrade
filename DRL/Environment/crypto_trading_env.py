"""
CryptoTradingEnv - OpenAI Gym compatible cryptocurrency trading environment.

This environment is designed for training Deep Reinforcement Learning agents
to trade cryptocurrency. It implements realistic market dynamics including:
- Transaction costs (fees, slippage, market impact)
- Position management with leverage and margin
- Configurable observation and action spaces
- Multiple reward function options
- Domain randomization for robustness
- Comprehensive logging and metrics

Compatible with Stable-Baselines3 and RLlib without modifications.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
import os
from collections import deque
import warnings

from .config import EnvConfig, get_config
from .indicators import compute_indicators, normalize_features


class CryptoTradingEnv(gym.Env):
    """
    OpenAI Gym environment for cryptocurrency trading.

    Implements the Gym API with realistic market simulation including
    transaction costs, slippage, market impact, and margin trading.

    Attributes:
        observation_space: Gym space defining observation structure
        action_space: Gym space defining action structure
        metadata: Environment metadata (render modes, etc.)
    """

    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, config: Union[Dict, EnvConfig, None] = None):
        """
        Initialize the trading environment.

        Args:
            config: Configuration dictionary or EnvConfig object.
                   If None, uses default configuration.

        Example:
            >>> env = CryptoTradingEnv({"window_size": 50, "seed": 42})
            >>> env = CryptoTradingEnv(EnvConfig(timeframe="1h"))
        """
        super().__init__()

        # Parse configuration
        if config is None:
            self.config = get_config("default")
        elif isinstance(config, dict):
            self.config = EnvConfig(**config)
        elif isinstance(config, EnvConfig):
            self.config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        # Set random seed
        if self.config.seed is not None:
            self.seed(self.config.seed)

        # Load and prepare data
        self._load_data()
        self._prepare_features()

        # Define observation and action spaces
        self._setup_spaces()

        # Initialize state variables
        self._reset_state()

        # Trading history and metrics
        self.trade_history = []
        self.episode_history = []

        # Render mode
        self._render_mode = None

    def _load_data(self):
        """Load historical market data from parquet file or DataFrame."""
        if self.config.data_path is None:
            # Use default path structure
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            data_path = os.path.join(
                base_path,
                "EnvironmentData", "data", "binance",
                self.config.symbol, "parquet",
                self.config.timeframe,
                f"2018_01_01-2025_10_25.parquet"
            )
            self.config.data_path = data_path

        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(
                f"Data file not found: {self.config.data_path}\n"
                f"Please provide valid data_path in config or ensure data exists."
            )

        # Load data
        if self.config.verbose > 0:
            print(f"Loading data from: {self.config.data_path}")

        try:
            df = pd.read_parquet(self.config.data_path)
        except ImportError:
            raise ImportError(
                "pyarrow is required to read parquet files. "
                "Install with: pip install pyarrow"
            )

        # Validate required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Set timestamp as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        # Filter by date range if specified
        if self.config.start_date:
            df = df[df.index >= self.config.start_date]
        if self.config.end_date:
            df = df[df.index <= self.config.end_date]

        if len(df) == 0:
            raise ValueError("No data available after date filtering")

        self.raw_data = df
        if self.config.verbose > 0:
            print(f"Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")

    def _prepare_features(self):
        """Compute technical indicators and prepare features."""
        # Compute technical indicators
        df_with_indicators = compute_indicators(
            self.raw_data,
            self.config.indicators
        )

        # Add basic returns
        df_with_indicators['returns'] = df_with_indicators['close'].pct_change()
        df_with_indicators['log_returns'] = np.log(
            df_with_indicators['close'] / df_with_indicators['close'].shift(1)
        )

        # Normalize features if specified
        if self.config.normalization != "none":
            # Don't normalize price columns (open, high, low, close)
            # but normalize indicators and volume
            cols_to_normalize = [
                col for col in df_with_indicators.columns
                if col not in ['open', 'high', 'low', 'close'] and
                df_with_indicators[col].dtype in [np.float64, np.float32, np.int64, np.int32]
            ]

            df_with_indicators = normalize_features(
                df_with_indicators,
                method=self.config.normalization,
                window=self.config.normalization_window,
                columns=cols_to_normalize
            )

        # Fill NaN values (from indicators at the beginning)
        self.data = df_with_indicators.fillna(0)

        # Store feature column names for observation construction
        self.feature_columns = [
            col for col in self.data.columns
            if col not in ['timestamp']
        ]

        if self.config.verbose > 0:
            print(f"Prepared {len(self.feature_columns)} features including indicators")

    def _setup_spaces(self):
        """Define observation and action spaces."""
        # Action space
        if self.config.action_mode == "discrete":
            # Discrete: {0: hold, 1: buy, 2: sell, 3: exit/flatten}
            self.action_space = spaces.Discrete(self.config.discrete_actions)

        elif self.config.action_mode == "continuous":
            # Continuous: [-1, 1] representing target position fraction
            self.action_space = spaces.Box(
                low=self.config.continuous_bounds[0],
                high=self.config.continuous_bounds[1],
                shape=(1,),
                dtype=np.float32
            )

        # Observation space
        if self.config.observation_mode == "vector":
            # Calculate observation size
            obs_size = self._calculate_observation_size()
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_size,),
                dtype=np.float32
            )

        elif self.config.observation_mode == "dict":
            # Dictionary observation with separate components
            self.observation_space = spaces.Dict({
                'market': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.config.window_size, len(self.feature_columns)),
                    dtype=np.float32
                ),
                'position': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(4,),  # position, entry_price, unrealized_pnl, margin_used
                    dtype=np.float32
                ),
                'account': spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(3,),  # cash, holdings_value, total_value
                    dtype=np.float32
                )
            })

    def _calculate_observation_size(self) -> int:
        """Calculate the total size of the observation vector."""
        # Market features: window_size * num_features
        market_size = self.config.window_size * len(self.feature_columns)

        # Position state: position, entry_price, unrealized_pnl, margin_used
        position_size = 4

        # Account state: cash, holdings_value, total_value
        account_size = 3

        return market_size + position_size + account_size

    def _reset_state(self):
        """Initialize/reset internal state variables."""
        self.current_step = 0
        self.done = False

        # Account state
        self.balance = self.config.initial_balance
        self.position = self.config.initial_position
        self.entry_price = 0.0
        self.total_fees_paid = 0.0
        self.total_slippage_paid = 0.0

        # Performance tracking
        self.initial_portfolio_value = self.config.initial_balance
        self.portfolio_value = self.config.initial_balance
        self.peak_portfolio_value = self.config.initial_balance
        self.total_trades = 0

        # Reward tracking
        self.prev_portfolio_value = self.config.initial_balance
        self.cumulative_reward = 0.0

        # Episode randomization
        if self.config.randomize_start and self.config.mode == "train":
            max_start = len(self.data) - self.config.max_episode_steps - self.config.window_size - 1
            self.episode_start_idx = self.np_random.integers(
                self.config.warmup_period,
                max(self.config.warmup_period + 1, max_start)
            )
        else:
            self.episode_start_idx = self.config.warmup_period

        self.current_step = self.episode_start_idx

        # Randomize environment parameters if configured
        if self.config.mode == "train":
            if self.config.randomize_fees:
                self.fee_rate = self.np_random.uniform(*self.config.fee_range)
            else:
                self.fee_rate = self.config.fee_taker

            if self.config.randomize_slippage:
                self.slippage_coef = self.np_random.uniform(*self.config.slippage_range)
            else:
                self.slippage_coef = self.config.slippage_coef

            if self.config.randomize_latency:
                self.latency = self.np_random.integers(*self.config.latency_range)
            else:
                self.latency = self.config.latency_steps
        else:
            # Eval mode: use default values
            self.fee_rate = self.config.fee_taker
            self.slippage_coef = self.config.slippage_coef
            self.latency = self.config.latency_steps

        # Pending orders (for latency simulation)
        self.pending_orders = deque()

        # Clear trade history for new episode
        self.trade_history = []

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (not used currently)

        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        self._reset_state()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (discrete int or continuous array)

        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated
            info: Additional information dictionary
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before calling step().")

        # Execute pending orders (latency simulation)
        self._execute_pending_orders()

        # Process action and create order
        self._process_action(action)

        # Advance time
        self.current_step += 1

        # Update portfolio value
        current_price = self._get_current_price()
        self._update_portfolio_value(current_price)

        # Apply funding/borrowing costs if holding position
        self._apply_holding_costs()

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.episode_start_idx + self.config.max_episode_steps

        self.done = terminated or truncated

        # Get next observation
        observation = self._get_observation()
        info = self._get_info()

        # Update previous portfolio value for next step
        self.prev_portfolio_value = self.portfolio_value
        self.cumulative_reward += reward

        # Log step if configured
        if self.config.log_trades and self.config.verbose >= 2:
            self._log_step(action, reward)

        return observation, reward, bool(terminated), bool(truncated), info

    def _process_action(self, action: Union[int, np.ndarray]):
        """
        Process action and execute trade.

        Args:
            action: Action from agent
        """
        current_price = self._get_current_price()

        # Decode action to target position
        if self.config.action_mode == "discrete":
            target_position = self._decode_discrete_action(action)
        else:
            target_position = self._decode_continuous_action(action)

        # Calculate trade size
        position_change = target_position - self.position

        # Apply lot size constraints
        position_change = self._apply_lot_size(position_change)

        # Skip if no significant change
        if abs(position_change) < self.config.min_trade_size:
            return

        # Execute trade (with latency if configured)
        if self.latency > 0:
            # Queue order for future execution
            execution_step = self.current_step + self.latency
            self.pending_orders.append({
                'step': execution_step,
                'size': position_change,
                'price': current_price
            })
        else:
            # Immediate execution
            self._execute_trade(position_change, current_price)

    def _decode_discrete_action(self, action: int) -> float:
        """
        Decode discrete action to target position.

        Args:
            action: Discrete action (0=hold, 1=buy, 2=sell, 3=exit)

        Returns:
            Target position in base currency
        """
        max_position = (self.portfolio_value * self.config.max_position_pct /
                       self._get_current_price())

        if action == 0:  # Hold
            return self.position
        elif action == 1:  # Buy/Long
            return max_position * self.config.leverage
        elif action == 2:  # Sell/Short
            return -max_position * self.config.leverage
        elif action == 3:  # Exit/Flatten
            return 0.0
        else:
            return self.position

    def _decode_continuous_action(self, action: np.ndarray) -> float:
        """
        Decode continuous action to target position.

        Args:
            action: Continuous action in [-1, 1]

        Returns:
            Target position in base currency
        """
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)

        # Clip to bounds
        action_value = np.clip(
            action_value,
            self.config.continuous_bounds[0],
            self.config.continuous_bounds[1]
        )

        # Map to target position
        max_position = (self.portfolio_value * self.config.max_position_pct /
                       self._get_current_price())

        target_position = action_value * max_position * self.config.leverage

        return target_position

    def _apply_lot_size(self, size: float) -> float:
        """Round trade size to nearest lot size."""
        if self.config.lot_size > 0:
            return np.round(size / self.config.lot_size) * self.config.lot_size
        return size

    def _execute_pending_orders(self):
        """Execute any pending orders that have reached their execution time."""
        while self.pending_orders and self.pending_orders[0]['step'] <= self.current_step:
            order = self.pending_orders.popleft()
            # Execute at current price (price may have moved)
            self._execute_trade(order['size'], self._get_current_price())

    def _execute_trade(self, size: float, price: float):
        """
        Execute a trade with realistic transaction costs.

        Args:
            size: Position change (positive=buy, negative=sell)
            price: Execution price
        """
        if abs(size) < self.config.min_trade_size:
            return

        # Calculate slippage
        slippage = self._calculate_slippage(size, price)

        # Calculate market impact
        impact = self._calculate_market_impact(size, price) if self.config.market_impact else 0

        # Effective execution price
        execution_price = price + slippage + impact

        # Calculate fees
        trade_value = abs(size * execution_price)
        fee = trade_value * self.fee_rate

        # Calculate spread cost
        spread_cost = trade_value * (self.config.bid_ask_spread / 2)

        # Total transaction cost
        total_cost = fee + spread_cost

        # Update position
        old_position = self.position
        self.position += size

        # Update balance
        self.balance -= (size * execution_price + total_cost)

        # Update entry price (weighted average for partial positions)
        if self.position != 0:
            if np.sign(old_position) == np.sign(self.position):
                # Adding to position
                total_value = old_position * self.entry_price + size * execution_price
                self.entry_price = total_value / self.position if self.position != 0 else 0
            else:
                # Reversing or closing
                self.entry_price = execution_price
        else:
            self.entry_price = 0

        # Track costs
        self.total_fees_paid += fee
        self.total_slippage_paid += abs(slippage * size)
        self.total_trades += 1

        # Log trade
        if self.config.log_trades:
            self.trade_history.append({
                'step': self.current_step,
                'timestamp': self.data.index[self.current_step],
                'size': size,
                'price': execution_price,
                'fee': fee,
                'slippage': slippage,
                'impact': impact,
                'position': self.position,
                'balance': self.balance,
                'portfolio_value': self.portfolio_value
            })

    def _calculate_slippage(self, size: float, price: float) -> float:
        """
        Calculate slippage based on trade size and volatility.

        Args:
            size: Trade size
            price: Current price

        Returns:
            Slippage in price units
        """
        if self.config.custom_slippage_fn:
            return self.config.custom_slippage_fn(size, price, self.data, self.current_step)

        # Linear slippage model
        # slippage = coef * |size| * volatility
        if self.current_step >= 20:
            recent_returns = self.data['returns'].iloc[self.current_step-20:self.current_step]
            volatility = recent_returns.std()
        else:
            volatility = 0.01  # Default

        slippage = self.slippage_coef * abs(size) * volatility * price
        return slippage * np.sign(size)  # Positive for buys, negative for sells

    def _calculate_market_impact(self, size: float, price: float) -> float:
        """
        Calculate market impact based on trade size relative to volume.

        Args:
            size: Trade size
            price: Current price

        Returns:
            Price impact
        """
        if self.config.custom_impact_fn:
            return self.config.custom_impact_fn(size, price, self.data, self.current_step)

        # Impact proportional to trade size relative to average volume
        if self.current_step >= 20:
            avg_volume = self.data['volume'].iloc[self.current_step-20:self.current_step].mean()
        else:
            avg_volume = self.data['volume'].iloc[self.current_step]

        volume_fraction = abs(size) / (avg_volume + 1e-8)
        impact = self.config.impact_coef * volume_fraction * price

        return impact * np.sign(size)

    def _apply_holding_costs(self):
        """Apply funding rates and borrowing costs for holding positions."""
        if self.position == 0:
            return

        current_price = self._get_current_price()
        position_value = abs(self.position * current_price)

        # Funding rate (for futures)
        if self.config.funding_rate > 0:
            if self.current_step % self.config.funding_interval_steps == 0:
                funding_cost = position_value * self.config.funding_rate
                if self.position < 0:  # Short positions pay funding
                    self.balance -= funding_cost

        # Borrowing cost for shorts (annual rate, apply per step)
        if self.position < 0 and self.config.borrow_rate_annual > 0:
            # Convert annual rate to per-step rate
            steps_per_year = 365 * 24 if self.config.timeframe == "1h" else 365
            step_rate = self.config.borrow_rate_annual / steps_per_year
            borrow_cost = position_value * step_rate
            self.balance -= borrow_cost

    def _update_portfolio_value(self, current_price: float):
        """Update portfolio value based on current price."""
        holdings_value = self.position * current_price
        self.portfolio_value = self.balance + holdings_value

        # Update peak for drawdown calculation
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on configured reward function.

        Returns:
            Reward value
        """
        if self.config.custom_reward_fn:
            return self.config.custom_reward_fn(self)

        if self.config.reward_type == "nav_delta":
            # Simple change in portfolio value
            reward = self.portfolio_value - self.prev_portfolio_value

        elif self.config.reward_type == "nav_delta_minus_tx":
            # Change in portfolio value minus transaction costs
            reward = self.portfolio_value - self.prev_portfolio_value

        elif self.config.reward_type == "risk_adjusted":
            # PnL minus volatility penalty
            pnl = self.portfolio_value - self.prev_portfolio_value

            # Calculate recent volatility
            if len(self.trade_history) >= 10:
                recent_pnls = [
                    t['portfolio_value'] - self.trade_history[i-1]['portfolio_value']
                    if i > 0 else 0
                    for i, t in enumerate(self.trade_history[-10:])
                ]
                volatility = np.std(recent_pnls)
            else:
                volatility = 0

            reward = pnl - self.config.risk_lambda * volatility

        elif self.config.reward_type == "sharpe":
            # Sharpe-like ratio (computed at episode end)
            reward = self.portfolio_value - self.prev_portfolio_value
            # Sharpe ratio computed in episode metrics

        elif self.config.reward_type == "sparse":
            # Only reward at episode end
            reward = 0.0
            if self.done:
                reward = (self.portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value

        else:
            reward = self.portfolio_value - self.prev_portfolio_value

        # Apply additional penalties
        if self.config.transaction_penalty > 0 and len(self.trade_history) > 0:
            if self.trade_history[-1]['step'] == self.current_step:
                trade_size = abs(self.trade_history[-1]['size'])
                reward -= self.config.transaction_penalty * trade_size

        if self.config.position_penalty > 0:
            reward -= self.config.position_penalty * abs(self.position)

        # Scale reward
        reward *= self.config.reward_scale

        return reward

    def _check_termination(self) -> bool:
        """
        Check if episode should terminate.

        Returns:
            True if episode should end
        """
        # Bankruptcy check
        if self.config.terminate_on_bankruptcy:
            if self.portfolio_value <= self.initial_portfolio_value * self.config.bankruptcy_threshold:
                if self.config.verbose > 0:
                    print(f"Episode terminated: Bankruptcy (NAV={self.portfolio_value:.2f})")
                return True

        # Margin call check
        if self.config.terminate_on_margin_call and self.config.use_margin:
            if self._check_margin_call():
                if self.config.verbose > 0:
                    print(f"Episode terminated: Margin call")
                return True

        # Reached end of data
        if self.current_step >= len(self.data) - 1:
            if self.config.verbose > 0:
                print(f"Episode terminated: End of data reached")
            return True

        return False

    def _check_margin_call(self) -> bool:
        """Check if account is in margin call."""
        if not self.config.use_margin or self.position == 0:
            return False

        current_price = self._get_current_price()
        position_value = abs(self.position * current_price)
        margin_used = position_value / self.config.leverage

        # Calculate margin ratio: equity / margin_used
        margin_ratio = self.portfolio_value / (margin_used + 1e-8)

        if margin_ratio < self.config.maintenance_margin_req:
            # Liquidation
            liquidation_cost = self.portfolio_value * self.config.liquidation_penalty
            self.balance -= liquidation_cost
            self.position = 0
            self.portfolio_value = self.balance
            return True

        return False

    def _get_current_price(self) -> float:
        """Get current market price (close price)."""
        return float(self.data['close'].iloc[self.current_step])

    def _get_observation(self) -> Union[np.ndarray, Dict]:
        """
        Construct observation for the agent.

        Returns:
            Observation (vector or dict based on config)
        """
        # Get market window
        start_idx = max(0, self.current_step - self.config.window_size + 1)
        end_idx = self.current_step + 1

        market_window = self.data[self.feature_columns].iloc[start_idx:end_idx].values

        # Pad if at beginning of episode
        if market_window.shape[0] < self.config.window_size:
            padding = np.zeros((self.config.window_size - market_window.shape[0], len(self.feature_columns)))
            market_window = np.vstack([padding, market_window])

        # Position state
        current_price = self._get_current_price()
        unrealized_pnl = (current_price - self.entry_price) * self.position if self.entry_price > 0 else 0
        margin_used = abs(self.position * current_price) / self.config.leverage if self.config.use_margin else 0

        position_state = np.array([
            self.position,
            self.entry_price,
            unrealized_pnl,
            margin_used
        ], dtype=np.float32)

        # Account state
        holdings_value = self.position * current_price
        account_state = np.array([
            self.balance,
            holdings_value,
            self.portfolio_value
        ], dtype=np.float32)

        # Combine based on observation mode
        if self.config.observation_mode == "vector":
            # Flatten market window and concatenate all states
            market_flat = market_window.flatten()
            observation = np.concatenate([market_flat, position_state, account_state]).astype(np.float32)
            return observation

        elif self.config.observation_mode == "dict":
            return {
                'market': market_window.astype(np.float32),
                'position': position_state,
                'account': account_state
            }

    def _get_info(self) -> Dict:
        """
        Get additional information dictionary.

        Returns:
            Info dict with current state and metrics
        """
        current_price = self._get_current_price()

        info = {
            'step': self.current_step,
            'timestamp': str(self.data.index[self.current_step]),
            'price': current_price,
            'position': self.position,
            'balance': self.balance,
            'portfolio_value': self.portfolio_value,
            'total_trades': self.total_trades,
            'total_fees': self.total_fees_paid,
            'total_slippage': self.total_slippage_paid,
        }

        # Add episode metrics if episode is done
        if self.done:
            info['episode_metrics'] = self._compute_episode_metrics()

        return info

    def _compute_episode_metrics(self) -> Dict:
        """
        Compute comprehensive performance metrics for the episode.

        Returns:
            Dictionary of performance metrics
        """
        # Total return
        total_return = (self.portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value

        # Calculate step-by-step returns
        if len(self.trade_history) > 1:
            step_returns = []
            for i in range(1, len(self.trade_history)):
                ret = (self.trade_history[i]['portfolio_value'] -
                      self.trade_history[i-1]['portfolio_value']) / self.trade_history[i-1]['portfolio_value']
                step_returns.append(ret)

            step_returns = np.array(step_returns)

            # Sharpe ratio (annualized)
            if len(step_returns) > 1 and step_returns.std() > 0:
                steps_per_year = 365 * 24 if self.config.timeframe == "1h" else 365
                sharpe_ratio = (step_returns.mean() / step_returns.std()) * np.sqrt(steps_per_year)
            else:
                sharpe_ratio = 0.0

            # Max drawdown
            portfolio_values = [t['portfolio_value'] for t in self.trade_history]
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

            # Volatility (annualized)
            steps_per_year = 365 * 24 if self.config.timeframe == "1h" else 365
            volatility = step_returns.std() * np.sqrt(steps_per_year)

            # Win rate
            winning_trades = np.sum(step_returns > 0)
            win_rate = winning_trades / len(step_returns) if len(step_returns) > 0 else 0

        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            volatility = 0.0
            win_rate = 0.0

        metrics = {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'volatility': volatility,
            'total_trades': self.total_trades,
            'total_fees': self.total_fees_paid,
            'total_slippage': self.total_slippage_paid,
            'win_rate': win_rate,
            'final_portfolio_value': self.portfolio_value,
            'episode_length': len(self.trade_history)
        }

        if self.config.verbose > 0:
            print(f"\n{'='*60}")
            print(f"EPISODE METRICS")
            print(f"{'='*60}")
            print(f"Total Return: {metrics['total_return_pct']:.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            print(f"Volatility: {metrics['volatility']:.2f}")
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
            print(f"Total Fees: ${metrics['total_fees']:.2f}")
            print(f"Final Portfolio Value: ${metrics['final_portfolio_value']:.2f}")
            print(f"{'='*60}\n")

        return metrics

    def _log_step(self, action, reward):
        """Log step information (verbose mode)."""
        print(f"Step {self.current_step} | "
              f"Action: {action} | "
              f"Price: ${self._get_current_price():.2f} | "
              f"Position: {self.position:.4f} | "
              f"Balance: ${self.balance:.2f} | "
              f"Portfolio: ${self.portfolio_value:.2f} | "
              f"Reward: {reward:.4f}")

    def render(self, mode='human'):
        """
        Render the environment.

        Args:
            mode: Render mode ('human' or 'ansi')
        """
        if mode == 'human' or mode == 'ansi':
            current_price = self._get_current_price()
            output = f"\n{'='*60}\n"
            output += f"Step: {self.current_step} | Time: {self.data.index[self.current_step]}\n"
            output += f"{'='*60}\n"
            output += f"Price: ${current_price:.2f}\n"
            output += f"Position: {self.position:.4f} BTC\n"
            output += f"Balance: ${self.balance:.2f}\n"
            output += f"Portfolio Value: ${self.portfolio_value:.2f}\n"
            output += f"Total Return: {((self.portfolio_value/self.initial_portfolio_value - 1) * 100):.2f}%\n"
            output += f"Total Trades: {self.total_trades}\n"
            output += f"{'='*60}\n"

            if mode == 'human':
                print(output)
            return output

    def seed(self, seed: Optional[int] = None):
        """
        Set random seed for reproducibility.

        Args:
            seed: Random seed
        """
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self):
        """Clean up environment resources."""
        pass

    def get_trade_history(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame.

        Returns:
            DataFrame with all executed trades
        """
        if not self.trade_history:
            return pd.DataFrame()

        return pd.DataFrame(self.trade_history)

    def get_episode_summary(self) -> Dict:
        """
        Get summary of current/last episode.

        Returns:
            Dictionary with episode summary
        """
        return self._compute_episode_metrics()
