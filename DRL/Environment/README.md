# CryptoTradingEnv - Deep Reinforcement Learning Trading Environment

A realistic, Gym-compatible cryptocurrency trading environment for training Deep Reinforcement Learning agents. Designed for compatibility with Stable-Baselines3 and RLlib without modification.

## Features

- ✅ **Full OpenAI Gym API compatibility** - Works with SB3, RLlib, and any RL framework
- ✅ **Realistic market simulation** - Transaction costs, slippage, market impact, and latency
- ✅ **Flexible observation spaces** - Vector or dictionary observations with technical indicators
- ✅ **Multiple action modes** - Discrete or continuous action spaces
- ✅ **Configurable rewards** - Multiple reward functions including risk-adjusted options
- ✅ **Domain randomization** - Improve generalization with parameter randomization
- ✅ **Margin trading support** - Leverage, funding rates, and liquidation mechanics
- ✅ **Comprehensive logging** - Trade history, metrics, and performance tracking
- ✅ **No lookahead bias** - All observations strictly use past data only

## Installation

```bash
# Install required dependencies
pip install gymnasium numpy pandas pyarrow

# For RL training
pip install stable-baselines3
# or
pip install ray[rllib]
```

## Quick Start

### Basic Usage

```python
from DRL.Environment import CryptoTradingEnv

# Create environment with default config
env = CryptoTradingEnv()

# Reset environment
obs, info = env.reset(seed=42)

# Run episode with random actions
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

# Get episode metrics
metrics = env.get_episode_summary()
print(f"Total Return: {metrics['total_return_pct']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

### Custom Configuration

```python
from DRL.Environment import CryptoTradingEnv, get_config

# Use preset configuration
config = get_config("conservative", window_size=100, seed=42)
env = CryptoTradingEnv(config)

# Or create custom config
custom_config = {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "window_size": 50,
    "action_mode": "continuous",
    "max_position_pct": 0.5,
    "leverage": 2.0,
    "fee_taker": 0.00075,
    "reward_type": "risk_adjusted",
    "seed": 42
}
env = CryptoTradingEnv(custom_config)
```

### Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from DRL.Environment import CryptoTradingEnv, get_config

# Create environment
config = get_config("default", window_size=50, seed=42)
env = CryptoTradingEnv(config)

# Create and train agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluate
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print(info['episode_metrics'])
```

## Configuration Reference

### Data Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_path` | str | None | Path to parquet data file |
| `symbol` | str | "BTCUSDT" | Trading pair symbol |
| `timeframe` | str | "1h" | Data timeframe (15m, 1h, 4h, 1d) |
| `start_date` | str | None | Filter data from this date |
| `end_date` | str | None | Filter data to this date |

### Observation Space

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | int | 50 | Number of historical bars |
| `observation_mode` | str | "vector" | "vector" or "dict" |
| `indicators` | list | [...] | Technical indicators to include |
| `normalization` | str | "zscore" | "zscore", "minmax", or "none" |
| `normalization_window` | int | 200 | Rolling window for normalization |

**Observation Vector Shape**: `(window_size * num_features + 7,)` where:
- Market features: `window_size * num_features` (OHLCV + indicators)
- Position state: 4 values (position, entry_price, unrealized_pnl, margin_used)
- Account state: 3 values (cash, holdings_value, portfolio_value)

**Observation Dict Shape**:
```python
{
    'market': (window_size, num_features),  # Historical market data
    'position': (4,),                        # Position state
    'account': (3,)                          # Account state
}
```

### Action Space

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action_mode` | str | "continuous" | "discrete" or "continuous" |
| `discrete_actions` | int | 4 | Number of discrete actions |
| `continuous_bounds` | tuple | (-1, 1) | Continuous action range |

**Discrete Actions**:
- 0: Hold (no change)
- 1: Buy/Long (enter maximum long position)
- 2: Sell/Short (enter maximum short position)
- 3: Exit (close all positions)

**Continuous Actions**:
- Single value in [-1, 1]
- -1: Maximum short position
- 0: Flat (no position)
- +1: Maximum long position

### Position Constraints

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_position_pct` | float | 1.0 | Max position as fraction of portfolio |
| `min_trade_size` | float | 0.01 | Minimum trade size |
| `lot_size` | float | 0.001 | Position size increment |
| `leverage` | float | 1.0 | Leverage multiplier |
| `max_leverage` | float | 3.0 | Maximum allowed leverage |

### Transaction Costs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fee_taker` | float | 0.00075 | Taker fee (0.075%) |
| `fee_maker` | float | 0.00025 | Maker fee (0.025%) |
| `slippage_coef` | float | 0.0001 | Slippage coefficient |
| `impact_coef` | float | 0.0005 | Market impact coefficient |
| `bid_ask_spread` | float | 0.0001 | Bid-ask spread (0.01%) |
| `funding_rate` | float | 0.0001 | Funding rate per interval |
| `borrow_rate_annual` | float | 0.05 | Annual borrowing rate (5%) |

### Execution Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `execution_type` | str | "market" | "market" or "limit" |
| `slippage_model` | str | "linear" | Slippage calculation method |
| `market_impact` | bool | True | Enable market impact |
| `latency_steps` | int | 0 | Execution delay in steps |
| `latency_random` | bool | False | Random latency |

### Reward Functions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reward_type` | str | "nav_delta_minus_tx" | Reward function type |
| `risk_lambda` | float | 0.01 | Risk penalty coefficient |
| `transaction_penalty` | float | 0.0 | Additional transaction penalty |
| `position_penalty` | float | 0.0 | Position size penalty |
| `reward_scale` | float | 1.0 | Reward scaling factor |

**Reward Types**:
- `"nav_delta"`: Simple change in portfolio value
- `"nav_delta_minus_tx"`: Portfolio change minus transaction costs
- `"risk_adjusted"`: PnL - λ × volatility
- `"sharpe"`: Sharpe-style reward (similar to nav_delta)
- `"sparse"`: Only reward at episode end

### Episode Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_episode_steps` | int | 1000 | Maximum steps per episode |
| `warmup_period` | int | 50 | Initial steps for indicators |
| `terminate_on_bankruptcy` | bool | True | End on low NAV |
| `bankruptcy_threshold` | float | 0.1 | NAV threshold (10% of initial) |
| `terminate_on_margin_call` | bool | True | End on margin call |

### Randomization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `randomize_start` | bool | True | Random episode start index |
| `randomize_fees` | bool | False | Random fee rates |
| `randomize_slippage` | bool | False | Random slippage |
| `randomize_latency` | bool | False | Random latency |
| `fee_range` | tuple | (0.0001, 0.001) | Fee randomization range |
| `slippage_range` | tuple | (5e-5, 5e-4) | Slippage range |
| `latency_range` | tuple | (0, 3) | Latency range (steps) |

### Initial Conditions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_balance` | float | 10000.0 | Starting capital (USDT) |
| `initial_position` | float | 0.0 | Starting position (BTC) |

### Logging & Metrics

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_trades` | bool | True | Enable trade logging |
| `log_frequency` | int | 1 | Log every N steps |
| `verbose` | int | 0 | Verbosity (0=silent, 1=episode, 2=step) |
| `compute_metrics` | bool | True | Compute episode metrics |

## Data Format

The environment expects parquet files with the following columns:

**Required Columns**:
- `timestamp`: DateTime index
- `open`: Open price
- `high`: High price
- `low`: Low price
- `close`: Close price
- `volume`: Trading volume

**Optional Columns**:
- `quote_volume`: Quote currency volume
- `num_trades`: Number of trades
- Order book data (if using `use_orderbook=True`)

### Data Location

Default data path structure:
```
EnvironmentData/
└── data/
    └── binance/
        └── {SYMBOL}/
            └── parquet/
                ├── 15m/
                │   └── 2018_01_01-2025_10_25.parquet
                ├── 1h/
                │   └── 2018_01_01-2025_10_25.parquet
                ├── 4h/
                │   └── 2018_01_01-2025_10_25.parquet
                └── 1d/
                    └── 2018_01_01-2025_10_25.parquet
```

## Technical Indicators

Supported indicators (via `indicators` config):

- `ema_{period}`: Exponential Moving Average
- `sma_{period}`: Simple Moving Average
- `rsi_{period}`: Relative Strength Index
- `macd`: MACD indicator (returns macd, signal, histogram)
- `bb_upper`, `bb_middle`, `bb_lower`: Bollinger Bands
- `atr_{period}`: Average True Range
- `stoch`: Stochastic Oscillator
- `obv`: On-Balance Volume
- `vwap`: Volume Weighted Average Price
- `returns_{period}`: Simple returns
- `log_returns_{period}`: Log returns
- `volatility_{period}`: Rolling volatility

Example:
```python
indicators = ["ema_10", "ema_30", "rsi_14", "macd", "atr_14", "bb_upper", "bb_lower"]
```

## Environment Presets

Convenient preset configurations:

### Default
Balanced configuration for general use.
```python
env = CryptoTradingEnv(get_config("default"))
```

### Minimal
Quick testing with fewer features.
```python
env = CryptoTradingEnv(get_config("minimal"))
```

### High Frequency
HFT-style with higher fees and latency.
```python
env = CryptoTradingEnv(get_config("high_frequency"))
```

### Conservative
Low risk, low leverage trading.
```python
env = CryptoTradingEnv(get_config("conservative"))
```

### Aggressive
High leverage with less risk penalty.
```python
env = CryptoTradingEnv(get_config("aggressive"))
```

## Episode Metrics

Computed at episode end (accessible via `info['episode_metrics']`):

- `total_return`: Total return (fraction)
- `total_return_pct`: Total return (percentage)
- `sharpe_ratio`: Annualized Sharpe ratio
- `max_drawdown`: Maximum drawdown (fraction)
- `max_drawdown_pct`: Maximum drawdown (percentage)
- `volatility`: Annualized volatility
- `total_trades`: Number of trades executed
- `total_fees`: Total fees paid
- `total_slippage`: Total slippage costs
- `win_rate`: Fraction of profitable steps
- `final_portfolio_value`: Final portfolio value
- `episode_length`: Number of steps

## Advanced Features

### Custom Reward Function

```python
def custom_reward(env):
    # Access environment state
    pnl = env.portfolio_value - env.prev_portfolio_value

    # Custom logic
    if env.position != 0:
        # Penalize holding positions
        pnl -= 0.001 * abs(env.position)

    return pnl

config = {
    "custom_reward_fn": custom_reward
}
env = CryptoTradingEnv(config)
```

### Custom Slippage Model

```python
def custom_slippage(size, price, data, step):
    # Custom slippage calculation
    volatility = data['returns'].iloc[max(0, step-20):step].std()
    return 0.0002 * abs(size) * volatility * price * np.sign(size)

config = {
    "custom_slippage_fn": custom_slippage
}
env = CryptoTradingEnv(config)
```

### Trade History Export

```python
# After episode completes
trade_df = env.get_trade_history()
trade_df.to_csv("trades.csv")

# Columns: step, timestamp, size, price, fee, slippage, impact, position, balance, portfolio_value
```

## Reproducibility

Ensure reproducible episodes:

```python
# Set seed in config
env = CryptoTradingEnv({"seed": 42})

# Or during reset
obs, info = env.reset(seed=42)

# Deterministic evaluation mode
config = get_config("default", mode="eval", seed=42)
env = CryptoTradingEnv(config)
```

## Train/Eval Modes

```python
# Training mode: randomization enabled
train_env = CryptoTradingEnv({"mode": "train", "randomize_start": True})

# Evaluation mode: deterministic, no randomization
eval_env = CryptoTradingEnv({"mode": "eval", "randomize_start": False})
```

## Validation & Testing

The environment includes comprehensive unit tests:

```bash
# Run tests
python -m pytest DRL/Environment/tests/

# Run specific test
python -m pytest DRL/Environment/tests/test_crypto_env.py::test_basic_workflow
```

## Performance Considerations

- **Memory**: Uses ~50-100MB for 1 year of hourly data with indicators
- **Speed**: ~1000-2000 steps/second on modern CPU
- **Streaming**: Set `stream_data=True` for very large datasets (not yet implemented)

## Troubleshooting

### ImportError: No module named 'gymnasium'

```bash
pip install gymnasium
```

### ImportError: pyarrow required

```bash
pip install pyarrow
```

### Data file not found

Ensure data exists at the expected path or specify `data_path` in config:

```python
config = {
    "data_path": "/path/to/your/data.parquet"
}
env = CryptoTradingEnv(config)
```

### Observation/Action space mismatch with SB3

Verify your policy matches the observation mode:
- Vector mode: Use `MlpPolicy`
- Dict mode: Use `MultiInputPolicy`

```python
from stable_baselines3 import PPO

# Vector observations
env = CryptoTradingEnv({"observation_mode": "vector"})
model = PPO("MlpPolicy", env)

# Dict observations
env = CryptoTradingEnv({"observation_mode": "dict"})
model = PPO("MultiInputPolicy", env)
```

## Examples

See `DRL/Environment/examples/` for complete examples:
- `basic_usage.py`: Simple environment usage
- `train_sb3.py`: Training with Stable-Baselines3
- `custom_config.py`: Custom configuration examples
- `walk_forward.py`: Walk-forward validation

## API Reference

### CryptoTradingEnv

Main environment class implementing Gym interface.

**Methods**:
- `__init__(config)`: Initialize environment
- `reset(seed, options)`: Reset to initial state
- `step(action)`: Execute action and advance one step
- `render(mode)`: Render current state
- `seed(seed)`: Set random seed
- `close()`: Clean up resources
- `get_trade_history()`: Get trade history as DataFrame
- `get_episode_summary()`: Get episode metrics

**Attributes**:
- `observation_space`: Gym observation space
- `action_space`: Gym action space
- `config`: Environment configuration
- `portfolio_value`: Current portfolio value
- `position`: Current position size
- `balance`: Current cash balance

## Contributing

Contributions welcome! Areas for improvement:
- Additional technical indicators
- More sophisticated execution models
- Order book simulation
- Multi-asset environments
- Vectorized environments for parallel training

## License

MIT License

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{crypto_trading_env,
  title = {CryptoTradingEnv: A Realistic Gym Environment for Cryptocurrency Trading},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/CryptoTrade}
}
```

## Support

For issues, questions, or contributions:
- GitHub Issues: [Report bugs or request features]
- Documentation: This README
- Examples: See `examples/` directory

---

**Version**: 1.0.0
**Last Updated**: 2025
