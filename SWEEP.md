# SWEEP.md - CryptoTrade Project Reference

This file contains common commands, configuration details, and project structure information for the CryptoTrade DRL Environment.

---

## 🐍 Python Environment

### Virtual Environment
- **Path**: `/home/kali/PycharmProjects/CryptoTrade/venv/bin/python`
- **Python Version**: 3.13.9
- **Activation**: `source /home/kali/PycharmProjects/CryptoTrade/venv/bin/activate`

### PyCharm Configuration
- **Project Interpreter**: `/home/kali/PycharmProjects/CryptoTrade/venv/bin/python`
- **Set via**: File → Settings → Project → Python Interpreter

---

## 📦 Dependencies

### Core Dependencies (Installed)
- `gymnasium>=0.29.0` - RL environment framework
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `pyarrow>=14.0.0` - Parquet file support
- `stable-baselines3>=2.0.0` - RL training algorithms
- `torch>=2.0.0` - PyTorch backend
- `sb3-contrib>=2.0.0` - Additional SB3 algorithms

### Install Commands
```bash
# Full installation
/home/kali/PycharmProjects/CryptoTrade/venv/bin/pip install -r requirements.txt

# Minimal installation
/home/kali/PycharmProjects/CryptoTrade/venv/bin/pip install -r requirements-minimal.txt

# Core RL packages only
/home/kali/PycharmProjects/CryptoTrade/venv/bin/pip install stable-baselines3 torch sb3-contrib
```

---

## 🧪 Testing

### Run All Tests
```bash
/home/kali/PycharmProjects/CryptoTrade/venv/bin/python -m pytest /home/kali/PycharmProjects/CryptoTrade/DRL/Environment/tests/ -v
```

### Run Specific Test File
```bash
/home/kali/PycharmProjects/CryptoTrade/venv/bin/python -m pytest /home/kali/PycharmProjects/CryptoTrade/DRL/Environment/tests/test_crypto_env.py -v
```

### Run Specific Test
```bash
/home/kali/PycharmProjects/CryptoTrade/venv/bin/python -m pytest /home/kali/PycharmProjects/CryptoTrade/DRL/Environment/tests/test_crypto_env.py::TestCryptoTradingEnv::test_basic_trading_workflow -v
```

### Run Tests with Coverage
```bash
/home/kali/PycharmProjects/CryptoTrade/venv/bin/python -m pytest /home/kali/PycharmProjects/CryptoTrade/DRL/Environment/tests/ --cov=DRL.Environment --cov-report=html
```

---

## 🏃 Running Examples

### Basic Usage Example
```bash
/home/kali/PycharmProjects/CryptoTrade/venv/bin/python /home/kali/PycharmProjects/CryptoTrade/DRL/Environment/examples/basic_usage.py
```

### Train with Stable-Baselines3
```bash
/home/kali/PycharmProjects/CryptoTrade/venv/bin/python /home/kali/PycharmProjects/CryptoTrade/DRL/Environment/examples/train_sb3.py
```

### Custom Configuration Example
```bash
/home/kali/PycharmProjects/CryptoTrade/venv/bin/python /home/kali/PycharmProjects/CryptoTrade/DRL/Environment/examples/custom_config.py
```

### Evaluate Agent
```bash
/home/kali/PycharmProjects/CryptoTrade/venv/bin/python /home/kali/PycharmProjects/CryptoTrade/DRL/Environment/examples/evaluate_agent.py
```

---

## 📁 Project Structure

```
CryptoTrade/
├── DRL/
│   └── Environment/
│       ├── __init__.py              # Package initialization
│       ├── crypto_trading_env.py    # Main environment class
│       ├── config.py                # Configuration system
│       ├── indicators.py            # Technical indicators
│       ├── README.md                # Environment documentation
│       ├── SPECIFICATION.md         # Technical specification
│       ├── examples/
│       │   ├── basic_usage.py       # Basic example
│       │   ├── train_sb3.py         # SB3 training example
│       │   ├── custom_config.py     # Custom config example
│       │   └── evaluate_agent.py    # Evaluation example
│       └── tests/
│           └── test_crypto_env.py   # Unit tests (28 tests)
├── EnvironmentData/
│   └── data/
│       └── binance/
│           └── BTCUSDT/
│               └── parquet/
│                   └── 1h/
│                       └── 2018_01_01-2025_10_25.parquet
├── venv/                            # Virtual environment
├── requirements.txt                 # Full dependencies
├── requirements-minimal.txt         # Minimal dependencies
└── SWEEP.md                         # This file
```

---

## 🔧 Common Development Tasks

### Check Environment Status
```bash
/home/kali/PycharmProjects/CryptoTrade/venv/bin/python -c "from DRL.Environment import CryptoTradingEnv, get_config; env = CryptoTradingEnv(get_config('minimal', seed=42)); obs, info = env.reset(); print('✓ Environment working'); print(f'Observation shape: {obs.shape}')"
```

### Verify Dependencies
```bash
/home/kali/PycharmProjects/CryptoTrade/venv/bin/python -c "import gymnasium, numpy, pandas, pyarrow, stable_baselines3, torch; print('✓ All dependencies installed')"
```

### Count Lines of Code
```bash
wc -l /home/kali/PycharmProjects/CryptoTrade/DRL/Environment/*.py /home/kali/PycharmProjects/CryptoTrade/DRL/Environment/tests/*.py
```

### Check for Linter Errors (if flake8 installed)
```bash
/home/kali/PycharmProjects/CryptoTrade/venv/bin/python -m flake8 /home/kali/PycharmProjects/CryptoTrade/DRL/Environment/*.py --max-line-length=120 --extend-ignore=E501,W503
```

---

## 📊 Data Information

### Data Location
```
/home/kali/PycharmProjects/CryptoTrade/EnvironmentData/data/binance/BTCUSDT/parquet/1h/
```

### Data Stats
- **Symbol**: BTCUSDT
- **Timeframe**: 1h
- **Data Points**: 68,904 candlesticks
- **Date Range**: 2018-01-01 to 2025-11-16
- **File Size**: ~4.1 MB
- **Format**: Parquet (requires pyarrow)

### Supported Timeframes
- 15m (15 minutes)
- 1h (1 hour)
- 4h (4 hours)
- 1d (1 day)

---

## 🎯 Environment Quick Reference

### Create Environment
```python
from DRL.Environment import CryptoTradingEnv, get_config

# Use preset
env = CryptoTradingEnv(get_config("default"))

# Or custom config
config = {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "window_size": 50,
    "action_mode": "continuous",
    "seed": 42
}
env = CryptoTradingEnv(config)
```

### Available Presets
- `default` - Balanced configuration
- `minimal` - Quick testing setup
- `conservative` - Low risk, low leverage
- `aggressive` - High leverage, less risk penalty
- `high_frequency` - HFT-style configuration

### Observation Modes
- `vector` - Flat array (use with MlpPolicy)
- `dict` - Structured dict (use with MultiInputPolicy)

### Action Modes
- `discrete` - {0: hold, 1: buy, 2: sell, 3: exit}
- `continuous` - [-1, 1] target position

### Reward Types
- `nav_delta` - Simple portfolio change
- `nav_delta_minus_tx` - Portfolio change minus costs (default)
- `risk_adjusted` - PnL minus volatility penalty
- `sharpe` - Sharpe-style reward
- `sparse` - Only reward at episode end

---

## 🐛 Troubleshooting

### PyCharm Shows Import Errors
**Problem**: IDE shows "No module named 'gymnasium'"
**Solution**: Configure PyCharm to use venv interpreter:
1. File → Settings → Project → Python Interpreter
2. Select: `/home/kali/PycharmProjects/CryptoTrade/venv/bin/python`
3. Restart PyCharm if needed

### Module Not Found When Running
**Problem**: `ModuleNotFoundError: No module named 'XXX'`
**Solution**: Use the venv Python explicitly:
```bash
/home/kali/PycharmProjects/CryptoTrade/venv/bin/python your_script.py
```

### Data File Not Found
**Problem**: `FileNotFoundError: Data file not found`
**Solution**: Specify data path in config:
```python
config = {
    "data_path": "/path/to/your/data.parquet"
}
```

---

## 📚 Documentation

- **README**: `/home/kali/PycharmProjects/CryptoTrade/DRL/Environment/README.md`
- **Specification**: `/home/kali/PycharmProjects/CryptoTrade/DRL/Environment/SPECIFICATION.md`
- **Config Reference**: See `config.py` for all 50+ parameters

---

## ✅ Test Status

**Last Test Run**: All 28 tests PASSING ✅
- Environment initialization ✅
- Observation/Action spaces ✅
- Trading mechanics ✅
- Position constraints ✅
- Episode termination ✅
- Reproducibility ✅
- Technical indicators ✅
- Multiple reward functions ✅

---

## 🎓 Code Style Preferences

### Naming Conventions
- Classes: `PascalCase` (e.g., `CryptoTradingEnv`)
- Functions/Methods: `snake_case` (e.g., `get_config`)
- Private Methods: `_snake_case` (e.g., `_calculate_reward`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_CONFIG`)

### Import Style
```python
# Standard library
import os
import sys

# Third-party
import numpy as np
import pandas as pd
import gymnasium as gym

# Local
from .config import EnvConfig, get_config
from .indicators import compute_indicators
```

### Docstring Format
- Use triple-quoted strings
- Include Args, Returns, Example sections
- Keep line length reasonable (~80-120 chars)

---

## 🔄 Version Information

- **Environment Version**: 1.0.0
- **Python**: 3.13.9
- **Gymnasium**: 1.2.2
- **NumPy**: 2.3.4
- **Pandas**: 2.3.3
- **PyArrow**: 22.0.0
- **PyTorch**: 2.9.1+cpu
- **Stable-Baselines3**: 2.7.0
- **SB3-Contrib**: 2.7.0

---

## 📋 Environment Health Check

**Last Check**: 2025-01-16

### ✅ Core Components (All Working)
- ✅ Environment loads successfully
- ✅ All 28 unit tests passing (5.22s)
- ✅ Data files present (68,904 data points)
- ✅ Technical indicators working
- ✅ Trading mechanics functional
- ✅ Examples run successfully

### ✅ RL Training Libraries (All Installed)
- ✅ **stable-baselines3** - v2.7.0 (RL training algorithms)
- ✅ **torch** - v2.9.1+cpu (PyTorch backend)
- ✅ **sb3-contrib** - v2.7.0 (additional algorithms: RecurrentPPO, TQC, QRDQN, etc.)

**Status**: All RL training dependencies are installed and working! You can now train agents using PPO, A2C, DQN, SAC, TD3, and more.

---

**Last Updated**: 2025-01-16
**Maintained For**: AI Assistant Context & Developer Reference
