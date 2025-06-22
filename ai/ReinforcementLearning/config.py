"""
Configuration for Reinforcement Learning Trading System
Optimized for limited computational resources
"""

# System Configuration
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically detect CUDA availability
BATCH_SIZE = 32  # Small batch size for limited memory
LEARNING_RATE = 3e-4
GAMMA = 0.99
BUFFER_SIZE = 10000  # Smaller buffer for memory efficiency

# Trading Configuration
TRADING_FEES = 0.001  # 0.1% trading fee
INITIAL_BALANCE = 10000
MIN_TRADE_AMOUNT = 0.01
MAX_POSITION_SIZE = 0.95  # Max 95% of balance in one position

# Timeframes supported (in minutes)
TIMEFRAMES = {
    '1m': 1,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '4h': 240,
    '1d': 1440
}

# Technical Indicators Configuration
INDICATORS_CONFIG = {
    # Trend Indicators
    'sma': [5, 10, 20, 50, 100, 200],
    'ema': [5, 10, 20, 50, 100, 200],
    'wma': [10, 20, 50],
    
    # Momentum Indicators
    'rsi': [7, 14, 21],
    'stochastic': [(5, 3), (14, 3), (21, 3)],
    'williams_r': [14, 21],
    'roc': [10, 20],
    'momentum': [10, 20],
    
    # Volatility Indicators
    'bollinger_bands': [(20, 2), (20, 2.5), (50, 2)],
    'atr': [7, 14, 21],
    'keltner_channels': [(20, 2), (50, 2)],
    'donchian_channels': [20, 50],
    
    # Volume Indicators
    'obv': True,
    'mfi': [14, 21],
    'vwap': True,
    'volume_sma': [10, 20],
    'cmf': [20],
    'fi': [13],
    
    # Trend Strength
    'adx': [14, 21],
    'cci': [14, 20],
    'aroon': [14, 25],
    'psar': [(0.02, 0.2)],
    
    # Market Structure
    'pivot_points': True,
    'support_resistance': 3,  # Number of levels
    'fibonacci_retracement': True,
    
    # Custom Indicators
    'macd': [(12, 26, 9), (5, 35, 5)],
    'ichimoku': True,
    'supertrend': [(10, 3), (20, 5)],
    'heikin_ashi': True
}

# State Space Configuration
STATE_WINDOW = 50  # Look back 50 periods
FEATURE_SCALING = True
NORMALIZE_REWARDS = True

# Training Configuration
EPISODES = 1000
STEPS_PER_EPISODE = 2000
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 50
CHECKPOINT_FREQUENCY = 10  # Save model every 10 episodes

# Reward Configuration
REWARD_CONFIG = {
    'profit_weight': 1.0,
    'drawdown_penalty': 2.0,  # Heavy penalty for drawdowns
    'sharpe_bonus': 0.5,
    'trade_cost_penalty': 0.1,
    'holding_penalty': 0.001,  # Small penalty for holding
    'win_rate_bonus': 0.3
}

# Agent Configuration
AGENT_CONFIG = {
    'ppo': {
        'clip_ratio': 0.2,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'epochs': 4,  # Fewer epochs for faster training
        'mini_batch_size': 8
    },
    'dqn': {
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'target_update_freq': 100,
        'double_dqn': True,
        'dueling': True  # Dueling DQN for better performance
    },
    'a2c': {
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'gae_lambda': 0.95
    }
}

# Lightweight Neural Network Architecture
NETWORK_CONFIG = {
    'hidden_sizes': [128, 64, 32],  # Smaller network for faster training
    'activation': 'relu',
    'dropout': 0.1,
    'batch_norm': True
}