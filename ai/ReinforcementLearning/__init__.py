"""
Reinforcement Learning Trading System
Optimized for limited computational resources

Features:
- Multiple RL algorithms (PPO, DQN, A2C)
- Comprehensive technical indicators (50+ indicators)
- Multi-timeframe support
- Reward optimization for maximum profit with minimal drawdowns
- GPU acceleration support
- Efficient memory usage
- Concurrent distributed training (multiple participants simultaneously)
- Resume training at any time without losing progress
"""

from .environment.trading_env import TradingEnvironment
from .agents.ppo_agent import PPOAgent
from .training.trainer import RLTrainer
from .features.feature_extractor import FeatureExtractor
from .distributed_concurrent import ConcurrentDistributedTrainer, ConcurrentRLTrainer
from .config import *

__all__ = [
    'TradingEnvironment',
    'PPOAgent',
    'RLTrainer',
    'FeatureExtractor',
    'ConcurrentDistributedTrainer',
    'ConcurrentRLTrainer'
]