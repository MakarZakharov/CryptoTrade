"""DRL агенты для торговли криптовалютами - Enterprise Edition."""

from .reinforcement_learning_algorithms import (
    BaseReinforcementLearningAgent,
    ProximalPolicyOptimizationAgent
)

# Backward compatibility
BaseAgent = BaseReinforcementLearningAgent
PPOAgent = ProximalPolicyOptimizationAgent

__all__ = [
    'BaseReinforcementLearningAgent',
    'ProximalPolicyOptimizationAgent',
    # Backward compatibility
    'BaseAgent',
    'PPOAgent'
]