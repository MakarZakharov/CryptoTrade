"""Алгоритмы обучения с подкреплением для торговли."""

from .base_reinforcement_learning_agent import BaseReinforcementLearningAgent
from .policy_gradient_agents import ProximalPolicyOptimizationAgent

# Backward compatibility aliases
BaseAgent = BaseReinforcementLearningAgent
PPOAgent = ProximalPolicyOptimizationAgent

__all__ = [
    'BaseReinforcementLearningAgent',
    'ProximalPolicyOptimizationAgent',
    # Backward compatibility
    'BaseAgent', 
    'PPOAgent'
]