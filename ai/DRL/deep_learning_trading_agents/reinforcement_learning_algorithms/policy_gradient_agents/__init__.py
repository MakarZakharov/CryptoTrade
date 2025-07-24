"""Policy gradient агенты для торговли."""

from .proximal_policy_optimization_agent import ProximalPolicyOptimizationAgent

# Backward compatibility
PPOAgent = ProximalPolicyOptimizationAgent

__all__ = [
    'ProximalPolicyOptimizationAgent',
    # Backward compatibility
    'PPOAgent'
]