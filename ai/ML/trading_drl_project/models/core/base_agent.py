"""
Base Agent Abstract Class

Abstract base class for all DRL trading agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Space


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents
    
    This class defines the interface that all trading agents must implement.
    It provides common functionality and ensures consistency across different
    agent implementations.
    """
    
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        config: Dict[str, Any],
        device: Optional[str] = None
    ):
        """
        Initialize base agent
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space  
            config: Agent configuration dictionary
            device: PyTorch device (cuda/cpu)
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training state
        self.is_training = True
        self.step_count = 0
        self.episode_count = 0
        
        # Metrics tracking
        self.training_metrics = {}
        self.validation_metrics = {}
        
    @abstractmethod
    def predict(
        self, 
        observation: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Predict action given observation
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, additional_info)
        """
        pass
    
    @abstractmethod
    def learn(
        self, 
        total_timesteps: int,
        callback: Optional[Any] = None
    ) -> "BaseAgent":
        """
        Train the agent
        
        Args:
            total_timesteps: Number of training steps
            callback: Optional callback for monitoring
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save agent model and state
        
        Args:
            path: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load agent model and state
        
        Args:
            path: Path to load the model from
        """
        pass
    
    def set_training_mode(self, training: bool = True) -> None:
        """Set agent training mode"""
        self.is_training = training
        
    def set_evaluation_mode(self) -> None:
        """Set agent to evaluation mode"""
        self.set_training_mode(False)
        
    def reset(self) -> None:
        """Reset agent state for new episode"""
        self.episode_count += 1
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics"""
        return {
            'training': self.training_metrics.copy(),
            'validation': self.validation_metrics.copy(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'is_training': self.is_training
        }
    
    def update_metrics(self, metrics: Dict[str, Any], validation: bool = False) -> None:
        """
        Update agent metrics
        
        Args:
            metrics: Dictionary of metrics to update
            validation: Whether these are validation metrics
        """
        target_metrics = self.validation_metrics if validation else self.training_metrics
        target_metrics.update(metrics)
    
    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return self.config.copy()
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Update agent configuration"""
        self.config.update(config)
        
    @property
    def observation_dim(self) -> int:
        """Get observation dimension"""
        if hasattr(self.observation_space, 'shape'):
            return int(np.prod(self.observation_space.shape))
        return self.observation_space.n
    
    @property
    def action_dim(self) -> int:
        """Get action dimension"""
        if hasattr(self.action_space, 'shape'):
            return int(np.prod(self.action_space.shape))
        return self.action_space.n
    
    def __repr__(self) -> str:
        """String representation of agent"""
        return f"{self.__class__.__name__}(obs_dim={self.observation_dim}, action_dim={self.action_dim})"
