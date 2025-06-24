"""
Base Agent class for RL trading agents
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import os


class BaseAgent(ABC):
    """Base class for all RL agents"""
    
    def __init__(self, 
                 observation_space,
                 action_space,
                 config: Dict,
                 device: str = None):
        """
        Initialize base agent
        
        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            config: Agent configuration
            device: Device to use (cuda/cpu)
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize networks
        self._build_networks()
        
        # Training stats
        self.training_steps = 0
        self.episodes = 0
        
    @abstractmethod
    def _build_networks(self):
        """Build neural networks"""
        pass
    
    @abstractmethod
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given observation"""
        pass
    
    @abstractmethod
    def update(self, batch: Dict) -> Dict:
        """Update agent with batch of experiences"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save agent to file"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load agent from file"""
        pass
    
    def preprocess_observation(self, observation: np.ndarray) -> torch.Tensor:
        """Preprocess observation for network"""
        # Flatten observation if needed
        if len(observation.shape) == 3:
            # (batch, window, features) -> (batch, window * features)
            observation = observation.reshape(observation.shape[0], -1)
        elif len(observation.shape) == 2:
            # (window, features) -> (window * features)
            observation = observation.flatten()
        
        # Convert to tensor
        obs_tensor = torch.FloatTensor(observation).to(self.device)
        
        # Add batch dimension if needed
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        return obs_tensor
    
    def get_stats(self) -> Dict:
        """Get training statistics"""
        return {
            'training_steps': self.training_steps,
            'episodes': self.episodes
        }


class LightweightNetwork(nn.Module):
    """
    Lightweight neural network for limited hardware
    Uses techniques like batch normalization and dropout for efficiency
    """
    
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 hidden_sizes: list = [128, 64, 32],
                 activation: str = 'relu',
                 dropout: float = 0.1,
                 batch_norm: bool = True):
        """
        Initialize network
        
        Args:
            input_size: Input dimension
            output_size: Output dimension
            hidden_sizes: List of hidden layer sizes
            activation: Activation function
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(self.activation)
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)