"""
PPO (Proximal Policy Optimization) Agent
Optimized for limited computational resources
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional
from .base_agent import BaseAgent, LightweightNetwork
from ..config import AGENT_CONFIG, NETWORK_CONFIG, DEVICE


class PPOAgent(BaseAgent):
    """
    PPO agent for trading
    Uses clipped objective for stable training
    """
    
    def __init__(self, 
                 observation_space,
                 action_space,
                 config: Dict = None):
        """
        Initialize PPO agent
        
        Args:
            observation_space: Gym observation space
            action_space: Gym action space
            config: PPO configuration
        """
        # Merge with default config
        ppo_config = AGENT_CONFIG['ppo'].copy()
        if config:
            ppo_config.update(config)
        
        # Action space parameters (needed before super().__init__ which calls _build_networks)
        self.action_dim = action_space.shape[0]
        
        super().__init__(observation_space, action_space, ppo_config, DEVICE)
        
        # PPO specific parameters
        self.clip_ratio = ppo_config['clip_ratio']
        self.value_loss_coef = ppo_config['value_loss_coef']
        self.entropy_coef = ppo_config['entropy_coef']
        self.max_grad_norm = ppo_config['max_grad_norm']
        self.ppo_epochs = ppo_config['epochs']
        self.mini_batch_size = ppo_config['mini_batch_size']
        
        # Action space parameters
        self.action_low = torch.FloatTensor(action_space.low).to(self.device)
        self.action_high = torch.FloatTensor(action_space.high).to(self.device)
        
    def _build_networks(self):
        """Build actor and critic networks"""
        # Calculate input size
        obs_shape = self.observation_space.shape
        if len(obs_shape) == 2:
            input_size = obs_shape[0] * obs_shape[1]
        else:
            input_size = obs_shape[0]
        
        # Actor network (policy)
        self.actor = LightweightNetwork(
            input_size=input_size,
            output_size=self.action_dim * 2,  # Mean and log_std
            hidden_sizes=NETWORK_CONFIG['hidden_sizes'],
            activation=NETWORK_CONFIG['activation'],
            dropout=NETWORK_CONFIG['dropout'],
            batch_norm=NETWORK_CONFIG['batch_norm']
        ).to(self.device)
        
        # Critic network (value function)
        self.critic = LightweightNetwork(
            input_size=input_size,
            output_size=1,
            hidden_sizes=NETWORK_CONFIG['hidden_sizes'],
            activation=NETWORK_CONFIG['activation'],
            dropout=NETWORK_CONFIG['dropout'],
            batch_norm=NETWORK_CONFIG['batch_norm']
        ).to(self.device)
        
        # Optimizers with lower learning rate for stability
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=self.config.get('learning_rate', 1e-4),  # Reduced from 3e-4
            eps=1e-5
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=self.config.get('learning_rate', 1e-4),  # Reduced from 3e-4
            eps=1e-5
        )
        
        # Enable anomaly detection in debug mode
        torch.autograd.set_detect_anomaly(True)
    
    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action using actor network
        
        Args:
            observation: Current observation
            deterministic: If True, use mean action
            
        Returns:
            Action array
        """
        # Set network to evaluation mode for inference
        self.actor.eval()
        
        with torch.no_grad():
            obs_tensor = self.preprocess_observation(observation)
            
            # Get action distribution parameters
            actor_output = self.actor(obs_tensor)
            mean = actor_output[:, :self.action_dim]
            log_std = actor_output[:, self.action_dim:]
            std = torch.exp(log_std)
            
            if deterministic:
                action = mean
            else:
                # Sample from normal distribution
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
            
            # Apply tanh squashing and scale to action space
            action = torch.tanh(action)
            action = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
            
            return action.cpu().numpy().squeeze()
    
    def compute_advantages(self, 
                          rewards: torch.Tensor,
                          values: torch.Tensor,
                          dones: torch.Tensor,
                          gamma: float = 0.99,
                          gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Reward tensor
            values: Value predictions
            dones: Done flags
            gamma: Discount factor
            gae_lambda: GAE lambda
            
        Returns:
            advantages, returns
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Work backwards through trajectory
        gae = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(self, batch: Dict) -> Dict:
        """
        Update agent using PPO algorithm
        
        Args:
            batch: Dictionary containing:
                - observations: Batch of observations
                - actions: Batch of actions taken
                - rewards: Batch of rewards
                - next_observations: Batch of next observations
                - dones: Batch of done flags
                - old_log_probs: Log probabilities from behavior policy
                
        Returns:
            Dictionary of training metrics
        """
        # Convert to tensors
        observations = torch.FloatTensor(batch['observations']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['old_log_probs']).to(self.device)
        
        # Flatten observations if needed
        if len(observations.shape) == 3:
            batch_size = observations.shape[0]
            observations = observations.reshape(batch_size, -1)
        
        # Compute values and advantages
        with torch.no_grad():
            values = self.critic(observations).squeeze()
            advantages, returns = self.compute_advantages(rewards, values, dones)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Set networks to training mode for updates
        self.actor.train()
        self.critic.train()
        
        # Training metrics
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Create mini-batches
            indices = np.arange(len(observations))
            np.random.shuffle(indices)
            
            for start in range(0, len(observations), self.mini_batch_size):
                end = min(start + self.mini_batch_size, len(observations))
                mb_indices = indices[start:end]
                
                mb_obs = observations[mb_indices]
                mb_actions = actions[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                
                # Forward pass
                actor_output = self.actor(mb_obs)
                mean = actor_output[:, :self.action_dim]
                log_std = actor_output[:, self.action_dim:]
                
                # Clamp log_std to prevent numerical instability
                log_std = torch.clamp(log_std, min=-20, max=2)
                std = torch.exp(log_std)
                
                # Check for NaN values and replace with zeros
                mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
                std = torch.where(torch.isnan(std), torch.ones_like(std), std)
                std = torch.clamp(std, min=1e-6)  # Ensure std is not too small
                
                # Compute log probabilities
                dist = torch.distributions.Normal(mean, std)
                
                # Unscale actions for log prob calculation
                unscaled_actions = (mb_actions - self.action_low) / (self.action_high - self.action_low) * 2.0 - 1.0
                unscaled_actions = torch.atanh(torch.clamp(unscaled_actions, -0.999, 0.999))
                
                log_probs = dist.log_prob(unscaled_actions).sum(dim=1)
                
                # Compute ratio
                ratio = torch.exp(log_probs - mb_old_log_probs)
                
                # Clipped objective
                obj1 = ratio * mb_advantages
                obj2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                actor_loss = -torch.min(obj1, obj2).mean()
                
                # Value loss
                values = self.critic(mb_obs).squeeze()
                critic_loss = nn.MSELoss()(values, mb_returns)
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Total loss
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy
                
                # Backward pass
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                # Update
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Track metrics
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
        
        # Update training steps
        self.training_steps += len(observations) * self.ppo_epochs
        
        # Return metrics
        n_updates = self.ppo_epochs * (len(observations) // self.mini_batch_size)
        return {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'training_steps': self.training_steps
        }
    
    def save(self, path: str):
        """Save agent to file"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_steps': self.training_steps,
            'episodes': self.episodes,
            'config': self.config
        }, path)
    
    def load(self, path: str):
        """Load agent from file"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
        self.episodes = checkpoint['episodes']