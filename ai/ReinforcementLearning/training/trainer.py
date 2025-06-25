"""
Training system for RL agents
Optimized for limited computational resources with efficient memory usage
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Callable
from collections import deque
import os
import json
from datetime import datetime
import time

from ..environment.trading_env import TradingEnvironment
from ..agents.ppo_agent import PPOAgent
from ..config import *


class RLTrainer:
    """
    Trainer for RL trading agents
    Handles training loop, evaluation, and model saving
    """
    
    def __init__(self,
                 agent,
                 train_env: TradingEnvironment,
                 val_env: Optional[TradingEnvironment] = None,
                 config: Dict = None):
        """
        Initialize trainer
        
        Args:
            agent: RL agent to train
            train_env: Training environment
            val_env: Validation environment
            config: Training configuration
        """
        self.agent = agent
        self.train_env = train_env
        self.val_env = val_env
        self.config = config or {}
        
        # Training parameters
        self.episodes = self.config.get('episodes', EPISODES)
        self.steps_per_episode = self.config.get('steps_per_episode', STEPS_PER_EPISODE)
        self.checkpoint_freq = self.config.get('checkpoint_frequency', CHECKPOINT_FREQUENCY)
        self.early_stopping_patience = self.config.get('early_stopping_patience', EARLY_STOPPING_PATIENCE)
        
        # Buffer for PPO
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.steps_per_episode,
            observation_space=train_env.observation_space,
            action_space=train_env.action_space
        )
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_profits': [],
            'episode_drawdowns': [],
            'episode_sharpe': [],
            'val_rewards': [],
            'val_profits': [],
            'training_time': [],
            'metrics': []
        }
        
        # Best model tracking
        self.best_val_reward = -np.inf
        self.patience_counter = 0
        
        # Create directories
        self.model_dir = os.path.join('models', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train(self, callbacks: Optional[List[Callable]] = None):
        """
        Main training loop
        
        Args:
            callbacks: List of callback functions
        """
        print(f"Starting training on {self.agent.device}")
        print(f"Training for {self.episodes} episodes")
        print("-" * 50)
        
        for episode in range(self.episodes):
            start_time = time.time()
            
            # Collect rollout
            rollout_metrics = self._collect_rollout()
            
            # Update agent
            if self.rollout_buffer.is_full():
                update_metrics = self._update_agent()
                self.rollout_buffer.reset()
            else:
                update_metrics = {}
            
            # Validation
            if episode % 10 == 0 and self.val_env is not None:
                val_metrics = self._validate()
                self._check_early_stopping(val_metrics['total_reward'])
            else:
                val_metrics = {}
            
            # Record metrics
            episode_time = time.time() - start_time
            self._record_metrics(episode, rollout_metrics, update_metrics, val_metrics, episode_time)
            
            # Print progress
            if episode % 10 == 0:
                self._print_progress(episode, rollout_metrics, val_metrics)
            
            # Save checkpoint
            if episode % self.checkpoint_freq == 0:
                self._save_checkpoint(episode)
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, episode)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered at episode {episode}")
                break
        
        # Save final model
        self._save_checkpoint(episode, final=True)
        self._save_training_history()
        
        print("\nTraining completed!")
        self._print_final_summary()
    
    def _collect_rollout(self) -> Dict:
        """Collect experience rollout"""
        obs = self.train_env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.steps_per_episode):
            # Get action from agent
            action = self.agent.act(obs, deterministic=False)
            
            # Store old log prob for PPO
            with torch.no_grad():
                # Set network to evaluation mode for inference
                self.agent.actor.eval()
                
                obs_tensor = self.agent.preprocess_observation(obs)
                actor_output = self.agent.actor(obs_tensor)
                mean = actor_output[:, :self.agent.action_dim]
                log_std = actor_output[:, self.agent.action_dim:]
                std = torch.exp(log_std)
                
                dist = torch.distributions.Normal(mean, std)
                action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.agent.device)
                
                # Unscale action for log prob
                unscaled_action = (action_tensor - self.agent.action_low) / (self.agent.action_high - self.agent.action_low) * 2.0 - 1.0
                unscaled_action = torch.atanh(torch.clamp(unscaled_action, -0.999, 0.999))
                
                old_log_prob = dist.log_prob(unscaled_action).sum(dim=1).cpu().numpy()
            
            # Step environment
            next_obs, reward, done, info = self.train_env.step(action)
            
            # Store transition
            self.rollout_buffer.add(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                old_log_prob=old_log_prob[0]
            )
            
            episode_reward += reward
            episode_length += 1
            
            if done:
                obs = self.train_env.reset()
            else:
                obs = next_obs
        
        # Get final metrics
        metrics = self.train_env.get_metrics()
        metrics['episode_reward'] = episode_reward
        metrics['episode_length'] = episode_length
        
        return metrics
    
    def _update_agent(self) -> Dict:
        """Update agent with collected experience"""
        # Get batch from buffer
        batch = self.rollout_buffer.get_batch()
        
        # Update agent
        update_metrics = self.agent.update(batch)
        
        return update_metrics
    
    def _validate(self) -> Dict:
        """Run validation episode"""
        obs = self.val_env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Act deterministically during validation
            action = self.agent.act(obs, deterministic=True)
            obs, reward, done, info = self.val_env.step(action)
            total_reward += reward
        
        # Get final metrics
        metrics = self.val_env.get_metrics()
        metrics['total_reward'] = total_reward
        
        return metrics
    
    def _check_early_stopping(self, val_reward: float):
        """Check early stopping condition"""
        if val_reward > self.best_val_reward:
            self.best_val_reward = val_reward
            self.patience_counter = 0
            
            # Save best model
            best_path = os.path.join(self.model_dir, 'best_model.pth')
            self.agent.save(best_path)
        else:
            self.patience_counter += 1
    
    def _record_metrics(self, episode: int, rollout_metrics: Dict, 
                       update_metrics: Dict, val_metrics: Dict, episode_time: float):
        """Record training metrics"""
        self.training_history['episode_rewards'].append(rollout_metrics['episode_reward'])
        self.training_history['episode_profits'].append(rollout_metrics.get('total_profit', 0))
        self.training_history['episode_drawdowns'].append(rollout_metrics.get('max_drawdown', 0))
        self.training_history['episode_sharpe'].append(rollout_metrics.get('sharpe_ratio', 0))
        self.training_history['training_time'].append(episode_time)
        
        if val_metrics:
            self.training_history['val_rewards'].append(val_metrics['total_reward'])
            self.training_history['val_profits'].append(val_metrics.get('total_profit', 0))
        
        # Detailed metrics
        metrics_entry = {
            'episode': episode,
            'rollout': rollout_metrics,
            'update': update_metrics,
            'validation': val_metrics,
            'time': episode_time
        }
        self.training_history['metrics'].append(metrics_entry)
    
    def _print_progress(self, episode: int, rollout_metrics: Dict, val_metrics: Dict):
        """Print training progress"""
        print(f"\nEpisode {episode}/{self.episodes}")
        print(f"  Reward: {rollout_metrics['episode_reward']:.4f}")
        print(f"  Profit: ${rollout_metrics.get('total_profit', 0):.2f}")
        print(f"  Drawdown: {rollout_metrics.get('max_drawdown', 0):.2%}")
        print(f"  Sharpe: {rollout_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Win Rate: {rollout_metrics.get('win_rate', 0):.2%}")
        print(f"  Trades: {rollout_metrics.get('total_trades', 0)}")
        
        if val_metrics:
            print(f"  Val Reward: {val_metrics['total_reward']:.4f}")
            print(f"  Val Profit: ${val_metrics.get('total_profit', 0):.2f}")
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save model checkpoint"""
        if final:
            path = os.path.join(self.model_dir, 'final_model.pth')
        else:
            path = os.path.join(self.model_dir, f'checkpoint_ep{episode}.pth')
        
        self.agent.save(path)
        print(f"  Saved checkpoint: {path}")
    
    def _save_training_history(self):
        """Save training history"""
        history_path = os.path.join(self.model_dir, 'training_history.json')
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        # Recursively convert all numpy types
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(item) for item in d]
            else:
                return convert_numpy(d)
        
        history_to_save = convert_dict(self.training_history)
        
        with open(history_path, 'w') as f:
            json.dump(history_to_save, f, indent=2)
    
    def _print_final_summary(self):
        """Print final training summary"""
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        
        # Calculate statistics
        rewards = self.training_history['episode_rewards']
        profits = self.training_history['episode_profits']
        
        print(f"Average Episode Reward: {np.mean(rewards):.4f} (+/- {np.std(rewards):.4f})")
        print(f"Average Profit: ${np.mean(profits):.2f} (+/- ${np.std(profits):.2f})")
        print(f"Best Validation Reward: {self.best_val_reward:.4f}")
        print(f"Total Training Time: {sum(self.training_history['training_time']):.1f}s")
        print(f"Model saved to: {self.model_dir}")


class RolloutBuffer:
    """
    Buffer for storing rollout experience
    Optimized for memory efficiency
    """
    
    def __init__(self, buffer_size: int, observation_space, action_space):
        """Initialize buffer"""
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Pre-allocate arrays for efficiency
        obs_shape = observation_space.shape
        action_shape = action_space.shape
        
        self.observations = np.zeros((buffer_size,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((buffer_size,) + action_shape, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.next_observations = np.zeros((buffer_size,) + obs_shape, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.old_log_probs = np.zeros(buffer_size, dtype=np.float32)
        
        self.position = 0
        self.full = False
    
    def add(self, obs, action, reward, next_obs, done, old_log_prob):
        """Add transition to buffer"""
        self.observations[self.position] = obs
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_obs
        self.dones[self.position] = done
        self.old_log_probs[self.position] = old_log_prob
        
        self.position += 1
        if self.position >= self.buffer_size:
            self.full = True
            self.position = 0
    
    def get_batch(self) -> Dict:
        """Get all data as batch"""
        if self.full:
            batch_size = self.buffer_size
        else:
            batch_size = self.position
        
        return {
            'observations': self.observations[:batch_size],
            'actions': self.actions[:batch_size],
            'rewards': self.rewards[:batch_size],
            'next_observations': self.next_observations[:batch_size],
            'dones': self.dones[:batch_size],
            'old_log_probs': self.old_log_probs[:batch_size]
        }
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.full or self.position >= self.buffer_size
    
    def reset(self):
        """Reset buffer"""
        self.position = 0
        self.full = False