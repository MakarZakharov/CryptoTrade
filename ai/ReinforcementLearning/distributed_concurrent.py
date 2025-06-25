"""
Concurrent Distributed Training System
Allows multiple participants to train the same model simultaneously
"""

import os
import json
import time
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
import threading
import multiprocessing
from collections import deque
import pandas as pd

from .environment.trading_env import TradingEnvironment
from .agents.ppo_agent import PPOAgent
from .training.trainer import RLTrainer, RolloutBuffer
from .config import *


class ConcurrentDistributedTrainer:
    """
    Manages concurrent training by multiple participants
    Uses experience sharing and model averaging
    """
    
    def __init__(self, 
                 shared_dir: str = "concurrent_models",
                 participant_id: str = None,
                 sync_interval: int = 10):
        """
        Initialize concurrent training system
        
        Args:
            shared_dir: Directory for shared models and experiences
            participant_id: Unique identifier for this participant
            sync_interval: Episodes between synchronizations
        """
        self.shared_dir = shared_dir
        self.participant_id = participant_id or f"participant_{os.getenv('USERNAME', 'unknown')}_{int(time.time())}"
        self.sync_interval = sync_interval
        
        # Create directory structure
        self.models_dir = os.path.join(shared_dir, "models")
        self.experiences_dir = os.path.join(shared_dir, "experiences")
        self.state_dir = os.path.join(shared_dir, "state")
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.experiences_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        
        # Global state file
        self.global_state_file = os.path.join(self.state_dir, "global_state.json")
        
        # Experience buffer for sharing
        self.shared_experiences = deque(maxlen=10000)
        
    def initialize_or_resume(self, 
                           data_path: str,
                           timeframe: str = '1h',
                           total_episodes: int = 1000) -> Dict:
        """
        Initialize new training or resume from existing state
        
        Returns:
            Training state dictionary
        """
        if os.path.exists(self.global_state_file):
            with open(self.global_state_file, 'r') as f:
                state = json.load(f)
            print(f"Resuming training from episode {state['total_episodes_completed']}")
        else:
            state = {
                "start_time": datetime.now().isoformat(),
                "total_episodes_target": total_episodes,
                "total_episodes_completed": 0,
                "participants": {},
                "model_versions": [],
                "best_reward": -np.inf,
                "data_path": data_path,
                "timeframe": timeframe
            }
            self._save_global_state(state)
            print("Initialized new concurrent training")
        
        # Register participant
        state['participants'][self.participant_id] = {
            "episodes_contributed": 0,
            "last_sync": datetime.now().isoformat(),
            "status": "active"
        }
        
        return state
    
    def train_concurrent(self,
                        agent: PPOAgent,
                        train_env: TradingEnvironment,
                        val_env: TradingEnvironment,
                        episodes: int = 100,
                        state: Dict = None) -> Dict:
        """
        Train with concurrent synchronization
        
        Args:
            agent: PPO agent to train
            train_env: Training environment
            val_env: Validation environment
            episodes: Number of episodes to train
            state: Current global state
            
        Returns:
            Updated training metrics
        """
        trainer = ConcurrentRLTrainer(
            agent=agent,
            train_env=train_env,
            val_env=val_env,
            coordinator=self,
            state=state,
            config={
                'episodes': episodes,
                'steps_per_episode': STEPS_PER_EPISODE,
                'checkpoint_frequency': self.sync_interval
            }
        )
        
        # Start background sync thread
        sync_thread = threading.Thread(
            target=self._background_sync,
            args=(agent, state),
            daemon=True
        )
        sync_thread.start()
        
        # Train
        trainer.train()
        
        # Final sync
        self._sync_with_peers(agent, state)
        
        return trainer.get_metrics()
    
    def _sync_with_peers(self, agent: PPOAgent, state: Dict):
        """Synchronize model with other participants"""
        # Save current model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"model_{self.participant_id}_{timestamp}.pth"
        model_path = os.path.join(self.models_dir, model_filename)
        agent.save(model_path)
        
        # Get all recent models from other participants
        peer_models = self._get_peer_models(exclude=self.participant_id)
        
        if peer_models:
            # Average models using federated averaging
            self._federated_average(agent, peer_models)
            print(f"Synchronized with {len(peer_models)} peer models")
        
        # Update global state
        state['model_versions'].append({
            "filename": model_filename,
            "participant": self.participant_id,
            "timestamp": timestamp,
            "episode": state['total_episodes_completed']
        })
        
        # Clean old models (keep last 10 per participant)
        self._cleanup_old_models()
        
        self._save_global_state(state)
    
    def _get_peer_models(self, exclude: str = None) -> List[str]:
        """Get recent model files from other participants"""
        models = []
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pth') and (exclude is None or exclude not in filename):
                # Check if model is recent (within last hour)
                filepath = os.path.join(self.models_dir, filename)
                if time.time() - os.path.getmtime(filepath) < 3600:
                    models.append(filepath)
        
        return models[-5:]  # Return last 5 models
    
    def _federated_average(self, agent: PPOAgent, peer_model_paths: List[str]):
        """
        Average model weights with peer models
        Implements Federated Averaging (FedAvg)
        """
        # Get current model state
        current_actor_state = agent.actor.state_dict()
        current_critic_state = agent.critic.state_dict()
        
        # Initialize averaged states
        avg_actor_state = {k: v.clone() for k, v in current_actor_state.items()}
        avg_critic_state = {k: v.clone() for k, v in current_critic_state.items()}
        
        # Weight for averaging (current model + peers)
        total_models = len(peer_model_paths) + 1
        
        # Load and average peer models
        for model_path in peer_model_paths:
            try:
                checkpoint = torch.load(model_path, map_location=agent.device)
                
                # Average actor weights
                peer_actor_state = checkpoint['actor_state_dict']
                for key in avg_actor_state:
                    if key in peer_actor_state:
                        avg_actor_state[key] += peer_actor_state[key]
                
                # Average critic weights
                peer_critic_state = checkpoint['critic_state_dict']
                for key in avg_critic_state:
                    if key in peer_critic_state:
                        avg_critic_state[key] += peer_critic_state[key]
                        
            except Exception as e:
                print(f"Error loading peer model {model_path}: {e}")
                total_models -= 1
        
        # Complete averaging
        if total_models > 1:
            for key in avg_actor_state:
                # Convert to float if needed to avoid casting errors
                if avg_actor_state[key].dtype in [torch.long, torch.int32, torch.int64]:
                    avg_actor_state[key] = avg_actor_state[key].float()
                avg_actor_state[key] /= total_models
            for key in avg_critic_state:
                # Convert to float if needed to avoid casting errors
                if avg_critic_state[key].dtype in [torch.long, torch.int32, torch.int64]:
                    avg_critic_state[key] = avg_critic_state[key].float()
                avg_critic_state[key] /= total_models
            
            # Load averaged weights
            agent.actor.load_state_dict(avg_actor_state)
            agent.critic.load_state_dict(avg_critic_state)
    
    def share_experience(self, experience_batch: Dict):
        """Share experience with other participants"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f"exp_{self.participant_id}_{timestamp}.npz"
        filepath = os.path.join(self.experiences_dir, filename)
        
        # Save experience batch
        np.savez_compressed(
            filepath,
            observations=experience_batch['observations'],
            actions=experience_batch['actions'],
            rewards=experience_batch['rewards'],
            next_observations=experience_batch['next_observations'],
            dones=experience_batch['dones']
        )
        
        # Clean old experiences (keep last hour)
        self._cleanup_old_experiences()
    
    def get_shared_experiences(self, max_samples: int = 1000) -> Optional[Dict]:
        """Get shared experiences from other participants"""
        all_experiences = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'dones': []
        }
        
        # Load recent experience files
        exp_files = []
        for filename in os.listdir(self.experiences_dir):
            if filename.endswith('.npz') and self.participant_id not in filename:
                filepath = os.path.join(self.experiences_dir, filename)
                # Only get experiences from last 30 minutes
                if time.time() - os.path.getmtime(filepath) < 1800:
                    exp_files.append(filepath)
        
        if not exp_files:
            return None
        
        # Load and combine experiences
        for filepath in exp_files[-10:]:  # Last 10 files
            try:
                data = np.load(filepath)
                for key in all_experiences:
                    all_experiences[key].append(data[key])
            except Exception as e:
                print(f"Error loading experience {filepath}: {e}")
        
        # Combine and sample
        if all_experiences['observations']:
            combined = {}
            for key in all_experiences:
                combined[key] = np.concatenate(all_experiences[key], axis=0)
            
            # Random sample if too many
            n_samples = len(combined['observations'])
            if n_samples > max_samples:
                indices = np.random.choice(n_samples, max_samples, replace=False)
                for key in combined:
                    combined[key] = combined[key][indices]
            
            return combined
        
        return None
    
    def _background_sync(self, agent: PPOAgent, state: Dict):
        """Background thread for periodic synchronization"""
        while True:
            time.sleep(300)  # Sync every 5 minutes
            try:
                self._sync_with_peers(agent, state)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Background sync completed")
            except Exception as e:
                print(f"Background sync error: {e}")
    
    def _cleanup_old_models(self):
        """Remove old model files"""
        models_by_participant = {}
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pth'):
                filepath = os.path.join(self.models_dir, filename)
                mtime = os.path.getmtime(filepath)
                
                # Extract participant from filename
                parts = filename.split('_')
                if len(parts) >= 2:
                    participant = parts[1]
                    if participant not in models_by_participant:
                        models_by_participant[participant] = []
                    models_by_participant[participant].append((mtime, filepath))
        
        # Keep only last 10 models per participant
        for participant, models in models_by_participant.items():
            models.sort(reverse=True)  # Sort by modification time
            for _, filepath in models[10:]:
                try:
                    os.remove(filepath)
                except:
                    pass
    
    def _cleanup_old_experiences(self):
        """Remove old experience files"""
        for filename in os.listdir(self.experiences_dir):
            if filename.endswith('.npz'):
                filepath = os.path.join(self.experiences_dir, filename)
                # Remove files older than 1 hour
                if time.time() - os.path.getmtime(filepath) > 3600:
                    try:
                        os.remove(filepath)
                    except:
                        pass
    
    def _save_global_state(self, state: Dict):
        """Save global training state"""
        with open(self.global_state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_training_summary(self) -> Dict:
        """Get summary of distributed training progress"""
        if os.path.exists(self.global_state_file):
            with open(self.global_state_file, 'r') as f:
                state = json.load(f)
            
            active_participants = sum(
                1 for p in state['participants'].values() 
                if p.get('status') == 'active'
            )
            
            return {
                "total_episodes": state['total_episodes_completed'],
                "target_episodes": state['total_episodes_target'],
                "active_participants": active_participants,
                "total_participants": len(state['participants']),
                "best_reward": state['best_reward'],
                "model_versions": len(state['model_versions'])
            }
        
        return {}


class ConcurrentRLTrainer(RLTrainer):
    """
    Modified trainer for concurrent distributed training
    """
    
    def __init__(self,
                 agent,
                 train_env: TradingEnvironment,
                 val_env: Optional[TradingEnvironment] = None,
                 coordinator: ConcurrentDistributedTrainer = None,
                 state: Dict = None,
                 config: Dict = None):
        """Initialize concurrent trainer"""
        super().__init__(agent, train_env, val_env, config)
        self.coordinator = coordinator
        self.global_state = state
        self.local_episodes = 0
        
    def _collect_rollout(self) -> Dict:
        """Collect rollout with experience sharing"""
        metrics = super()._collect_rollout()
        
        # Share experience periodically
        if self.local_episodes % 5 == 0:
            batch = self.rollout_buffer.get_batch()
            self.coordinator.share_experience(batch)
        
        return metrics
    
    def _update_agent(self) -> Dict:
        """Update agent with local and shared experiences"""
        # Get local batch
        local_batch = self.rollout_buffer.get_batch()
        
        # Get shared experiences from peers
        shared_batch = self.coordinator.get_shared_experiences()
        
        if shared_batch:
            # Combine local and shared experiences
            combined_batch = {}
            for key in local_batch:
                if key in shared_batch:
                    combined_batch[key] = np.concatenate([
                        local_batch[key],
                        shared_batch[key][:len(local_batch[key])//2]  # Use 50% shared
                    ], axis=0)
                else:
                    combined_batch[key] = local_batch[key]
            
            # Update with combined batch
            update_metrics = self.agent.update(combined_batch)
        else:
            # Update with local batch only
            update_metrics = self.agent.update(local_batch)
        
        return update_metrics
    
    def train(self, callbacks: Optional[List[Callable]] = None):
        """Training loop with concurrent synchronization"""
        print(f"Starting concurrent training")
        print(f"Participant: {self.coordinator.participant_id}")
        print("-" * 50)
        
        start_episode = self.global_state['total_episodes_completed']
        
        for episode in range(self.episodes):
            self.local_episodes += 1
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
                
                # Update global best
                if val_metrics['total_reward'] > self.global_state['best_reward']:
                    self.global_state['best_reward'] = val_metrics['total_reward']
                    best_path = os.path.join(self.coordinator.models_dir, 'best_model.pth')
                    self.agent.save(best_path)
                    print(f"New best model saved! Reward: {val_metrics['total_reward']:.4f}")
            else:
                val_metrics = {}
            
            # Record metrics
            episode_time = time.time() - start_time
            self._record_metrics(episode, rollout_metrics, update_metrics, val_metrics, episode_time)
            
            # Print progress
            if episode % 10 == 0:
                self._print_progress(episode, rollout_metrics, val_metrics)
            
            # Sync with peers
            if (episode + 1) % self.coordinator.sync_interval == 0:
                print(f"Syncing with other participants...")
                self.coordinator._sync_with_peers(self.agent, self.global_state)
            
            # Update global state
            self.global_state['total_episodes_completed'] = start_episode + episode + 1
            self.global_state['participants'][self.coordinator.participant_id]['episodes_contributed'] += 1
            
            # Save state periodically
            if episode % 50 == 0:
                self.coordinator._save_global_state(self.global_state)
        
        # Final save
        self.coordinator._save_global_state(self.global_state)
        print(f"\nTraining session completed! Contributed {self.local_episodes} episodes")
    
    def get_metrics(self) -> Dict:
        """Get training metrics"""
        return {
            'local_episodes': self.local_episodes,
            'global_episodes': self.global_state['total_episodes_completed'],
            'best_reward': self.global_state['best_reward'],
            'participant_id': self.coordinator.participant_id
        }