"""
Базовый класс для всех DRL агентов.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import logging
import os
from datetime import datetime


class BaseAgent(ABC):
    """Абстрактный базовый класс для всех DRL агентов."""
    
    def __init__(self, env, config: Dict[str, Any] = None):
        self.env = env
        self.config = config or {}
        self.model = None
        self.training_stats = {
            'total_timesteps': 0,
            'episodes': 0,
            'best_reward': float('-inf'),
            'training_time': 0
        }
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логирования."""
        logger = logging.getLogger(f'{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    @abstractmethod
    def create_model(self, **kwargs):
        """Создание модели агента."""
        pass
    
    @abstractmethod
    def train(self, total_timesteps: int, **kwargs):
        """Обучение агента."""
        pass
    
    @abstractmethod
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Предсказание действия."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Сохранение модели."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Загрузка модели."""
        pass
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Оценка производительности агента."""
        if self.model is None:
            raise ValueError("Модель не создана. Сначала создайте или загрузите модель.")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            terminated = truncated = False
            
            while not (terminated or truncated):
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Получение статистики обучения."""
        return self.training_stats.copy()
    
    def update_training_stats(self, **kwargs):
        """Обновление статистики обучения."""
        self.training_stats.update(kwargs)


class AgentFactory:
    """Фабрика для создания агентов."""
    
    @staticmethod
    def create_agent(agent_type: str, env, config: Dict[str, Any] = None):
        """
        Создание агента по типу.
        
        Args:
            agent_type: Тип агента ('PPO', 'A2C', 'DDPG', 'DQN')
            env: Среда обучения
            config: Конфигурация агента
            
        Returns:
            Экземпляр агента
        """
        agent_type = agent_type.upper()
        
        if agent_type == 'PPO':
            from .ppo_agent import PPOAgent
            return PPOAgent(env, config)
        elif agent_type == 'A2C':
            from .a2c_agent import A2CAgent
            return A2CAgent(env, config)
        elif agent_type == 'DDPG':
            from .ddpg_agent import DDPGAgent
            return DDPGAgent(env, config)
        elif agent_type == 'DQN':
            from .dqn_agent import DQNAgent
            return DQNAgent(env, config)
        else:
            raise ValueError(f"Неизвестный тип агента: {agent_type}")


def get_default_config(agent_type: str) -> Dict[str, Any]:
    """Получение конфигурации по умолчанию для агента."""
    configs = {
        'PPO': {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1
        },
        'A2C': {
            'learning_rate': 7e-4,
            'n_steps': 5,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1
        },
        'DDPG': {
            'learning_rate': 1e-3,
            'buffer_size': 1000000,
            'learning_starts': 100,
            'batch_size': 100,
            'tau': 0.005,
            'gamma': 0.98,
            'train_freq': 1,
            'gradient_steps': 1,
            'verbose': 1
        },
        'DQN': {
            'learning_rate': 1e-4,
            'buffer_size': 1000000,
            'learning_starts': 50000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 10000,
            'exploration_fraction': 0.1,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
            'verbose': 1
        }
    }
    
    return configs.get(agent_type.upper(), {})