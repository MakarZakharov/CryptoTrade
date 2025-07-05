import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from .base_agent import BaseAgent

class DQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.vec_env = None
        
    def create_model(self, env, model_config=None):
        """Создать модель DQN."""
        # Оборачиваем среду
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Параметры модели по умолчанию
        default_config = {
            'learning_rate': 1e-4,
            'buffer_size': 100000,
            'learning_starts': 1000,
            'batch_size': 32,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1000,
            'exploration_fraction': 0.1,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
            'verbose': 1
        }
        
        if model_config:
            default_config.update(model_config)
        
        # Создаем модель DQN
        self.model = DQN(
            "MlpPolicy",
            self.vec_env,
            **default_config
        )
        
        return self.model
    
    def train(self, total_timesteps=100000, callback=None):
        """Обучить агента."""
        if not self.model:
            raise ValueError("Модель не создана. Вызовите create_model() сначала.")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
        
        return self.model
    
    def act(self, state):
        """Выбрать действие."""
        if not self.model:
            return np.array([0.0])
        
        action, _ = self.model.predict(state, deterministic=True)
        return action
    
    def save(self, path):
        """Сохранить модель."""
        if self.model:
            self.model.save(path)
    
    def load(self, path, env=None):
        """Загрузить модель."""
        if env:
            self.vec_env = DummyVecEnv([lambda: env])
        
        self.model = DQN.load(path, env=self.vec_env)
        return self.model 