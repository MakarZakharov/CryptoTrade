import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from .base_agent import BaseAgent

class PPOAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.vec_env = None
        
    def create_model(self, env, model_config=None):
        """Создать модель PPO."""
        # Оборачиваем среду
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Параметры модели по умолчанию
        default_config = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'verbose': 1
        }
        
        if model_config:
            default_config.update(model_config)
        
        # Создаем модель PPO
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            device="cpu",  # Используем CPU для лучшей производительности с MlpPolicy
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
        
        self.model = PPO.load(path, env=self.vec_env)
        return self.model 