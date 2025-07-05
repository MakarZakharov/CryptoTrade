import os
import numpy as np
import torch
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
        """Створити покращену модель PPO з оптимізованими параметрами."""
        # Оборачиваємо середовище
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Покращені параметри моделі для кращого exploration та навчання
        default_config = {
            # Адаптивний learning rate для стабільного навчання
            'learning_rate': self._linear_schedule(2.5e-4, 5e-5),
            
            # Збільшені кроки для кращого збору досвіду
            'n_steps': 4096,  # Збільшено з 2048
            
            # Оптимізований batch size
            'batch_size': 128,  # Збільшено з 64
            
            # Більше епох для кращого навчання
            'n_epochs': 15,  # Збільшено з 10
            
            # Параметри для довгострокового планування
            'gamma': 0.995,  # Збільшено з 0.99 для торгівлі
            'gae_lambda': 0.98,  # Збільшено з 0.95
            
            # Адаптивний clip range
            'clip_range': self._linear_schedule(0.3, 0.1),  # Починаємо з більшого exploration
            
            # Важливо для exploration в торгівлі
            'ent_coef': 0.02,  # Збільшено з 0.0 для кращого exploration
            
            # Value function коефіцієнт
            'vf_coef': 0.25,  # Зменшено з 0.5 для балансу
            
            # Градієнтний клипінг
            'max_grad_norm': 0.8,  # Збільшено для стабільності
            
            # Параметри нейронної мережі
            'policy_kwargs': {
                'net_arch': {
                    'pi': [256, 256, 128],  # Policy network architecture
                    'vf': [256, 256, 128]   # Value function network architecture
                },
                'activation_fn': torch.nn.ReLU,
                'ortho_init': True,  # Ортогональна ініціалізація
                'log_std_init': -0.5,  # Початкова стандартна девіація для exploration
            },
            
            # State Dependent Exploration (SDE) параметри - передаються безпосередньо до PPO
            'use_sde': True,  # State Dependent Exploration
            'sde_sample_freq': 4,  # Частота оновлення SDE
            
            'verbose': 1,
            'tensorboard_log': f"logs/{self.config.symbol}_{self.config.timeframe}_{self.config.reward_scheme}/tensorboard",
            'device': 'auto'  # Автоматично обираємо CPU/GPU
        }
        
        if model_config:
            default_config.update(model_config)
        
        # Створюємо модель PPO з покращеними параметрами
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            **default_config
        )
        
        print(f"✅ Створена покращена PPO модель:")
        print(f"   Learning rate: {default_config['learning_rate'](1) if callable(default_config['learning_rate']) else default_config['learning_rate']}")
        print(f"   Network architecture: {default_config['policy_kwargs']['net_arch']}")
        print(f"   Exploration coefficient: {default_config['ent_coef']}")
        print(f"   Batch size: {default_config['batch_size']}")
        print(f"   Steps per update: {default_config['n_steps']}")
        
        return self.model
    
    def _linear_schedule(self, initial_value: float, final_value: float):
        """
        Лінійний розклад для параметрів навчання.
        
        Args:
            initial_value: Початкове значення
            final_value: Кінцеве значення
            
        Returns:
            Функція розкладу
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.
            
            Args:
                progress_remaining: Progress remaining (1.0 at start, 0.0 at end)
                
            Returns:
                Current value
            """
            return final_value + progress_remaining * (initial_value - final_value)
        
        return func
    
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