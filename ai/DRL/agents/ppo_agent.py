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
        self.device = self._get_device()
        
    def _get_device(self):
        """Определить доступное устройство (GPU или CPU)."""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 Используется GPU: {gpu_name}")
            print(f"💾 Доступная видеопамять: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = "cpu"
            print("🔧 GPU недоступен, используется CPU")
        return device
        
    def create_model(self, env, model_config=None):
        """Создать модель PPO."""
        # Оборачиваем среду
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Оптимизированные параметры для прибыльной торговли на 15мин
        default_config = {
            'learning_rate': 1e-4,  # Более стабильное обучение
            'n_steps': 1024,  # Меньше шагов для частых обновлений
            'batch_size': 128,  # Больше размер батча для стабильности
            'n_epochs': 4,  # Меньше эпох для предотвращения переобучения
            'gamma': 0.995,  # Выше для важности будущих наград
            'gae_lambda': 0.98,  # Выше для лучшей оценки преимуществ
            'clip_range': 0.15,  # Более консервативная политика
            'ent_coef': 0.01,  # Небольшое исследование для стабильности
            'vf_coef': 0.25,  # Меньший вес функции ценности
            'max_grad_norm': 0.3,  # Более строгий клиппинг градиентов
            'verbose': 1,
            # Дополнительные параметры для стабильности
            'use_sde': False,  # Отключаем стохастическое исследование
            'sde_sample_freq': -1,
            'target_kl': 0.01,  # Ограничиваем изменения политики
            'normalize_advantage': True  # Нормализация преимуществ
        }
        
        if model_config:
            default_config.update(model_config)
        
        # Создаем модель PPO
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            device=self.device,
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