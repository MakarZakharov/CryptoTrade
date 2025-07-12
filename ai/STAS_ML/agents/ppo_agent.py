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
        
        # ПОКРАЩЕНІ параметри для максимізації прибутковості
        default_config = {
            'learning_rate': 3e-4,  # Збільшено для швидшого навчання
            'n_steps': 2048,  # Збільшено для кращого sampling
            'batch_size': 64,  # Зменшено для частіших оновлень
            'n_epochs': 10,  # Збільшено для глибшого навчання
            'gamma': 0.995,  # Збільшено для довгострокового планування
            'gae_lambda': 0.98,  # Збільшено для кращої оцінки переваг
            'clip_range': 0.2,  # Збільшено для більшої гнучкості
            'ent_coef': 0.01,  # Збільшено для більшої експлорації
            'vf_coef': 0.5,  # Збільшено для кращої value function
            'max_grad_norm': 0.5,  # Залишено оптимальним
            'verbose': 1,
            # Параметри для ефективності
            'use_sde': False,
            'sde_sample_freq': -1,
            'target_kl': 0.01,  # Зменшено для стабільності
            'normalize_advantage': True,
            # Покращена архітектура для торгівлі
            'policy_kwargs': {
                'net_arch': [256, 128, 64],  # Більша мережа для складніших стратегій
                'activation_fn': torch.nn.Tanh,  # Tanh краще для фінансових даних
                'normalize_images': False,
                'ortho_init': True,  # Кращая ініціалізація
                'log_std_init': -0.5  # Збільшена початкова дисперсія для експлорації
            }
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