import os
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from .base_agent import BaseAgent

class DQNAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model = None
        self.vec_env = None
        self.device = self._get_device()
        
    def _get_device(self):
        """Визначити доступний пристрій (GPU або CPU)."""
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 Використовується GPU: {gpu_name}")
            print(f"💾 Доступна відеопам'ять: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = "cpu"
            print("🔧 GPU недоступний, використовується CPU")
        return device
        
    def create_model(self, env, model_config=None):
        """Створити модель DQN."""
        # Оборачиваємо середовище
        self.vec_env = DummyVecEnv([lambda: env])
        
        # Стабільні параметри для попередження високих втрат
        default_config = {
            'learning_rate': 1e-5,  # Дуже низький learning rate
            'buffer_size': 50000,  # Менший буфер для кращого контролю
            'learning_starts': 5000,  # Раніше початок навчання
            'batch_size': 32,  # Менший batch size для стабільності
            'tau': 0.001,  # Дуже повільне оновлення target network
            'gamma': 0.99,  # Стандартний дисконт фактор
            'train_freq': 8,  # Рідше навчання для стабільності
            'gradient_steps': 1,  # Один градієнтний крок
            'target_update_interval': 2000,  # Рідше оновлення target network
            'exploration_fraction': 0.5,  # Більше exploration
            'exploration_initial_eps': 0.9,  # Менший початковий epsilon
            'exploration_final_eps': 0.01,  # Дуже низький кінцевий epsilon
            'max_grad_norm': 1.0,  # Жорсткий клипінг градієнтів
            'verbose': 1,
            # Стабільна архітектура нейронної мережі
            'policy_kwargs': {
                'net_arch': [64, 64],  # Менша архітектура для стабільності
                'activation_fn': torch.nn.Tanh,  # Tanh для обмеження значень
                'normalize_images': False
            }
        }
        
        if model_config:
            default_config.update(model_config)
        
        # Створюємо модель DQN
        self.model = DQN(
            "MlpPolicy",
            self.vec_env,
            device=self.device,
            **default_config
        )
        
        print(f"✅ Створена DQN модель:")
        print(f"   Learning rate: {default_config['learning_rate']}")
        print(f"   Network architecture: {default_config['policy_kwargs']['net_arch']}")
        print(f"   Buffer size: {default_config['buffer_size']}")
        print(f"   Exploration: {default_config['exploration_initial_eps']} → {default_config['exploration_final_eps']}")
        
        return self.model
    
    def train(self, total_timesteps=100000, callback=None):
        """Навчити агента."""
        if not self.model:
            raise ValueError("Модель не створена. Викличте create_model() спочатку.")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
        
        return self.model
    
    def act(self, state):
        """Обрати дію."""
        if not self.model:
            return np.array([0.0])
        
        action, _ = self.model.predict(state, deterministic=True)
        return action
    
    def save(self, path):
        """Зберегти модель."""
        if self.model:
            self.model.save(path)
    
    def load(self, path, env=None):
        """Завантажити модель."""
        if env:
            self.vec_env = DummyVecEnv([lambda: env])
        
        self.model = DQN.load(path, env=self.vec_env)
        return self.model