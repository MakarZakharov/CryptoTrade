import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from .base_agent import BaseAgent


class ExplorationMaintenanceCallback(BaseCallback):
    """Callback для підтримки мінімального рівня експлорації під час навчання."""
    
    def __init__(self, min_std=1.0, check_frequency=500):
        super().__init__()
        self.min_std = min_std
        self.check_frequency = check_frequency
        self.step_count = 0
    
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Перевіряємо та корегуємо експлорацію кожні check_frequency кроків
        if self.step_count % self.check_frequency == 0:
            if hasattr(self.model.policy, 'log_std'):
                current_std = torch.exp(self.model.policy.log_std).mean().item()
                
                if current_std < self.min_std:
                    # АГРЕСИВНО підвищуємо std якщо він занадто низький
                    target_log_std = np.log(self.min_std)
                    with torch.no_grad():
                        self.model.policy.log_std.fill_(target_log_std)
                    print(f"🔧 FORCED EXPLORATION: std {current_std:.3f} -> {self.min_std:.3f}")
                else:
                    print(f"✅ Exploration OK: std={current_std:.3f}")
        
        return True

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
        
    def _create_parallel_envs(self, env, n_envs=4):
        """Створити паралельні середовища для прискорення."""
        import platform
        
        # ВИПРАВЛЕННЯ: На Windows використовуємо DummyVecEnv через проблеми з multiprocessing
        if platform.system() == "Windows":
            print(f"🔧 Windows виявлено - використовуємо DummyVecEnv для стабільності")
            return DummyVecEnv([lambda: env])
        
        try:
            # Спробуємо створити паралельні середовища тільки на Unix системах
            env_fns = []
            for i in range(n_envs):
                env_fns.append(lambda: env)
            
            # Використовуємо SubprocVecEnv для паралелізації
            vec_env = SubprocVecEnv(env_fns)
            print(f"✅ Створено {n_envs} паралельних середовищ для прискорення")
            return vec_env
        except Exception as e:
            print(f"⚠️ Не вдалося створити паралельні середовища: {e}")
            print("🔄 Використовуємо послідовне середовище")
            return DummyVecEnv([lambda: env])

    def create_model(self, env, model_config=None):
        """Создать модель PPO."""
        # Створюємо векторизоване середовище (паралельне якщо можливо)
        self.vec_env = self._create_parallel_envs(env, n_envs=4)
        
        # 🚨 КРИТИЧНО: ЕКСТРЕНІ ПАРАМЕТРИ ДЛЯ ФОРСОВАНОЇ ЕКСПЛОРАЦІЇ
        default_config = {
            'learning_rate': 3e-4,  
            'n_steps': 2048,  
            'batch_size': 64,  
            'n_epochs': 10,  
            'gamma': 0.99,  
            'gae_lambda': 0.95,  
            'clip_range': 0.2,  
            # 🚨 МАКСИМАЛЬНА ЕНТРОПІЯ для форсованої експлорації
            'ent_coef': 1.0,  # МАКСИМАЛЬНО ЗБІЛЬШЕНО для екстремальної експлорації
            'vf_coef': 0.5,  
            'max_grad_norm': 0.5,  
            'verbose': 1,
            # 🚨 КРИТИЧНО: ЗБІЛЬШУЄМО target_kl - занадто низьке значення блокує навчання
            'target_kl': 0.05,  # ЗБІЛЬШЕНО з 0.01 до 0.05 - бачимо "Early stopping due to max kl"
            'normalize_advantage': True,
            # 🚨 РАДИКАЛЬНІ ЗМІНИ policy_kwargs
            'policy_kwargs': {
                'net_arch': [32, 32],  # КАРДИНАЛЬНО ЗМЕНШЕНО - проста мережа для кращої експлорації
                'activation_fn': torch.nn.ReLU,  # ПОВЕРТАЄМО ReLU
                'normalize_images': False,
                'ortho_init': True,  # ВМИКАЄМО НАЗАД - може допомогти з ініціалізацією
                # 🚨 МАКСИМАЛЬНА log_std_init для екстремальної експлорації
                'log_std_init': 1.0,  # МАКСИМАЛЬНО ЗБІЛЬШЕНО: 1.0 = std=2.7 (екстремальна експлорація)
                'optimizer_class': torch.optim.Adam,
                'optimizer_kwargs': {'eps': 1e-5}
            }
        }
        
        if model_config:
            # КРИТИЧНО: Правильно об'єднуємо policy_kwargs замість повної заміни
            if 'policy_kwargs' in model_config and 'policy_kwargs' in default_config:
                # Об'єднуємо policy_kwargs, надаючи пріоритет model_config
                merged_policy_kwargs = default_config['policy_kwargs'].copy()
                merged_policy_kwargs.update(model_config['policy_kwargs'])
                
                # Тимчасово зберігаємо об'єднані policy_kwargs
                temp_policy_kwargs = merged_policy_kwargs
                
                # Оновлюємо config без policy_kwargs
                model_config_without_policy = {k: v for k, v in model_config.items() if k != 'policy_kwargs'}
                default_config.update(model_config_without_policy)
                
                # Встановлюємо об'єднані policy_kwargs
                default_config['policy_kwargs'] = temp_policy_kwargs
                
                print(f"🔧 Об'єднано policy_kwargs:")
                print(f"   log_std_init: {default_config['policy_kwargs'].get('log_std_init', 'NOT SET')}")
                print(f"   net_arch: {default_config['policy_kwargs'].get('net_arch', 'NOT SET')}")
                print(f"   ortho_init: {default_config['policy_kwargs'].get('ortho_init', 'NOT SET')}")
            else:
                default_config.update(model_config)
        
        print(f"🔧 Створюємо PPO модель з КРИТИЧНИМИ параметрами експлорації:")
        print(f"   ent_coef: {default_config['ent_coef']}")
        print(f"   log_std_init: {default_config['policy_kwargs']['log_std_init']}")
        print(f"   clip_range: {default_config['clip_range']}")
        print(f"   target_kl: {default_config['target_kl']}")
        
        # Создаем модель PPO
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            device=self.device,
            **default_config
        )
        
        # КРИТИЧНО: Після створення моделі форсуємо ініціалізацію log_std
        print(f"🔧 Форсована ініціалізація log_std після створення моделі...")
        self._force_exploration_init()
        
        # НОВИЙ: Встановлюємо callback для підтримки експлорації під час навчання
        self._setup_exploration_maintenance()
        
        return self.model
    
    def _force_exploration_init(self):
        """Форсована ініціалізація параметрів експлорації."""
        if hasattr(self.model.policy, 'log_std'):
            # Безпосередньо встановлюємо log_std для МАКСИМАЛЬНОЇ експлорації
            with torch.no_grad():
                self.model.policy.log_std.fill_(0.5)  # std = exp(0.5) ≈ 1.65
                print(f"✅ log_std форсовано встановлено на 0.5 (std ≈ 1.65)")
        elif hasattr(self.model.policy, 'action_net') and hasattr(self.model.policy.action_net, 'log_std'):
            with torch.no_grad():
                self.model.policy.action_net.log_std.fill_(0.5)
                print(f"✅ action_net.log_std форсовано встановлено на 0.5")
        else:
            print(f"⚠️ Не вдалося знайти log_std параметр в policy")
    
    def _setup_exploration_maintenance(self):
        """Встановлення механізму підтримки експлорації під час навчання."""
        # Створюємо callback для підтримки мінімального рівня експлорації
        self.exploration_callback = ExplorationMaintenanceCallback()
    
    def train(self, total_timesteps=100000, callback=None):
        """Обучить агента з форсованою підтримкою експлорації."""
        if not self.model:
            raise ValueError("Модель не создана. Вызовите create_model() сначала.")
        
        # Комбінуємо callbacks: основний + exploration maintenance
        from stable_baselines3.common.callbacks import CallbackList
        callbacks = []
        if callback:
            callbacks.append(callback)
        if hasattr(self, 'exploration_callback'):
            callbacks.append(self.exploration_callback)
        
        # Правильно комбінуємо callbacks за допомогою CallbackList
        if len(callbacks) > 1:
            final_callback = CallbackList(callbacks)
        elif len(callbacks) == 1:
            final_callback = callbacks[0]
        else:
            final_callback = None
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=final_callback
        )
        
        return self.model
    
    def act(self, state):
        """Выбрать действие с форсованою експлорацією."""
        if not self.model:
            return np.array([0.0])
        
        # КРИТИЧНО: Використовуємо НЕ детермінований режим для експлорації
        action, _ = self.model.predict(state, deterministic=False)
        
        # ДОДАТКОВА форсована експлорація через шум
        exploration_noise = np.random.normal(0, 0.1, size=action.shape)
        action = action + exploration_noise
        action = np.clip(action, -1.0, 1.0)
        
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