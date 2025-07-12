"""
Ультра-агрессивный PPO агент для высокочастотной торговли и скальпинга.
Оптимизирован для достижения 300% годовой доходности на 15-минутных интервалах.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from typing import Dict, Any, Optional, Callable
import time
from .base_agent import BaseAgent


class UltraAggressiveCallback(BaseCallback):
    """Callback для ультра-агрессивного обучения с мониторингом ключевых метрик."""
    
    def __init__(self, 
                 target_annual_return: float = 3.0,  # 300% годовых
                 max_drawdown_limit: float = 0.20,   # 20% максимум
                 min_win_rate: float = 0.60,         # 60% минимум
                 auto_stop_training: bool = True,
                 verbose: int = 1):
        super(UltraAggressiveCallback, self).__init__(verbose)
        self.target_annual_return = target_annual_return
        self.max_drawdown_limit = max_drawdown_limit
        self.min_win_rate = min_win_rate
        self.auto_stop_training = auto_stop_training
        
        # Метрики для мониторинга
        self.best_mean_reward = -np.inf
        self.episodes_completed = 0
        self.performance_history = []
        self.last_eval_timestep = 0
        self.eval_frequency = 10000  # Каждые 10k шагов
        
        # Флаги для автоматической остановки
        self.target_reached = False
        self.risk_exceeded = False
        
    def _on_step(self) -> bool:
        """Проверка на каждом шаге."""
        # Периодическая оценка производительности
        if self.num_timesteps - self.last_eval_timestep >= self.eval_frequency:
            self._evaluate_performance()
            self.last_eval_timestep = self.num_timesteps
            
        # Автоматическая остановка при достижении целей или превышении рисков
        if self.auto_stop_training and (self.target_reached or self.risk_exceeded):
            if self.verbose >= 1:
                reason = "цель достигнута" if self.target_reached else "превышен лимит рисков"
                print(f"\n🛑 Автоматическая остановка обучения: {reason}")
            return False
            
        return True
    
    def _evaluate_performance(self):
        """Оценка текущей производительности агента."""
        try:
            # Получаем информацию из среды
            if hasattr(self.training_env, 'get_attr'):
                env_infos = self.training_env.get_attr('_get_info')
                if env_infos and len(env_infos) > 0:
                    info = env_infos[0]()
                    
                    total_return = info.get('total_return', 0)
                    max_drawdown = info.get('max_drawdown', 0)
                    win_rate = info.get('win_rate', 0)
                    
                    # Проецируем годовую доходность
                    steps_per_day = 96  # 24ч * 4 интервала/час
                    days_simulated = max(1, self.num_timesteps / steps_per_day / len(self.training_env.envs))
                    projected_annual = (1 + total_return) ** (365 / days_simulated) - 1
                    
                    performance = {
                        'timestep': self.num_timesteps,
                        'total_return': total_return,
                        'projected_annual': projected_annual,
                        'max_drawdown': max_drawdown,
                        'win_rate': win_rate,
                        'days_simulated': days_simulated
                    }
                    
                    self.performance_history.append(performance)
                    
                    # Логируем метрики
                    self.logger.record("ultra_aggressive/total_return", total_return)
                    self.logger.record("ultra_aggressive/projected_annual_return", projected_annual)
                    self.logger.record("ultra_aggressive/max_drawdown", max_drawdown)
                    self.logger.record("ultra_aggressive/win_rate", win_rate)
                    
                    if self.verbose >= 1:
                        print(f"\n📊 Оценка производительности [Шаг {self.num_timesteps:,}]:")
                        print(f"   Текущая доходность: {total_return:.2%}")
                        print(f"   Проекция на год: {projected_annual:.1%}")
                        print(f"   Макс. просадка: {max_drawdown:.2%}")
                        print(f"   Win rate: {win_rate:.1%}")
                        print(f"   Дней симуляции: {days_simulated:.1f}")
                    
                    # Проверяем условия для автоматической остановки
                    if projected_annual >= self.target_annual_return and win_rate >= self.min_win_rate:
                        if self.verbose >= 1:
                            print(f"🎯 Цель достигнута! Доходность: {projected_annual:.1%}, Win rate: {win_rate:.1%}")
                        self.target_reached = True
                        
                    if max_drawdown > self.max_drawdown_limit:
                        if self.verbose >= 1:
                            print(f"⚠️ Превышен лимит просадки: {max_drawdown:.2%} > {self.max_drawdown_limit:.2%}")
                        self.risk_exceeded = True
        
        except Exception as e:
            if self.verbose >= 1:
                print(f"⚠️ Ошибка оценки производительности: {e}")


class ScalpingPolicy(ActorCriticPolicy):
    """Специализированная политика для скальпинга с улучшенной архитектурой."""
    
    def __init__(self, *args, **kwargs):
        # Оптимизированная архитектура для скальпинга
        super(ScalpingPolicy, self).__init__(*args, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        """Создать специализированную сеть для скальпинга."""
        self.mlp_extractor = ScalpingMlpExtractor(
            self.features_dim,
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),  # Более глубокая сеть
            activation_fn=nn.LeakyReLU,  # LeakyReLU для лучших градиентов
            device=self.device
        )


class ScalpingMlpExtractor(nn.Module):
    """Извлекатель признаков оптимизированный для скальпинга."""
    
    def __init__(self, features_dim: int, net_arch: Dict, activation_fn, device):
        super(ScalpingMlpExtractor, self).__init__()
        
        # Архитектура policy network (actor)
        policy_layers = []
        last_layer_dim = features_dim
        
        for layer_size in net_arch['pi']:
            policy_layers.append(nn.Linear(last_layer_dim, layer_size))
            policy_layers.append(activation_fn())
            policy_layers.append(nn.Dropout(0.1))  # Небольшой dropout для регуляризации
            last_layer_dim = layer_size
        
        self.policy_net = nn.Sequential(*policy_layers)
        
        # Архитектура value network (critic)
        value_layers = []
        last_layer_dim = features_dim
        
        for layer_size in net_arch['vf']:
            value_layers.append(nn.Linear(last_layer_dim, layer_size))
            value_layers.append(activation_fn())
            value_layers.append(nn.Dropout(0.1))
            last_layer_dim = layer_size
            
        self.value_net = nn.Sequential(*value_layers)
        
        self.latent_dim_pi = net_arch['pi'][-1]
        self.latent_dim_vf = net_arch['vf'][-1]
    
    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_actor(self, features):
        return self.policy_net(features)
    
    def forward_critic(self, features):
        return self.value_net(features)


class UltraAggressivePPOAgent(BaseAgent):
    """
    Ультра-агрессивный PPO агент для высокочастотной торговли.
    
    Особенности:
    - Оптимизирован для 15-минутных интервалов
    - Нацелен на 300% годовую доходность  
    - Строгий контроль рисков (макс 20% просадка)
    - Высокочастотное принятие решений
    - GPU ускорение
    - Автоматическая остановка при достижении целей
    """
    
    def __init__(self, config, use_gpu: bool = True, multi_env: bool = True):
        super().__init__(config)
        self.use_gpu = use_gpu
        self.multi_env = multi_env
        self.model = None
        self.vec_env = None
        self.device = self._setup_device()
        
    def _setup_device(self):
        """Настройка устройства для вычислений."""
        if self.use_gpu and torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 Ультра-агрессивный агент: GPU {gpu_name}")
            print(f"💾 Доступно памяти: {gpu_memory:.1f} GB")
            
            # Оптимизация GPU для быстрых вычислений
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            device = "cpu"
            print("🔧 Ультра-агрессивный агент: CPU режим")
            
        return device
    
    def create_model(self, env, model_config: Optional[Dict] = None):
        """Создать ультра-агрессивную модель PPO."""
        
        # Настройка мульти-процессной среды для ускорения
        if self.multi_env and hasattr(env, 'unwrapped'):
            n_envs = min(4, os.cpu_count())  # Используем до 4 процессов
            print(f"🔄 Создание {n_envs} параллельных сред для ускорения")
            
            def make_env(rank: int, seed: int = 0):
                def _init():
                    env_copy = type(env)(env.config)
                    env_copy.seed(seed + rank)
                    return env_copy
                set_random_seed(seed)
                return _init
            
            self.vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        else:
            self.vec_env = DummyVecEnv([lambda: env])
        
        # Ультра-агрессивные гиперпараметры для 300% годовых
        ultra_aggressive_config = {
            # Обучение
            'learning_rate': 5e-5,  # Более медленное, но стабильное обучение
            'n_steps': 2048,  # Больше шагов для лучшего сбора опыта
            'batch_size': 256,  # Большие батчи для стабильности
            'n_epochs': 6,  # Больше эпох для лучшего обучения
            
            # Дисконтирование и преимущества
            'gamma': 0.999,  # Очень высокое дисконтирование для долгосрочного планирования
            'gae_lambda': 0.99,  # Высокий GAE для точной оценки преимуществ
            
            # Политика
            'clip_range': 0.1,  # Консервативный клиппинг для стабильности
            'clip_range_vf': 0.1,  # Клиппинг функции ценности
            
            # Регуляризация
            'ent_coef': 0.001,  # Минимальная энтропия для эксплуатации
            'vf_coef': 0.5,  # Средний вес функции ценности
            'max_grad_norm': 0.5,  # Строгий клиппинг градиентов
            
            # Оптимизация
            'use_sde': False,  # Отключаем стохастическое исследование
            'sde_sample_freq': -1,
            'target_kl': 0.005,  # Очень строгое ограничение KL
            'normalize_advantage': True,
            
            # Устройство и производительность
            'device': self.device,
            'verbose': 1,
            
            # Специализированная политика
            'policy_kwargs': {
                'net_arch': dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                'activation_fn': torch.nn.LeakyReLU,
                'ortho_init': True,
                'log_std_init': -1.5,  # Более консервативное исследование
                'full_std': False,
                'use_expln': True
            }
        }
        
        # Переопределение пользовательскими параметрами
        if model_config:
            ultra_aggressive_config.update(model_config)
        
        print("🔥 Создание ультра-агрессивной PPO модели...")
        print(f"   📋 Шагов на обновление: {ultra_aggressive_config['n_steps']}")
        print(f"   📦 Размер батча: {ultra_aggressive_config['batch_size']}")
        print(f"   🔄 Эпох обучения: {ultra_aggressive_config['n_epochs']}")
        print(f"   🎯 Целевой KL: {ultra_aggressive_config['target_kl']}")
        print(f"   💻 Устройство: {self.device}")
        
        # Создаем модель
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            **ultra_aggressive_config
        )
        
        return self.model
    
    def train(self, 
              total_timesteps: int = 1000000,
              target_annual_return: float = 3.0,
              max_drawdown_limit: float = 0.20,
              min_win_rate: float = 0.60,
              auto_stop_training: bool = True,
              save_checkpoints: bool = True,
              checkpoint_frequency: int = 100000):
        """
        Обучить ультра-агрессивного агента.
        
        Args:
            total_timesteps: Общее количество шагов обучения
            target_annual_return: Целевая годовая доходность (3.0 = 300%)
            max_drawdown_limit: Максимальная допустимая просадка
            min_win_rate: Минимальный win rate
            auto_stop_training: Автоматическая остановка при достижении целей
            save_checkpoints: Сохранять чекпоинты
            checkpoint_frequency: Частота сохранения чекпоинтов
        """
        if not self.model:
            raise ValueError("Модель не создана. Вызовите create_model() сначала.")
        
        print(f"🔥 Запуск ультра-агрессивного обучения:")
        print(f"   🎯 Цель: {target_annual_return:.0%} годовых")
        print(f"   🛡️ Макс. просадка: {max_drawdown_limit:.1%}")
        print(f"   🏆 Мин. win rate: {min_win_rate:.1%}")
        print(f"   📈 Всего шагов: {total_timesteps:,}")
        print(f"   ⏱️ Расчетное время: {total_timesteps / 10000:.0f} минут")
        
        # Callback для мониторинга
        callback = UltraAggressiveCallback(
            target_annual_return=target_annual_return,
            max_drawdown_limit=max_drawdown_limit,
            min_win_rate=min_win_rate,
            auto_stop_training=auto_stop_training,
            verbose=1
        )
        
        # Callback для сохранения чекпоинтов
        callbacks = [callback]
        
        if save_checkpoints:
            # Создаем директорию для чекпоинтов
            checkpoint_dir = f"CryptoTrade/ai/DRL/models/ultra_aggressive_{int(time.time())}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Callback для периодического сохранения
            checkpoint_callback = CheckpointCallback(
                save_freq=checkpoint_frequency,
                save_path=checkpoint_dir,
                name_prefix="ultra_aggressive"
            )
            callbacks.append(checkpoint_callback)
        
        # Запуск обучения
        start_time = time.time()
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            training_time = time.time() - start_time
            print(f"✅ Обучение завершено за {training_time/60:.1f} минут")
            
            # Финальная оценка
            if hasattr(callback, 'performance_history') and callback.performance_history:
                final_perf = callback.performance_history[-1]
                print(f"\n🏆 Финальные результаты:")
                print(f"   Доходность: {final_perf['total_return']:.2%}")
                print(f"   Годовая проекция: {final_perf['projected_annual']:.1%}")
                print(f"   Макс. просадка: {final_perf['max_drawdown']:.2%}")
                print(f"   Win rate: {final_perf['win_rate']:.1%}")
                
                if final_perf['projected_annual'] >= target_annual_return:
                    print("🎯 Цель по доходности достигнута!")
                if final_perf['win_rate'] >= min_win_rate:
                    print("🏆 Цель по win rate достигнута!")
                if final_perf['max_drawdown'] <= max_drawdown_limit:
                    print("🛡️ Риски под контролем!")
            
        except KeyboardInterrupt:
            print("\n⏹️ Обучение остановлено пользователем")
            training_time = time.time() - start_time
            print(f"📊 Обучено за {training_time/60:.1f} минут")
        
        return self.model
    
    def act(self, state):
        """Действие агента - оптимизировано для скорости."""
        if not self.model:
            return np.array([0.0])
        
        # Быстрое предсказание без лишних проверок
        action, _ = self.model.predict(state, deterministic=True)
        return action
    
    def save(self, path: str, save_replay_buffer: bool = False):
        """Сохранить модель."""
        if self.model:
            self.model.save(path)
            if save_replay_buffer and hasattr(self.model, 'replay_buffer'):
                # Сохранение буфера опыта если нужно
                pass
            print(f"💾 Ультра-агрессивная модель сохранена: {path}")
    
    def load(self, path: str, env=None):
        """Загрузить модель."""
        if env:
            if self.multi_env:
                self.vec_env = SubprocVecEnv([lambda: env] * min(4, os.cpu_count()))
            else:
                self.vec_env = DummyVecEnv([lambda: env])
        
        self.model = PPO.load(path, env=self.vec_env)
        print(f"📁 Ультра-агрессивная модель загружена: {path}")
        return self.model


class CheckpointCallback(BaseCallback):
    """Callback для сохранения чекпоинтов."""
    
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "model"):
        super(CheckpointCallback, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose >= 1:
                print(f"💾 Чекпоинт сохранен: {path}")
        return True


def create_ultra_aggressive_agent(config, use_gpu: bool = True, multi_env: bool = True) -> UltraAggressivePPOAgent:
    """Быстрое создание ультра-агрессивного агента."""
    return UltraAggressivePPOAgent(config, use_gpu=use_gpu, multi_env=multi_env)