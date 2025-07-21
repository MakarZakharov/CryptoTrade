"""PPO агент для торговли криптовалютами."""

import time
from typing import Dict, Any, Optional, Union
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy

from .base_agent import BaseAgent
from ..config import DRLConfig, TradingConfig
from ..utils import DRLLogger
from ..environments import TradingEnv


class PPOAgent(BaseAgent):
    """
    PPO (Proximal Policy Optimization) агент для торговли.
    
    Реализует алгоритм PPO с настройками, оптимизированными для
    торговли криптовалютами. Поддерживает continuous и discrete
    action spaces, а также различные архитектуры нейронных сетей.
    """
    
    def __init__(
        self, 
        drl_config: DRLConfig, 
        trading_config: TradingConfig,
        logger: Optional[DRLLogger] = None
    ):
        """
        Инициализация PPO агента.
        
        Args:
            drl_config: конфигурация DRL параметров
            trading_config: конфигурация торговых параметров
            logger: логгер для записи операций
        """
        super().__init__(drl_config, trading_config, logger)
        
        # PPO специфичные параметры
        self.policy_kwargs = self._build_policy_kwargs()
        
        # Предупреждаем о выборе устройства
        self._warn_about_device_selection()
        
        self.logger.info(f"PPO агент инициализирован с устройством: {self.device}")
    

    
    def _build_policy_kwargs(self) -> Dict[str, Any]:
        """Построение параметров политики."""
        policy_kwargs = {
            "net_arch": self.drl_config.net_arch.copy(),
            "activation_fn": self._get_activation_function(),
            "normalize_images": False,
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": {
                "eps": 1e-5
            }
        }
        
        # LSTM поддержка
        if self.drl_config.use_lstm:
            policy_kwargs["net_arch"].append(dict(
                pi=[self.drl_config.lstm_hidden_size],
                vf=[self.drl_config.lstm_hidden_size]
            ))
            policy_kwargs["enable_critic_lstm"] = True
            policy_kwargs["lstm_hidden_size"] = self.drl_config.lstm_hidden_size
        
        # Настройки для continuous action space
        if self.trading_config.action_type == "continuous":
            # Инициализация log_std для контроля исследования
            policy_kwargs["log_std_init"] = 0.0  # std = exp(0) = 1.0
            policy_kwargs["ortho_init"] = True
            
            if self.drl_config.use_sde:
                policy_kwargs["sde_sample_freq"] = 4
        
        self.logger.debug(f"Policy kwargs: {policy_kwargs}")
        return policy_kwargs
    
    def _get_activation_function(self):
        """Получение функции активации."""
        activation_map = {
            "relu": torch.nn.ReLU,
            "tanh": torch.nn.Tanh,
            "elu": torch.nn.ELU,
            "leaky_relu": torch.nn.LeakyReLU,
            "swish": torch.nn.SiLU
        }
        
        return activation_map.get(self.drl_config.activation_fn, torch.nn.ReLU)
    
    def create_model(
        self, 
        env: Union[TradingEnv, VecEnv], 
        **kwargs
    ) -> PPO:
        """
        Создание PPO модели.
        
        Args:
            env: торговая среда
            **kwargs: дополнительные параметры
            
        Returns:
            Созданная PPO модель
        """
        # Подготовка среды
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        
        self.training_env = env
        
        # Получение параметров PPO
        ppo_params = self._get_ppo_parameters()
        ppo_params.update(kwargs)  # Переопределение параметрами из kwargs
        
        # Создание модели
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            device=self.device,
            policy_kwargs=self.policy_kwargs,
            **ppo_params
        )
        
        self.logger.info("PPO модель создана")
        self.logger.debug(f"Параметры PPO: {ppo_params}")
        
        return self.model
    
    def _get_ppo_parameters(self) -> Dict[str, Any]:
        """Получение параметров PPO из конфигурации."""
        return {
            "learning_rate": self.drl_config.learning_rate,
            "n_steps": self.drl_config.n_steps,
            "batch_size": self.drl_config.batch_size,
            "n_epochs": self.drl_config.n_epochs,
            "gamma": self.drl_config.gamma,
            "gae_lambda": self.drl_config.gae_lambda,
            "clip_range": self.drl_config.clip_range,
            "ent_coef": self.drl_config.ent_coef,
            "vf_coef": self.drl_config.vf_coef,
            "max_grad_norm": self.drl_config.max_grad_norm,
            "use_sde": self.drl_config.use_sde,
            "verbose": self.drl_config.verbose,
            "seed": self.drl_config.seed,
            "tensorboard_log": None
        }
    
    def train(
        self, 
        total_timesteps: int,
        callback=None,
        **kwargs
    ) -> PPO:
        """
        Обучение PPO агента.
        
        Args:
            total_timesteps: общее количество шагов обучения
            callback: callback функции для мониторинга
            **kwargs: дополнительные параметры
            
        Returns:
            Обученная PPO модель
        """
        if self.model is None:
            raise ValueError("Модель не создана. Вызовите create_model() сначала.")
        
        self.logger.info(f"Начинаем обучение PPO на {total_timesteps:,} шагов")
        
        # Засекаем время
        start_time = time.time()
        
        try:
            # Основное обучение
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                **kwargs
            )
            
            training_time = time.time() - start_time
            
            # Обновляем статистику
            self.update_training_stats(
                timesteps=total_timesteps,
                training_time=training_time
            )
            
            self.logger.info(f"Обучение завершено за {training_time:.2f} секунд")
            
        except Exception as e:
            self.logger.error(f"Ошибка во время обучения: {e}")
            raise
        
        return self.model
    
    def predict(
        self, 
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Union[int, np.ndarray]:
        """
        Предсказание действия с дополнительной обработкой для торговли.
        
        Args:
            observation: наблюдение из среды
            deterministic: использовать детерминистичную политику
            
        Returns:
            Предсказанное действие
        """
        if self.model is None:
            raise ValueError("Модель не создана.")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        
        # Дополнительная обработка для continuous действий
        if self.trading_config.action_type == "continuous":
            # Применяем границы действий
            action = np.clip(
                action, 
                self.trading_config.action_bounds[0],
                self.trading_config.action_bounds[1]
            )
            
            # Применяем мертвую зону для уменьшения шума
            dead_zone = 0.05
            if abs(action[0]) < dead_zone:
                action = np.array([0.0])
        
        return action
    
    def get_policy_statistics(self) -> Dict[str, Any]:
        """Получение статистики политики."""
        if self.model is None or not hasattr(self.model, 'policy'):
            return {}
        
        stats = {}
        
        # Статистика для continuous политик
        if hasattr(self.model.policy, 'log_std') and self.model.policy.log_std is not None:
            with torch.no_grad():
                log_std = self.model.policy.log_std.cpu().numpy()
                std = np.exp(log_std)
                
                stats.update({
                    'log_std_mean': float(np.mean(log_std)),
                    'log_std_std': float(np.std(log_std)),
                    'std_mean': float(np.mean(std)),
                    'std_min': float(np.min(std)),
                    'std_max': float(np.max(std))
                })
        
        # Статистика параметров сети
        if hasattr(self.model.policy, 'mlp_extractor'):
            total_params = sum(p.numel() for p in self.model.policy.parameters())
            trainable_params = sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)
            
            stats.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            })
        
        return stats
    
    def adjust_exploration(self, factor: float = 1.0):
        """
        Корректировка уровня исследования (exploration).
        
        Args:
            factor: фактор корректировки (>1 увеличивает исследование)
        """
        if self.model is None or not hasattr(self.model.policy, 'log_std'):
            self.logger.warning("Невозможно скорректировать исследование")
            return
        
        with torch.no_grad():
            current_log_std = self.model.policy.log_std.clone()
            new_log_std = current_log_std + np.log(factor)
            
            # Ограничиваем разумными пределами
            new_log_std = torch.clamp(new_log_std, -2.0, 1.0)  # std от ~0.14 до ~2.7
            
            self.model.policy.log_std.copy_(new_log_std)
            
            self.logger.info(f"Исследование скорректировано на фактор {factor:.3f}")
    
    def get_action_distribution(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Получение распределения действий для анализа.
        
        Args:
            observation: наблюдение из среды
            
        Returns:
            Статистика распределения действий
        """
        if self.model is None:
            return {}
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            # Получаем распределение от политики
            if hasattr(self.model.policy, 'get_distribution'):
                distribution = self.model.policy.get_distribution(obs_tensor)
                
                if hasattr(distribution, 'distribution'):
                    dist = distribution.distribution
                    
                    if hasattr(dist, 'loc') and hasattr(dist, 'scale'):
                        # Normal distribution (continuous)
                        return {
                            'mean': float(dist.loc.cpu().numpy()[0]),
                            'std': float(dist.scale.cpu().numpy()[0]),
                            'type': 'normal'
                        }
                    elif hasattr(dist, 'probs'):
                        # Categorical distribution (discrete)
                        probs = dist.probs.cpu().numpy()[0]
                        return {
                            'probabilities': probs.tolist(),
                            'entropy': float(-np.sum(probs * np.log(probs + 1e-8))),
                            'type': 'categorical'
                        }
        
        return {}
    
    def save_policy_only(self, path: str):
        """
        Сохранение только политики (для deployment).
        
        Args:
            path: путь для сохранения политики
        """
        if self.model is None or not hasattr(self.model, 'policy'):
            raise ValueError("Модель или политика не найдена")
        
        torch.save({
            'policy_state_dict': self.model.policy.state_dict(),
            'policy_kwargs': self.policy_kwargs,
            'action_space': self.training_env.action_space if self.training_env else None,
            'observation_space': self.training_env.observation_space if self.training_env else None
        }, path)
        
        self.logger.info(f"Политика сохранена: {path}")
    
    def load_policy_only(self, path: str, env: Optional[Union[TradingEnv, VecEnv]] = None):
        """
        Загрузка только политики.
        
        Args:
            path: путь к сохраненной политике
            env: среда (если нужно)
        """
        if not torch.cuda.is_available() and self.device == "cuda":
            map_location = torch.device('cpu')
        else:
            map_location = None
        
        checkpoint = torch.load(path, map_location=map_location)
        
        # Создаем минимальную модель если её нет
        if self.model is None and env is not None:
            self.create_model(env)
        
        if self.model is not None:
            self.model.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.logger.info(f"Политика загружена: {path}")
        
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Получение всех гиперпараметров PPO."""
        base_params = super().get_hyperparameters()
        
        ppo_specific = {
            'n_steps': self.drl_config.n_steps,
            'batch_size': self.drl_config.batch_size,
            'n_epochs': self.drl_config.n_epochs,
            'clip_range': self.drl_config.clip_range,
            'ent_coef': self.drl_config.ent_coef,
            'vf_coef': self.drl_config.vf_coef,
            'max_grad_norm': self.drl_config.max_grad_norm,
            'gae_lambda': self.drl_config.gae_lambda,
            'use_sde': self.drl_config.use_sde,
            'policy_kwargs': self.policy_kwargs,
            'device': self.device
        }
        
        base_params.update(ppo_specific)
        return base_params