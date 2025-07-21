"""A2C агент для торговли криптовалютами."""

import time
from typing import Dict, Any, Optional, Union
import torch
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from .base_agent import BaseAgent
from ..config import DRLConfig, TradingConfig
from ..utils import DRLLogger
from ..environments import TradingEnv


class A2CAgent(BaseAgent):
    """
    A2C (Advantage Actor-Critic) агент для торговли.
    
    Реализует алгоритм A2C - более простую версию PPO, которая
    хорошо подходит для быстрого прототипирования и базового
    обучения торговых стратегий.
    """
    
    def __init__(
        self, 
        drl_config: DRLConfig, 
        trading_config: TradingConfig,
        logger: Optional[DRLLogger] = None
    ):
        """
        Инициализация A2C агента.
        
        Args:
            drl_config: конфигурация DRL параметров
            trading_config: конфигурация торговых параметров
            logger: логгер для записи операций
        """
        super().__init__(drl_config, trading_config, logger)
        
        # A2C специфичные параметры
        self.device = self._determine_device()
        self.policy_kwargs = self._build_policy_kwargs()
        
        self.logger.info(f"A2C агент инициализирован с устройством: {self.device}")
    
    def _determine_device(self) -> str:
        """Определение доступного устройства."""
        if self.drl_config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                gpu_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"Используется GPU: {gpu_name} ({memory_gb:.1f} GB)")
            else:
                device = "cpu"
                self.logger.info("GPU недоступен, используется CPU")
        else:
            device = self.drl_config.device
        
        return device
    
    def _build_policy_kwargs(self) -> Dict[str, Any]:
        """Построение параметров политики."""
        policy_kwargs = {
            "net_arch": self.drl_config.net_arch.copy(),
            "activation_fn": self._get_activation_function(),
            "normalize_images": False,
            "optimizer_class": torch.optim.RMSprop,  # A2C традиционно использует RMSprop
            "optimizer_kwargs": {
                "eps": 1e-5,
                "alpha": 0.99  # RMSprop decay
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
            policy_kwargs["log_std_init"] = -0.5  # Меньше исследования чем у PPO
            policy_kwargs["ortho_init"] = True
        
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
    ) -> A2C:
        """
        Создание A2C модели.
        
        Args:
            env: торговая среда
            **kwargs: дополнительные параметры
            
        Returns:
            Созданная A2C модель
        """
        # Подготовка среды
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        
        self.training_env = env
        
        # Получение параметров A2C
        a2c_params = self._get_a2c_parameters()
        a2c_params.update(kwargs)  # Переопределение параметрами из kwargs
        
        # Создание модели
        self.model = A2C(
            policy="MlpPolicy",
            env=env,
            device=self.device,
            policy_kwargs=self.policy_kwargs,
            **a2c_params
        )
        
        self.logger.info("A2C модель создана")
        self.logger.debug(f"Параметры A2C: {a2c_params}")
        
        return self.model
    
    def _get_a2c_parameters(self) -> Dict[str, Any]:
        """Получение параметров A2C из конфигурации."""
        return {
            "learning_rate": self.drl_config.learning_rate,
            "n_steps": getattr(self.drl_config, 'n_steps', 5),  # A2C обычно использует меньше шагов
            "gamma": self.drl_config.gamma,
            "gae_lambda": getattr(self.drl_config, 'gae_lambda', 1.0),  # A2C часто без GAE
            "ent_coef": getattr(self.drl_config, 'ent_coef', 0.01),
            "vf_coef": getattr(self.drl_config, 'vf_coef', 0.5),
            "max_grad_norm": getattr(self.drl_config, 'max_grad_norm', 0.5),
            "rms_prop_eps": 1e-5,  # Специфично для A2C
            "use_rms_prop": True,  # A2C использует RMSprop
            "verbose": self.drl_config.verbose,
            "seed": self.drl_config.seed,
            "tensorboard_log": self.logs_dir / "tensorboard" if self.drl_config.tensorboard_log else None
        }
    
    def train(
        self, 
        total_timesteps: int,
        callback=None,
        **kwargs
    ) -> A2C:
        """
        Обучение A2C агента.
        
        Args:
            total_timesteps: общее количество шагов обучения
            callback: callback функции для мониторинга
            **kwargs: дополнительные параметры
            
        Returns:
            Обученная A2C модель
        """
        if self.model is None:
            raise ValueError("Модель не создана. Вызовите create_model() сначала.")
        
        self.logger.info(f"Начинаем обучение A2C на {total_timesteps:,} шагов")
        
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
            dead_zone = 0.08  # Больше чем у PPO/SAC, так как A2C может быть более шумным
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
        
        # Статистика value function
        if hasattr(self.model.policy, 'value_net'):
            # Можем получить статистику value network
            total_params = sum(p.numel() for p in self.model.policy.value_net.parameters())
            trainable_params = sum(p.numel() for p in self.model.policy.value_net.parameters() if p.requires_grad)
            
            stats.update({
                'value_net_parameters': total_params,
                'value_net_trainable_parameters': trainable_params
            })
        
        return stats
    
    def get_value_prediction(self, observation: np.ndarray) -> float:
        """
        Получение предсказания value function.
        
        Args:
            observation: наблюдение из среды
            
        Returns:
            Предсказанное значение стоимости состояния
        """
        if self.model is None or not hasattr(self.model.policy, 'predict_values'):
            return 0.0
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            value = self.model.policy.predict_values(obs_tensor)
            return float(value.cpu().item())
    
    def analyze_policy_gradient(self, observations: np.ndarray, actions: np.ndarray) -> Dict[str, Any]:
        """
        Анализ градиентов политики.
        
        Args:
            observations: наблюдения
            actions: действия
            
        Returns:
            Статистика градиентов
        """
        if self.model is None or len(observations) == 0:
            return {}
        
        try:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observations).to(self.device)
                
                # Получаем распределение действий
                if hasattr(self.model.policy, 'get_distribution'):
                    distribution = self.model.policy.get_distribution(obs_tensor)
                    
                    if hasattr(distribution, 'distribution'):
                        dist = distribution.distribution
                        
                        # Энтропия
                        entropy = dist.entropy().mean()
                        
                        # Log probabilities
                        if hasattr(dist, 'log_prob'):
                            actions_tensor = torch.FloatTensor(actions).to(self.device)
                            log_probs = dist.log_prob(actions_tensor)
                            
                            return {
                                'mean_entropy': float(entropy.cpu()),
                                'mean_log_prob': float(log_probs.mean().cpu()),
                                'std_log_prob': float(log_probs.std().cpu())
                            }
            
        except Exception as e:
            self.logger.warning(f"Ошибка анализа градиентов: {e}")
        
        return {}
    
    def get_advantage_estimates(self, observations: np.ndarray, rewards: np.ndarray) -> np.ndarray:
        """
        Получение оценок advantage.
        
        Args:
            observations: наблюдения
            rewards: награды
            
        Returns:
            Оценки advantage
        """
        if self.model is None or len(observations) == 0:
            return np.array([])
        
        try:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observations).to(self.device)
                
                # Получаем values
                values = self.model.policy.predict_values(obs_tensor)
                values_np = values.cpu().numpy().flatten()
                
                # Простое вычисление advantage (без GAE)
                advantages = np.zeros_like(rewards)
                
                for t in range(len(rewards) - 1):
                    delta = rewards[t] + self.drl_config.gamma * values_np[t + 1] - values_np[t]
                    advantages[t] = delta
                
                # Последний шаг
                if len(rewards) > 0:
                    advantages[-1] = rewards[-1] - values_np[-1]
                
                return advantages
                
        except Exception as e:
            self.logger.warning(f"Ошибка вычисления advantage: {e}")
            return np.zeros_like(rewards)
    
    def adjust_learning_rate(self, new_lr: float):
        """
        Корректировка скорости обучения.
        
        Args:
            new_lr: новая скорость обучения
        """
        if self.model is None:
            self.logger.warning("Модель не создана")
            return
        
        old_lr = self.model.learning_rate
        self.model.learning_rate = new_lr
        
        # Обновляем оптимизатор
        if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
            for param_group in self.model.policy.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        self.logger.info(f"Скорость обучения изменена: {old_lr} -> {new_lr}")
    
    def get_training_diagnostics(self) -> Dict[str, Any]:
        """Получение диагностической информации об обучении."""
        if self.model is None:
            return {}
        
        diagnostics = {}
        
        # Информация о модели
        if hasattr(self.model, '_last_obs') and self.model._last_obs is not None:
            diagnostics['has_last_obs'] = True
        
        # Информация об оптимизаторе
        if hasattr(self.model.policy, 'optimizer'):
            optimizer = self.model.policy.optimizer
            diagnostics.update({
                'optimizer_type': optimizer.__class__.__name__,
                'learning_rate': self.model.learning_rate,
                'optimizer_state_dict_keys': list(optimizer.state_dict().keys())
            })
        
        # Параметры модели
        if hasattr(self.model.policy, 'parameters'):
            total_params = sum(p.numel() for p in self.model.policy.parameters())
            trainable_params = sum(p.numel() for p in self.model.policy.parameters() if p.requires_grad)
            
            diagnostics.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'parameter_ratio': trainable_params / total_params if total_params > 0 else 0
            })
        
        return diagnostics
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Получение всех гиперпараметров A2C."""
        base_params = super().get_hyperparameters()
        
        a2c_specific = {
            'n_steps': getattr(self.drl_config, 'n_steps', 5),
            'gae_lambda': getattr(self.drl_config, 'gae_lambda', 1.0),
            'ent_coef': getattr(self.drl_config, 'ent_coef', 0.01),
            'vf_coef': getattr(self.drl_config, 'vf_coef', 0.5),
            'max_grad_norm': getattr(self.drl_config, 'max_grad_norm', 0.5),
            'rms_prop_eps': 1e-5,
            'use_rms_prop': True,
            'policy_kwargs': self.policy_kwargs,
            'device': self.device
        }
        
        base_params.update(a2c_specific)
        return base_params