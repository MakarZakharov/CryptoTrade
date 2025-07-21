"""SAC агент для торговли криптовалютами."""

import time
from typing import Dict, Any, Optional, Union
import torch
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

from .base_agent import BaseAgent
from ..config import DRLConfig, TradingConfig
from ..utils import DRLLogger
from ..environments import TradingEnv


class SACAgent(BaseAgent):
    """
    SAC (Soft Actor-Critic) агент для торговли.
    
    Реализует алгоритм SAC, оптимизированный для continuous action spaces
    в торговле криптовалютами. Хорошо подходит для задач, требующих
    баланса между исследованием и эксплуатацией.
    """
    
    def __init__(
        self, 
        drl_config: DRLConfig, 
        trading_config: TradingConfig,
        logger: Optional[DRLLogger] = None
    ):
        """
        Инициализация SAC агента.
        
        Args:
            drl_config: конфигурация DRL параметров
            trading_config: конфигурация торговых параметров
            logger: логгер для записи операций
        """
        super().__init__(drl_config, trading_config, logger)
        
        if trading_config.action_type != "continuous":
            raise ValueError("SAC поддерживает только continuous action spaces")
        
        # SAC специфичные параметры
        self.policy_kwargs = self._build_policy_kwargs()
        self.action_noise = self._create_action_noise()
        
        # Предупреждаем о выборе устройства
        self._warn_about_device_selection()
        
        self.logger.info(f"SAC агент инициализирован с устройством: {self.device}")
    

    
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
        
        # SAC использует отдельные сети для actor и critic
        if len(self.drl_config.net_arch) > 0:
            policy_kwargs["net_arch"] = dict(
                pi=self.drl_config.net_arch,
                qf=self.drl_config.net_arch
            )
        
        # State Dependent Exploration
        if self.drl_config.use_sde:
            policy_kwargs["use_sde"] = True
            policy_kwargs["sde_sample_freq"] = 4
            policy_kwargs["use_expln"] = False  # Используем обычное распределение
        
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
    
    def _create_action_noise(self) -> Optional[NormalActionNoise]:
        """Создание шума для действий (опционально)."""
        # SAC обычно не требует внешнего шума, но можем добавить для дополнительного исследования
        if hasattr(self.drl_config, 'action_noise_std') and self.drl_config.action_noise_std > 0:
            action_dim = 1  # Предполагаем одномерные действия для торговли
            noise_std = self.drl_config.action_noise_std
            
            action_noise = NormalActionNoise(
                mean=np.zeros(action_dim),
                sigma=noise_std * np.ones(action_dim)
            )
            
            self.logger.info(f"Создан action noise со стандартным отклонением: {noise_std}")
            return action_noise
        
        return None
    
    def create_model(
        self, 
        env: Union[TradingEnv, VecEnv], 
        **kwargs
    ) -> SAC:
        """
        Создание SAC модели.
        
        Args:
            env: торговая среда
            **kwargs: дополнительные параметры
            
        Returns:
            Созданная SAC модель
        """
        # Подготовка среды
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        
        self.training_env = env
        
        # Получение параметров SAC
        sac_params = self._get_sac_parameters()
        sac_params.update(kwargs)  # Переопределение параметрами из kwargs
        
        # Создание модели
        self.model = SAC(
            policy="MlpPolicy",
            env=env,
            device=self.device,
            policy_kwargs=self.policy_kwargs,
            action_noise=self.action_noise,
            **sac_params
        )
        
        self.logger.info("SAC модель создана")
        self.logger.debug(f"Параметры SAC: {sac_params}")
        
        return self.model
    
    def _get_sac_parameters(self) -> Dict[str, Any]:
        """Получение параметров SAC из конфигурации."""
        # Основные параметры
        params = {
            "learning_rate": self.drl_config.learning_rate,
            "buffer_size": self.drl_config.buffer_size,
            "batch_size": self.drl_config.batch_size,
            "gamma": self.drl_config.gamma,
            "tau": self.drl_config.tau,
            "ent_coef": self.drl_config.alpha,  # В SAC alpha это ent_coef
            "use_sde": self.drl_config.use_sde,
            "verbose": self.drl_config.verbose,
            "seed": self.drl_config.seed,
            "tensorboard_log": self.logs_dir / "tensorboard" if self.drl_config.tensorboard_log else None
        }
        
        # Обработка target_entropy
        if self.drl_config.target_entropy == "auto":
            params["target_entropy"] = "auto"
        else:
            try:
                params["target_entropy"] = float(self.drl_config.target_entropy)
            except (ValueError, TypeError):
                params["target_entropy"] = "auto"
        
        # Дополнительные параметры для улучшения производительности
        params["learning_starts"] = 1000  # Начинаем обучение после накопления опыта
        params["train_freq"] = (1, "step")  # Обучаемся каждый шаг
        params["gradient_steps"] = 1  # Один градиентный шаг за раз
        
        return params
    
    def train(
        self, 
        total_timesteps: int,
        callback=None,
        **kwargs
    ) -> SAC:
        """
        Обучение SAC агента.
        
        Args:
            total_timesteps: общее количество шагов обучения
            callback: callback функции для мониторинга
            **kwargs: дополнительные параметры
            
        Returns:
            Обученная SAC модель
        """
        if self.model is None:
            raise ValueError("Модель не создана. Вызовите create_model() сначала.")
        
        self.logger.info(f"Начинаем обучение SAC на {total_timesteps:,} шагов")
        
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
        
        # Применяем границы действий
        action = np.clip(
            action, 
            self.trading_config.action_bounds[0],
            self.trading_config.action_bounds[1]
        )
        
        # Применяем мертвую зону для уменьшения шума
        dead_zone = 0.03  # Меньше чем у PPO, так как SAC лучше контролирует исследование
        if abs(action[0]) < dead_zone:
            action = np.array([0.0])
        
        return action
    
    def get_policy_statistics(self) -> Dict[str, Any]:
        """Получение статистики политики SAC."""
        if self.model is None:
            return {}
        
        stats = {}
        
        # Статистика энтропийного коэффициента
        if hasattr(self.model, 'ent_coef') and self.model.ent_coef is not None:
            if isinstance(self.model.ent_coef, torch.Tensor):
                stats['entropy_coefficient'] = float(self.model.ent_coef.cpu().item())
            else:
                stats['entropy_coefficient'] = float(self.model.ent_coef)
        
        # Статистика target entropy
        if hasattr(self.model, 'target_entropy') and self.model.target_entropy is not None:
            stats['target_entropy'] = float(self.model.target_entropy)
        
        # Статистика replay buffer
        if hasattr(self.model, 'replay_buffer') and self.model.replay_buffer is not None:
            stats['buffer_size'] = self.model.replay_buffer.size()
            stats['buffer_full'] = self.model.replay_buffer.full
            if hasattr(self.model.replay_buffer, 'pos'):
                stats['buffer_pos'] = self.model.replay_buffer.pos
        
        # Статистика параметров сети
        if hasattr(self.model, 'actor') and self.model.actor is not None:
            total_params = sum(p.numel() for p in self.model.actor.parameters())
            trainable_params = sum(p.numel() for p in self.model.actor.parameters() if p.requires_grad)
            
            stats.update({
                'actor_parameters': total_params,
                'actor_trainable_parameters': trainable_params
            })
        
        if hasattr(self.model, 'critic') and self.model.critic is not None:
            total_params = sum(p.numel() for p in self.model.critic.parameters())
            trainable_params = sum(p.numel() for p in self.model.critic.parameters() if p.requires_grad)
            
            stats.update({
                'critic_parameters': total_params,
                'critic_trainable_parameters': trainable_params
            })
        
        return stats
    
    def adjust_entropy_coefficient(self, new_coef: float):
        """
        Корректировка энтропийного коэффициента.
        
        Args:
            new_coef: новый коэффициент энтропии
        """
        if self.model is None:
            self.logger.warning("Модель не создана")
            return
        
        if hasattr(self.model, 'ent_coef'):
            old_coef = self.model.ent_coef
            self.model.ent_coef = new_coef
            
            self.logger.info(f"Энтропийный коэффициент изменен: {old_coef} -> {new_coef}")
        else:
            self.logger.warning("Невозможно изменить энтропийный коэффициент")
    
    def get_q_values(self, observation: np.ndarray, action: np.ndarray) -> Dict[str, float]:
        """
        Получение Q-значений для анализа.
        
        Args:
            observation: наблюдение
            action: действие
            
        Returns:
            Q-значения от разных критиков
        """
        if self.model is None or not hasattr(self.model, 'critic'):
            return {}
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            
            try:
                # Получаем Q-значения от обоих критиков
                q_values = self.model.critic(obs_tensor, action_tensor)
                
                result = {}
                if isinstance(q_values, (list, tuple)):
                    for i, q_val in enumerate(q_values):
                        result[f'q_value_{i+1}'] = float(q_val.cpu().item())
                else:
                    result['q_value'] = float(q_values.cpu().item())
                
                return result
                
            except Exception as e:
                self.logger.warning(f"Не удалось получить Q-значения: {e}")
                return {}
    
    def get_action_distribution_info(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Получение информации о распределении действий.
        
        Args:
            observation: наблюдение из среды
            
        Returns:
            Информация о распределении
        """
        if self.model is None or not hasattr(self.model, 'actor'):
            return {}
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            try:
                # Получаем действие и log_prob от актора
                if hasattr(self.model.actor, 'action_dist'):
                    action_dist = self.model.actor.action_dist(obs_tensor)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                    
                    return {
                        'action_mean': float(action.cpu().numpy()[0]),
                        'log_prob': float(log_prob.cpu().numpy()[0]),
                        'entropy': float(-log_prob.cpu().numpy()[0])  # Приближение энтропии
                    }
                else:
                    # Альтернативный способ получения действия
                    action = self.model.actor(obs_tensor)
                    return {
                        'action_deterministic': float(action.cpu().numpy()[0])
                    }
                    
            except Exception as e:
                self.logger.warning(f"Не удалось получить информацию о распределении: {e}")
                return {}
    
    def save_components_separately(self, base_path: str):
        """
        Сохранение компонентов SAC отдельно.
        
        Args:
            base_path: базовый путь для сохранения
        """
        if self.model is None:
            raise ValueError("Модель не создана")
        
        # Сохраняем актор
        if hasattr(self.model, 'actor'):
            torch.save(self.model.actor.state_dict(), f"{base_path}_actor.pth")
        
        # Сохраняем критик
        if hasattr(self.model, 'critic'):
            torch.save(self.model.critic.state_dict(), f"{base_path}_critic.pth")
        
        # Сохраняем target критик
        if hasattr(self.model, 'critic_target'):
            torch.save(self.model.critic_target.state_dict(), f"{base_path}_critic_target.pth")
        
        # Сохраняем replay buffer
        if hasattr(self.model, 'replay_buffer'):
            torch.save(self.model.replay_buffer, f"{base_path}_replay_buffer.pkl")
        
        self.logger.info(f"Компоненты SAC сохранены с базовым путем: {base_path}")
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Получение всех гиперпараметров SAC."""
        base_params = super().get_hyperparameters()
        
        sac_specific = {
            'buffer_size': self.drl_config.buffer_size,
            'batch_size': self.drl_config.batch_size,
            'tau': self.drl_config.tau,
            'alpha': self.drl_config.alpha,
            'target_entropy': self.drl_config.target_entropy,
            'use_sde': self.drl_config.use_sde,
            'policy_kwargs': self.policy_kwargs,
            'device': self.device,
            'action_noise': str(self.action_noise) if self.action_noise else None
        }
        
        base_params.update(sac_specific)
        return base_params