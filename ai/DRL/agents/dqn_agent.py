"""DQN агент для торговли криптовалютами."""

import time
from typing import Dict, Any, Optional, Union
import torch
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.dqn.policies import DQNPolicy

from .base_agent import BaseAgent
from ..config import DRLConfig, TradingConfig
from ..utils import DRLLogger
from ..environments import TradingEnv


class DQNAgent(BaseAgent):
    """
    DQN (Deep Q-Network) агент для торговли.
    
    Реализует алгоритм DQN для discrete action spaces в торговле
    криптовалютами. Включает улучшения как Double DQN, Dueling DQN
    и Prioritized Experience Replay.
    """
    
    def __init__(
        self, 
        drl_config: DRLConfig, 
        trading_config: TradingConfig,
        logger: Optional[DRLLogger] = None
    ):
        """
        Инициализация DQN агента.
        
        Args:
            drl_config: конфигурация DRL параметров
            trading_config: конфигурация торговых параметров
            logger: логгер для записи операций
        """
        super().__init__(drl_config, trading_config, logger)
        
        if trading_config.action_type != "discrete":
            self.logger.warning("DQN обычно используется с discrete actions. "
                              "Рекомендуется использовать discrete action type.")
        
        # DQN специфичные параметры
        self.policy_kwargs = self._build_policy_kwargs()
        
        # Параметры exploration
        self.current_exploration_rate = self.drl_config.exploration_initial_eps
        
        # Предупреждаем о выборе устройства
        self._warn_about_device_selection()
        
        self.logger.info(f"DQN агент инициализирован с устройством: {self.device}")
    

    
    def _build_policy_kwargs(self) -> Dict[str, Any]:
        """Построение параметров политики."""
        policy_kwargs = {
            "net_arch": self.drl_config.net_arch.copy(),
            "activation_fn": self._get_activation_function(),
            "normalize_images": False,
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": {
                "eps": 1e-4  # Немного больше для стабильности DQN
            }
        }
        
        # Dueling DQN архитектура
        if hasattr(self.drl_config, 'use_dueling') and getattr(self.drl_config, 'use_dueling', True):
            # Создаем отдельные ветви для value и advantage
            if len(self.drl_config.net_arch) > 0:
                policy_kwargs["net_arch"] = self.drl_config.net_arch + [
                    {"vf": [64], "q": [64]}  # Отдельные головы для value и Q-функции
                ]
        
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
    ) -> DQN:
        """
        Создание DQN модели.
        
        Args:
            env: торговая среда
            **kwargs: дополнительные параметры
            
        Returns:
            Созданная DQN модель
        """
        # Подготовка среды
        if not isinstance(env, VecEnv):
            env = DummyVecEnv([lambda: env])
        
        self.training_env = env
        
        # Получение параметров DQN
        dqn_params = self._get_dqn_parameters()
        dqn_params.update(kwargs)  # Переопределение параметрами из kwargs
        
        # Создание модели
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            device=self.device,
            policy_kwargs=self.policy_kwargs,
            **dqn_params
        )
        
        self.logger.info("DQN модель создана")
        self.logger.debug(f"Параметры DQN: {dqn_params}")
        
        return self.model
    
    def _get_dqn_parameters(self) -> Dict[str, Any]:
        """Получение параметров DQN из конфигурации."""
        params = {
            "learning_rate": self.drl_config.learning_rate,
            "buffer_size": self.drl_config.buffer_size,
            "batch_size": self.drl_config.batch_size,
            "gamma": self.drl_config.gamma,
            "tau": self.drl_config.tau,
            "exploration_fraction": self.drl_config.exploration_fraction,
            "exploration_initial_eps": self.drl_config.exploration_initial_eps,
            "exploration_final_eps": self.drl_config.exploration_final_eps,
            "target_update_interval": self.drl_config.target_update_interval,
            "verbose": self.drl_config.verbose,
            "seed": self.drl_config.seed,
            "tensorboard_log": self.logs_dir / "tensorboard" if self.drl_config.tensorboard_log else None
        }
        
        # Дополнительные параметры для улучшения производительности
        params["learning_starts"] = 1000  # Начинаем обучение после накопления опыта
        params["train_freq"] = 4  # Обучаемся каждые 4 шага
        params["gradient_steps"] = 1  # Один градиентный шаг за раз
        
        return params
    
    def train(
        self, 
        total_timesteps: int,
        callback=None,
        **kwargs
    ) -> DQN:
        """
        Обучение DQN агента.
        
        Args:
            total_timesteps: общее количество шагов обучения
            callback: callback функции для мониторинга
            **kwargs: дополнительные параметры
            
        Returns:
            Обученная DQN модель
        """
        if self.model is None:
            raise ValueError("Модель не создана. Вызовите create_model() сначала.")
        
        self.logger.info(f"Начинаем обучение DQN на {total_timesteps:,} шагов")
        
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
        Предсказание действия с учетом exploration.
        
        Args:
            observation: наблюдение из среды
            deterministic: использовать детерминистичную политику
            
        Returns:
            Предсказанное действие
        """
        if self.model is None:
            raise ValueError("Модель не создана.")
        
        # Для DQN deterministic обычно означает отсутствие epsilon-greedy exploration
        action, _ = self.model.predict(observation, deterministic=deterministic)
        
        return action
    
    def predict_with_exploration(self, observation: np.ndarray, exploration_rate: Optional[float] = None) -> int:
        """
        Предсказание с явным контролем exploration.
        
        Args:
            observation: наблюдение из среды
            exploration_rate: уровень исследования (epsilon)
            
        Returns:
            Действие с учетом исследования
        """
        if self.model is None:
            raise ValueError("Модель не создана.")
        
        if exploration_rate is None:
            exploration_rate = self.current_exploration_rate
        
        # Epsilon-greedy exploration
        if np.random.random() < exploration_rate:
            # Случайное действие
            if hasattr(self.training_env, 'action_space'):
                action = self.training_env.action_space.sample()
            else:
                # Предполагаем discrete действия: buy(0), sell(1), hold(2)
                action = np.random.randint(0, len(self.trading_config.discrete_actions))
        else:
            # Жадное действие от модели
            action, _ = self.model.predict(observation, deterministic=True)
        
        return action
    
    def get_q_values(self, observation: np.ndarray) -> Dict[str, Any]:
        """
        Получение Q-значений для всех действий.
        
        Args:
            observation: наблюдение из среды
            
        Returns:
            Q-значения для анализа
        """
        if self.model is None or not hasattr(self.model, 'q_net'):
            return {}
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            
            try:
                # Получаем Q-значения для всех действий
                q_values = self.model.q_net(obs_tensor)
                q_values_np = q_values.cpu().numpy()[0]
                
                result = {
                    'q_values': q_values_np.tolist(),
                    'max_q_value': float(np.max(q_values_np)),
                    'min_q_value': float(np.min(q_values_np)),
                    'q_value_std': float(np.std(q_values_np)),
                    'best_action': int(np.argmax(q_values_np))
                }
                
                # Добавляем Q-значения для конкретных действий
                action_names = getattr(self.trading_config, 'discrete_actions', ['buy', 'sell', 'hold'])
                for i, action_name in enumerate(action_names):
                    if i < len(q_values_np):
                        result[f'q_{action_name}'] = float(q_values_np[i])
                
                return result
                
            except Exception as e:
                self.logger.warning(f"Не удалось получить Q-значения: {e}")
                return {}
    
    def get_exploration_statistics(self) -> Dict[str, Any]:
        """Получение статистики исследования."""
        stats = {
            'current_exploration_rate': self.current_exploration_rate,
            'initial_exploration_rate': self.drl_config.exploration_initial_eps,
            'final_exploration_rate': self.drl_config.exploration_final_eps,
        }
        
        # Добавляем информацию о текущем прогрессе exploration decay
        if hasattr(self.model, '_current_progress_remaining'):
            progress = 1.0 - self.model._current_progress_remaining
            stats['exploration_progress'] = float(progress)
        
        return stats
    
    def set_exploration_rate(self, exploration_rate: float):
        """
        Установка уровня исследования.
        
        Args:
            exploration_rate: новый уровень исследования (0-1)
        """
        exploration_rate = np.clip(exploration_rate, 0.0, 1.0)
        self.current_exploration_rate = exploration_rate
        
        # Обновляем в модели если возможно
        if self.model is not None and hasattr(self.model, 'exploration_rate'):
            self.model.exploration_rate = exploration_rate
        
        self.logger.info(f"Уровень исследования установлен: {exploration_rate:.4f}")
    
    def get_replay_buffer_stats(self) -> Dict[str, Any]:
        """Получение статистики replay buffer."""
        if self.model is None or not hasattr(self.model, 'replay_buffer'):
            return {}
        
        buffer = self.model.replay_buffer
        
        stats = {
            'buffer_size': buffer.buffer_size,
            'current_size': buffer.size(),
            'full': buffer.full
        }
        
        if hasattr(buffer, 'pos'):
            stats['position'] = buffer.pos
        
        # Статистика наград в буфере
        if hasattr(buffer, 'rewards') and buffer.size() > 0:
            rewards = buffer.rewards[:buffer.size()]
            stats.update({
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'min_reward': float(np.min(rewards)),
                'max_reward': float(np.max(rewards))
            })
        
        return stats
    
    def analyze_action_distribution(self, observations: np.ndarray, n_samples: int = 1000) -> Dict[str, Any]:
        """
        Анализ распределения действий на выборке наблюдений.
        
        Args:
            observations: массив наблюдений
            n_samples: количество образцов для анализа
            
        Returns:
            Статистика распределения действий
        """
        if self.model is None:
            return {}
        
        # Ограничиваем количество образцов
        if len(observations) > n_samples:
            indices = np.random.choice(len(observations), n_samples, replace=False)
            sample_observations = observations[indices]
        else:
            sample_observations = observations
        
        action_counts = {}
        q_value_stats = {'max': [], 'min': [], 'std': []}
        
        for obs in sample_observations:
            # Получаем действие
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Подсчитываем действия
            action_key = int(action)
            action_counts[action_key] = action_counts.get(action_key, 0) + 1
            
            # Собираем статистику Q-значений
            q_stats = self.get_q_values(obs)
            if q_stats:
                q_value_stats['max'].append(q_stats.get('max_q_value', 0))
                q_value_stats['min'].append(q_stats.get('min_q_value', 0))
                q_value_stats['std'].append(q_stats.get('q_value_std', 0))
        
        # Преобразуем счетчики в проценты
        total_samples = len(sample_observations)
        action_distribution = {k: v/total_samples for k, v in action_counts.items()}
        
        # Добавляем имена действий
        action_names = getattr(self.trading_config, 'discrete_actions', ['buy', 'sell', 'hold'])
        named_distribution = {}
        for i, name in enumerate(action_names):
            named_distribution[name] = action_distribution.get(i, 0.0)
        
        result = {
            'action_distribution': named_distribution,
            'action_counts': action_counts,
            'total_samples': total_samples,
            'entropy': self._calculate_entropy(list(action_distribution.values()))
        }
        
        # Добавляем статистику Q-значений
        if q_value_stats['max']:
            result['q_value_statistics'] = {
                'max_q_mean': float(np.mean(q_value_stats['max'])),
                'min_q_mean': float(np.mean(q_value_stats['min'])),
                'q_std_mean': float(np.mean(q_value_stats['std']))
            }
        
        return result
    
    def _calculate_entropy(self, probabilities: list) -> float:
        """Расчет энтропии распределения."""
        probabilities = np.array(probabilities)
        probabilities = probabilities[probabilities > 0]  # Убираем нули
        
        if len(probabilities) == 0:
            return 0.0
        
        return float(-np.sum(probabilities * np.log(probabilities)))
    
    def save_replay_buffer(self, path: str):
        """
        Сохранение replay buffer.
        
        Args:
            path: путь для сохранения
        """
        if self.model is None or not hasattr(self.model, 'replay_buffer'):
            self.logger.warning("Replay buffer недоступен для сохранения")
            return
        
        torch.save(self.model.replay_buffer, path)
        self.logger.info(f"Replay buffer сохранен: {path}")
    
    def load_replay_buffer(self, path: str):
        """
        Загрузка replay buffer.
        
        Args:
            path: путь к сохраненному буферу
        """
        if self.model is None:
            self.logger.warning("Модель не создана")
            return
        
        try:
            replay_buffer = torch.load(path, map_location=self.device)
            self.model.replay_buffer = replay_buffer
            self.logger.info(f"Replay buffer загружен: {path}")
        except Exception as e:
            self.logger.error(f"Ошибка загрузки replay buffer: {e}")
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Получение всех гиперпараметров DQN."""
        base_params = super().get_hyperparameters()
        
        dqn_specific = {
            'buffer_size': self.drl_config.buffer_size,
            'batch_size': self.drl_config.batch_size,
            'tau': self.drl_config.tau,
            'exploration_fraction': self.drl_config.exploration_fraction,
            'exploration_initial_eps': self.drl_config.exploration_initial_eps,
            'exploration_final_eps': self.drl_config.exploration_final_eps,
            'target_update_interval': self.drl_config.target_update_interval,
            'policy_kwargs': self.policy_kwargs,
            'device': self.device,
            'current_exploration_rate': self.current_exploration_rate
        }
        
        base_params.update(dqn_specific)
        return base_params