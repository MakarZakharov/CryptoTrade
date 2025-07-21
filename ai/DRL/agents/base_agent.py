"""Базовый класс для всех DRL агентов."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import os
import torch

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.policies import ActorCriticPolicy

from ..config import DRLConfig, TradingConfig
from ..utils import DRLLogger
from ..environments import TradingEnv


class BaseAgent(ABC):
    """
    Абстрактный базовый класс для всех DRL агентов.
    
    Определяет общий интерфейс для PPO, SAC, DQN и других агентов.
    Включает стандартизированные методы для обучения, сохранения,
    загрузки и оценки производительности.
    """
    
    def __init__(
        self, 
        drl_config: DRLConfig, 
        trading_config: TradingConfig,
        logger: Optional[DRLLogger] = None
    ):
        """
        Инициализация базового агента.
        
        Args:
            drl_config: конфигурация DRL параметров
            trading_config: конфигурация торговых параметров  
            logger: логгер для записи операций
        """
        self.drl_config = drl_config
        self.trading_config = trading_config
        self.logger = logger or DRLLogger(f"{self.__class__.__name__.lower()}")
        
        # Модель и среда
        self.model: Optional[BaseAlgorithm] = None
        self.training_env: Optional[Union[TradingEnv, VecEnv]] = None
        self.eval_env: Optional[TradingEnv] = None
        
        # Пути для сохранения
        self.models_dir = Path(drl_config.models_dir)
        self.logs_dir = Path(drl_config.logs_dir)
        self.experiment_name = self._generate_experiment_name()
        
        # Создаем директории
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Статистика обучения
        self.training_stats = {
            'total_timesteps': 0,
            'episodes': 0,
            'best_reward': float('-inf'),
            'best_model_path': None,
            'training_time': 0.0,
            'evaluation_rewards': [],
            'learning_curve': []
        }
        
        # Определение оптимального устройства
        self.device = self._determine_optimal_device()
        
        self.logger.info(f"Инициализирован агент {self.__class__.__name__}")
        self.logger.info(f"Эксперимент: {self.experiment_name}")
        self.logger.info(f"Устройство: {self.device}")
    
    def _generate_experiment_name(self) -> str:
        """Генерация имени эксперимента."""
        agent_name = self.__class__.__name__.replace('Agent', '').lower()
        return f"{agent_name}_{self.trading_config.symbol}_{self.trading_config.timeframe}_{self.drl_config.agent_type.lower()}"
    
    def _determine_optimal_device(self) -> str:
        """
        Определяет оптимальное устройство на основе типа политики и размера сети.
        
        Returns:
            Рекомендуемое устройство ('cpu' или 'cuda')
        """
        if self.drl_config.device == "auto":
            # Для MLP политик рекомендуем CPU, особенно для небольших сетей
            if self.trading_config.action_type in ["continuous", "discrete"]:
                # Проверяем размер сети
                total_params = sum(self.drl_config.net_arch)
                if total_params < 1000:  # Небольшие сети лучше работают на CPU
                    self.logger.info(f"Небольшая MLP сеть ({total_params} нейронов) - рекомендуется CPU")
                    return "cpu"
                elif total_params < 5000:  # Средние сети - зависит от доступности GPU
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        self.logger.info(f"Средняя MLP сеть ({total_params} нейронов) - используется GPU: {gpu_name} ({memory_gb:.1f}GB)")
                        return "cuda"
                    else:
                        self.logger.info(f"GPU недоступен, используется CPU для сети {total_params} нейронов")
                        return "cpu"
            
            # Для больших сетей или CNN - используем GPU если доступно
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.logger.info(f"Большая сеть - используется GPU: {gpu_name} ({memory_gb:.1f}GB)")
                return "cuda"
            else:
                self.logger.info("GPU недоступен, используется CPU")
                return "cpu"
        
        # Если устройство задано явно
        device = self.drl_config.device
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA недоступна, переключаемся на CPU")
            return "cpu"
        
        return device
    
    def _warn_about_device_selection(self):
        """Предупреждает пользователя об оптимальном выборе устройства."""
        if self.device == "cuda" and hasattr(self, 'model') and self.model is not None:
            if isinstance(getattr(self.model, 'policy', None), ActorCriticPolicy):
                net_size = sum(self.drl_config.net_arch)
                if net_size < 1000:
                    self.logger.warning(
                        f"Используется GPU для небольшой MLP сети ({net_size} нейронов). "
                        f"CPU может быть быстрее. Рассмотрите device='cpu' или device='auto'"
                    )
                elif net_size < 2000:
                    self.logger.info(
                        f"Используется GPU для средней MLP сети ({net_size} нейронов). "
                        f"Производительность может быть сопоставима с CPU."
                    )
    
    @abstractmethod
    def create_model(self, env: Union[TradingEnv, VecEnv], **kwargs) -> BaseAlgorithm:
        """
        Создание модели агента.
        
        Args:
            env: торговая среда
            **kwargs: дополнительные параметры
            
        Returns:
            Созданная модель
        """
        pass
    
    @abstractmethod
    def train(
        self, 
        total_timesteps: int,
        callback=None,
        **kwargs
    ) -> BaseAlgorithm:
        """
        Обучение агента.
        
        Args:
            total_timesteps: общее количество шагов
            callback: callback функции
            **kwargs: дополнительные параметры
            
        Returns:
            Обученная модель
        """
        pass
    
    def predict(
        self, 
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Union[int, np.ndarray]:
        """
        Предсказание действия на основе наблюдения.
        
        Args:
            observation: наблюдение из среды
            deterministic: использовать детерминистичную политику
            
        Returns:
            Предсказанное действие
        """
        if self.model is None:
            raise ValueError("Модель не создана. Вызовите create_model() сначала.")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Сохранение модели.
        
        Args:
            path: путь для сохранения (опционально)
            
        Returns:
            Путь к сохраненному файлу
        """
        if self.model is None:
            raise ValueError("Модель не создана. Нечего сохранять.")
        
        if path is None:
            path = self.models_dir / f"{self.experiment_name}_final"
        
        # Убеждаемся что директория существует
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(str(path))
        self.logger.info(f"Модель сохранена: {path}")
        
        # Сохраняем также метаданные
        self._save_metadata(str(path) + "_metadata.json")
        
        return str(path)
    
    def load(self, path: str, env: Optional[Union[TradingEnv, VecEnv]] = None) -> BaseAlgorithm:
        """
        Загрузка модели.
        
        Args:
            path: путь к модели
            env: среда (если нужно переопределить)
            
        Returns:
            Загруженная модель
        """
        if not os.path.exists(path) and not os.path.exists(path + ".zip"):
            raise FileNotFoundError(f"Модель не найдена: {path}")
        
        # Определяем класс модели из конфигурации
        if self.drl_config.agent_type == "PPO":
            from stable_baselines3 import PPO
            model_class = PPO
        elif self.drl_config.agent_type == "SAC":
            from stable_baselines3 import SAC
            model_class = SAC
        elif self.drl_config.agent_type == "DQN":
            from stable_baselines3 import DQN
            model_class = DQN
        else:
            raise ValueError(f"Неподдерживаемый тип агента: {self.drl_config.agent_type}")
        
        self.model = model_class.load(path, env=env or self.training_env)
        self.logger.info(f"Модель загружена: {path}")
        
        # Загружаем метаданные если есть
        metadata_path = str(path).replace(".zip", "") + "_metadata.json"
        if os.path.exists(metadata_path):
            self._load_metadata(metadata_path)
        
        return self.model
    
    def evaluate(
        self, 
        env: Optional[TradingEnv] = None, 
        n_episodes: int = 5,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Оценка производительности агента.
        
        Args:
            env: среда для оценки
            n_episodes: количество эпизодов
            deterministic: детерминистичная политика
            
        Returns:
            Словарь с метриками производительности
        """
        if self.model is None:
            raise ValueError("Модель не создана. Нечего оценивать.")
        
        eval_env = env or self.eval_env
        if eval_env is None:
            raise ValueError("Среда для оценки не задана.")
        
        episode_rewards = []
        episode_lengths = []
        portfolio_values = []
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Получаем финальную стоимость портфеля
            final_value = info.get('portfolio', {}).get('total_value', 0)
            portfolio_values.append(final_value)
            
            self.logger.debug(f"Эпизод {episode + 1}: награда={episode_reward:.4f}, длина={episode_length}")
        
        # Расчет метрик
        metrics = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_episode_length': float(np.mean(episode_lengths)),
            'mean_portfolio_value': float(np.mean(portfolio_values)),
            'total_return': float((np.mean(portfolio_values) - self.trading_config.initial_balance) / self.trading_config.initial_balance),
            'n_episodes': n_episodes
        }
        
        # Добавляем к статистике
        self.training_stats['evaluation_rewards'].extend(episode_rewards)
        
        self.logger.info(f"Оценка завершена: средняя награда={metrics['mean_reward']:.4f}, "
                        f"доходность={metrics['total_return']*100:.2f}%")
        
        return metrics
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Получение гиперпараметров агента."""
        base_params = self.drl_config.get_agent_params()
        
        if self.model is not None:
            # Добавляем параметры из модели
            model_params = {}
            for attr in ['learning_rate', 'gamma', 'batch_size', 'buffer_size']:
                if hasattr(self.model, attr):
                    model_params[attr] = getattr(self.model, attr)
            
            base_params.update(model_params)
        
        return base_params
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Получение статистики обучения."""
        return self.training_stats.copy()
    
    def update_training_stats(
        self, 
        timesteps: int = 0, 
        episodes: int = 0, 
        reward: Optional[float] = None,
        training_time: float = 0.0
    ):
        """
        Обновление статистики обучения.
        
        Args:
            timesteps: количество шагов
            episodes: количество эпизодов
            reward: текущая награда
            training_time: время обучения
        """
        self.training_stats['total_timesteps'] += timesteps
        self.training_stats['episodes'] += episodes
        self.training_stats['training_time'] += training_time
        
        if reward is not None:
            self.training_stats['learning_curve'].append({
                'timestep': self.training_stats['total_timesteps'],
                'reward': reward
            })
            
            if reward > self.training_stats['best_reward']:
                self.training_stats['best_reward'] = reward
                # Сохраняем лучшую модель
                best_path = self.models_dir / f"{self.experiment_name}_best"
                if self.model is not None:
                    self.model.save(str(best_path))
                    self.training_stats['best_model_path'] = str(best_path)
    
    def _save_metadata(self, path: str):
        """Сохранение метаданных эксперимента."""
        import json
        from datetime import datetime
        
        metadata = {
            'experiment_name': self.experiment_name,
            'agent_type': self.drl_config.agent_type,
            'trading_config': {
                'symbol': self.trading_config.symbol,
                'timeframe': self.trading_config.timeframe,
                'initial_balance': self.trading_config.initial_balance,
                'reward_scheme': self.trading_config.reward_scheme
            },
            'drl_config': self.get_hyperparameters(),
            'training_stats': self.training_stats,
            'created_at': datetime.now().isoformat(),
            'model_class': self.__class__.__name__
        }
        
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _load_metadata(self, path: str):
        """Загрузка метаданных эксперимента."""
        import json
        
        try:
            with open(path, 'r') as f:
                metadata = json.load(f)
            
            # Восстанавливаем статистику
            if 'training_stats' in metadata:
                self.training_stats.update(metadata['training_stats'])
            
            self.logger.info(f"Метаданные загружены из {path}")
        except Exception as e:
            self.logger.warning(f"Не удалось загрузить метаданные: {e}")
    
    def reset_training_stats(self):
        """Сброс статистики обучения."""
        self.training_stats = {
            'total_timesteps': 0,
            'episodes': 0,
            'best_reward': float('-inf'),
            'best_model_path': None,
            'training_time': 0.0,
            'evaluation_rewards': [],
            'learning_curve': []
        }
        self.logger.info("Статистика обучения сброшена")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели."""
        if self.model is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "algorithm": self.model.__class__.__name__,
            "experiment_name": self.experiment_name,
            "hyperparameters": self.get_hyperparameters(),
            "training_stats": self.get_training_stats(),
            "device": getattr(self.model, 'device', 'unknown'),
            "policy": str(self.model.policy) if hasattr(self.model, 'policy') else 'unknown'
        }
    
    def set_environments(self, training_env: Union[TradingEnv, VecEnv], eval_env: Optional[TradingEnv] = None):
        """
        Установка сред для обучения и оценки.
        
        Args:
            training_env: среда для обучения
            eval_env: среда для оценки
        """
        self.training_env = training_env
        self.eval_env = eval_env
        self.logger.info("Среды установлены")
    
    def __str__(self) -> str:
        """Строковое представление агента."""
        model_name = self.model.__class__.__name__ if self.model else "не создана"
        return f"{self.__class__.__name__}(модель={model_name}, эксперимент={self.experiment_name})"
    
    def __repr__(self) -> str:
        """Детальное представление агента."""
        return self.__str__()