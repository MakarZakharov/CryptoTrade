#!/usr/bin/env python3
"""
Базовий клас для всіх STAS_ML агентів.
Визначає загальний інтерфейс для PPO, DQN та інших агентів.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import os
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm


class BaseAgent(ABC):
    """
    Абстрактний базовий клас для всіх STAS_ML агентів торгівлі.
    
    Визначає загальний інтерфейс, який мають реалізувати всі агенти:
    - PPO агент
    - DQN агент
    - Майбутні агенти (A2C, SAC, тощо)
    """
    
    def __init__(self, config):
        """
        Ініціалізація базового агента.
        
        Args:
            config: Конфігурація торгівлі з параметрами агента
        """
        self.config = config
        self.model: Optional[BaseAlgorithm] = None
        self.training_env = None
        self.eval_env = None
        
        # Параметри збереження
        self.save_dir = getattr(config, 'save_dir', 'models')
        self.model_name = getattr(config, 'model_name', 'base_agent')
        
        # Статистика навчання
        self.training_stats = {
            'total_timesteps': 0,
            'episodes': 0,
            'best_reward': float('-inf'),
            'best_model_path': None
        }
    
    @abstractmethod
    def create_model(self, env, **kwargs) -> BaseAlgorithm:
        """
        Створити модель агента.
        
        Args:
            env: Торгове середовище
            **kwargs: Додаткові параметри для створення моделі
            
        Returns:
            BaseAlgorithm: Створена модель агента
        """
        pass
    
    @abstractmethod
    def train(self, total_timesteps: int, **kwargs) -> BaseAlgorithm:
        """
        Навчити агента.
        
        Args:
            total_timesteps: Загальна кількість кроків навчання
            **kwargs: Додаткові параметри навчання
            
        Returns:
            BaseAlgorithm: Навчена модель
        """
        pass
    
    @abstractmethod
    def act(self, observation: np.ndarray, deterministic: bool = True) -> Union[int, np.ndarray]:
        """
        Виконати дію на основі спостереження.
        
        Args:
            observation: Спостереження з середовища
            deterministic: Чи використовувати детерміністичну політику
            
        Returns:
            Union[int, np.ndarray]: Дія для виконання
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Зберегти модель агента.
        
        Args:
            path: Шлях для збереження моделі
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Завантажити модель агента.
        
        Args:
            path: Шлях до збереженої моделі
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Отримати інформацію про модель.
        
        Returns:
            Dict[str, Any]: Словник з інформацією про модель
        """
        if self.model is None:
            return {"status": "not_initialized"}
        
        info = {
            "status": "initialized",
            "algorithm": self.model.__class__.__name__,
            "total_timesteps": self.training_stats['total_timesteps'],
            "episodes": self.training_stats['episodes'],
            "best_reward": self.training_stats['best_reward'],
            "model_path": self.training_stats.get('best_model_path'),
            "config": {
                "save_dir": self.save_dir,
                "model_name": self.model_name
            }
        }
        
        return info
    
    def update_training_stats(self, timesteps: int, episodes: int = 0, reward: float = None):
        """
        Оновити статистику навчання.
        
        Args:
            timesteps: Кількість кроків
            episodes: Кількість епізодів
            reward: Поточна нагорода
        """
        self.training_stats['total_timesteps'] += timesteps
        self.training_stats['episodes'] += episodes
        
        if reward is not None and reward > self.training_stats['best_reward']:
            self.training_stats['best_reward'] = reward
            # Оновити шлях до найкращої моделі
            self.training_stats['best_model_path'] = os.path.join(
                self.save_dir, f"{self.model_name}_best"
            )
    
    def reset_training_stats(self):
        """Скинути статистику навчання."""
        self.training_stats = {
            'total_timesteps': 0,
            'episodes': 0,
            'best_reward': float('-inf'),
            'best_model_path': None
        }
    
    def set_environments(self, training_env, eval_env=None):
        """
        Встановити середовища для навчання та оцінки.
        
        Args:
            training_env: Середовище для навчання
            eval_env: Середовище для оцінки (опціонально)
        """
        self.training_env = training_env
        self.eval_env = eval_env
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Отримати гіперпараметри агента.
        
        Returns:
            Dict[str, Any]: Словник з гіперпараметрами
        """
        if self.model is None:
            return {}
        
        # Базові гіперпараметри, які є у всіх агентів
        base_params = {
            "learning_rate": getattr(self.model, 'learning_rate', None),
            "gamma": getattr(self.model, 'gamma', None),
            "verbose": getattr(self.model, 'verbose', None),
        }
        
        # Фільтруємо None значення
        return {k: v for k, v in base_params.items() if v is not None}
    
    def __str__(self) -> str:
        """Рядкове представлення агента."""
        if self.model is None:
            return f"{self.__class__.__name__}(not_initialized)"
        
        return f"{self.__class__.__name__}({self.model.__class__.__name__})"
    
    def __repr__(self) -> str:
        """Детальне представлення агента."""
        return self.__str__()