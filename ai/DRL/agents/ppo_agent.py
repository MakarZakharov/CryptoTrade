"""
PPO (Proximal Policy Optimization) агент для торговли криптовалютой.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .base_agent import BaseAgent


class TradingCallback(BaseCallback):
    """Callback для мониторинга обучения торгового агента."""
    
    def __init__(self, eval_freq: int = 10000, n_eval_episodes: int = 5, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Оценка производительности
            episode_rewards = []
            for _ in range(self.n_eval_episodes):
                obs = self.training_env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.training_env.step(action)
                    episode_reward += reward
                episode_rewards.append(episode_reward)
            
            mean_reward = np.mean(episode_rewards)
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"Новый лучший результат: {mean_reward:.2f}")
            
            if self.verbose > 0:
                print(f"Шаг {self.n_calls}: Средняя награда = {mean_reward:.2f}")
        
        return True


class PPOAgent(BaseAgent):
    """PPO агент для торговли криптовалютой."""
    
    def __init__(self, env, config: Dict[str, Any] = None):
        super().__init__(env, config)
        self.vec_env = None
        
    def create_model(self, **kwargs):
        """Создание PPO модели."""
        # Оборачиваем среду в векторизованную среду
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        # Параметры модели
        model_params = {
            'policy': 'MlpPolicy',
            'env': self.vec_env,
            'learning_rate': self.config.get('learning_rate', 3e-4),
            'n_steps': self.config.get('n_steps', 2048),
            'batch_size': self.config.get('batch_size', 64),
            'n_epochs': self.config.get('n_epochs', 10),
            'gamma': self.config.get('gamma', 0.99),
            'gae_lambda': self.config.get('gae_lambda', 0.95),
            'clip_range': self.config.get('clip_range', 0.2),
            'ent_coef': self.config.get('ent_coef', 0.01),
            'vf_coef': self.config.get('vf_coef', 0.5),
            'max_grad_norm': self.config.get('max_grad_norm', 0.5),
            'verbose': self.config.get('verbose', 1),
            'tensorboard_log': self.config.get('tensorboard_log', None),
            **kwargs
        }
        
        # Создание модели
        self.model = PPO(**model_params)
        
        self.logger.info("PPO модель создана успешно")
        self.logger.info(f"Параметры: {model_params}")
        
        return self.model
    
    def train(self, total_timesteps: int, **kwargs):
        """Обучение PPO агента."""
        if self.model is None:
            self.create_model()
        
        start_time = time.time()
        
        # Callback для мониторинга
        eval_freq = kwargs.get('eval_freq', 10000)
        callback = TradingCallback(eval_freq=eval_freq)
        
        self.logger.info(f"Начало обучения PPO на {total_timesteps} шагов")
        
        try:
            # Фильтруем неподдерживаемые параметры
            supported_kwargs = {k: v for k, v in kwargs.items() 
                              if k not in ['eval_freq', 'n_eval_episodes', 'eval_log_path']}
            
            # Обучение модели
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                **supported_kwargs
            )
            
            training_time = time.time() - start_time
            
            # Обновление статистики
            self.update_training_stats(
                total_timesteps=total_timesteps,
                training_time=training_time,
                best_reward=callback.best_mean_reward
            )
            
            self.logger.info(f"Обучение завершено за {training_time:.2f} секунд")
            self.logger.info(f"Лучшая средняя награда: {callback.best_mean_reward:.2f}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении: {e}")
            raise
        
        return self.model
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Предсказание действия PPO агентом."""
        if self.model is None:
            raise ValueError("Модель не создана. Сначала создайте или загрузите модель.")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Сохранение PPO модели."""
        if self.model is None:
            raise ValueError("Модель не создана. Нечего сохранять.")
        
        # Создание директории если не существует
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Сохранение модели
        self.model.save(path)
        
        # Сохранение статистики обучения
        stats_path = path + "_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        self.logger.info(f"Модель сохранена: {path}")
    
    def load(self, path: str):
        """Загрузка PPO модели."""
        try:
            # Загрузка модели
            self.model = PPO.load(path, env=self.vec_env)
            
            # Загрузка статистики если существует
            stats_path = path + "_stats.json"
            if os.path.exists(stats_path):
                import json
                with open(stats_path, 'r') as f:
                    self.training_stats = json.load(f)
            
            self.logger.info(f"Модель загружена: {path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            raise
    
    def fine_tune(self, additional_timesteps: int, **kwargs):
        """Дообучение модели на дополнительных данных."""
        if self.model is None:
            raise ValueError("Модель не создана. Сначала создайте или загрузите модель.")
        
        self.logger.info(f"Дообучение на {additional_timesteps} дополнительных шагов")
        
        # Дообучение
        start_time = time.time()
        self.model.learn(total_timesteps=additional_timesteps, **kwargs)
        training_time = time.time() - start_time
        
        # Обновление статистики
        self.training_stats['total_timesteps'] += additional_timesteps
        self.training_stats['training_time'] += training_time
        
        self.logger.info(f"Дообучение завершено за {training_time:.2f} секунд")
    
    def get_policy_info(self) -> Dict[str, Any]:
        """Получение информации о политике агента."""
        if self.model is None:
            return {}
        
        try:
            # Получение параметров сети
            policy_net = self.model.policy
            
            info = {
                'policy_class': str(type(policy_net)),
                'action_space': str(self.env.action_space),
                'observation_space': str(self.env.observation_space),
                'learning_rate': self.model.learning_rate,
                'gamma': self.model.gamma,
                'n_steps': self.model.n_steps,
                'batch_size': self.model.batch_size
            }
            
            # Добавляем информацию о слоях если возможно
            if hasattr(policy_net, 'mlp_extractor'):
                info['network_architecture'] = str(policy_net.mlp_extractor)
            
            return info
            
        except Exception as e:
            self.logger.error(f"Ошибка получения информации о политике: {e}")
            return {}


def create_ppo_agent(env, config: Dict[str, Any] = None) -> PPOAgent:
    """Удобная функция для создания PPO агента."""
    from .base_agent import get_default_config
    
    if config is None:
        config = get_default_config('PPO')
    
    return PPOAgent(env, config)


def main():
    """Пример использования PPO агента."""
    import sys
    import os
    
    # Добавляем путь к модулям
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
    from environment.trading_env import create_trading_environment, TradingConfig
    import pandas as pd
    import numpy as np
    
    # Создание тестовых данных
    np.random.seed(42)
    test_data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 102,
        'low': np.random.randn(1000).cumsum() + 98,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    })
    
    # Обеспечение логичности OHLC данных
    test_data['high'] = test_data[['open', 'high', 'close']].max(axis=1)
    test_data['low'] = test_data[['open', 'low', 'close']].min(axis=1)
    
    # Создание среды
    config = TradingConfig(initial_balance=10000.0, lookback_window=20)
    env = create_trading_environment(test_data, config)
    
    # Создание PPO агента
    agent = create_ppo_agent(env)
    
    print("Создание модели...")
    agent.create_model()
    
    print("Обучение агента...")
    agent.train(total_timesteps=10000)
    
    print("Оценка производительности...")
    eval_results = agent.evaluate(n_episodes=5)
    print(f"Средняя награда: {eval_results['mean_reward']:.2f}")
    
    print("Сохранение модели...")
    agent.save("models/ppo_trading_agent")
    
    print("Тест завершен!")


if __name__ == "__main__":
    main()