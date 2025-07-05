"""
Основной скрипт для обучения STAS_ML-агента.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
import torch

# Добавляем путь к модулям проекта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from CryptoTrade.ai.STAS_ML.config.trading_config import TradingConfig
from CryptoTrade.ai.STAS_ML.environment.trading_env import TradingEnv
from CryptoTrade.ai.STAS_ML.agents.dqn_agent import DQNAgent
from CryptoTrade.ai.STAS_ML.agents.ppo_agent import PPOAgent
from CryptoTrade.ai.STAS_ML.training.callbacks import TradingCallback, TensorboardCallback


class DRLTrainer:
    """Класс для обучения STAS_ML агентов."""
    
    def __init__(self, config: TradingConfig, save_dir: str = "models", resume_training: bool = True):
        self.config = config
        self.save_dir = save_dir
        self.resume_training = resume_training
        # Используем фиксированное имя без timestamp для постоянного обучения одной модели
        self.experiment_name = f"{config.symbol}_{config.timeframe}_{config.reward_scheme}"
        
        # Создаем директории
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"logs/{self.experiment_name}", exist_ok=True)
        
    def prepare_environment(self, train_split: float = 0.8, validation_split: float = 0.1):
        """Подготовить среды для обучения и валидации."""
        # Создаем базовую среду
        full_env = TradingEnv(self.config)
        
        # Разделяем данные на train/validation/test
        data_len = len(full_env.data)
        train_end = int(data_len * train_split)
        val_end = int(data_len * (train_split + validation_split))
        
        # Создаем конфигурации для разных периодов
        train_config = TradingConfig(**self.config.__dict__)
        val_config = TradingConfig(**self.config.__dict__)
        test_config = TradingConfig(**self.config.__dict__)
        
        # Создаем среды
        self.train_env = TradingEnv(train_config)
        self.train_env.data = self.train_env.data.iloc[:train_end]
        self.train_env = Monitor(self.train_env, f"logs/{self.experiment_name}/train")
        
        self.val_env = TradingEnv(val_config)
        self.val_env.data = self.val_env.data.iloc[train_end:val_end]
        self.val_env = Monitor(self.val_env, f"logs/{self.experiment_name}/val")
        
        self.test_env = TradingEnv(test_config)
        self.test_env.data = self.test_env.data.iloc[val_end:]
        
        print(f"Подготовлены среды:")
        print(f"  Train: {len(self.train_env.unwrapped.data)} записей")
        print(f"  Validation: {len(self.val_env.unwrapped.data)} записей")
        print(f"  Test: {len(self.test_env.data)} записей")
        
        return self.train_env, self.val_env, self.test_env
    
    def create_agent(self, agent_type: str = "PPO", model_config: Optional[Dict] = None):
        """Создать агента."""
        if agent_type.upper() == "DQN":
            self.agent = DQNAgent(self.config)
        elif agent_type.upper() == "PPO":
            self.agent = PPOAgent(self.config)
        else:
            raise ValueError(f"Неподдерживаемый тип агента: {agent_type}")
        
        # Проверяем, есть ли существующая модель для продолжения обучения
        model_dir = f"{self.save_dir}/{self.experiment_name}"
        possible_model_paths = [
            f"{model_dir}/final_model.zip",
            f"{model_dir}/best_model.zip",
            f"{model_dir}/final_model",
            f"{model_dir}/best_model"
        ]
        
        # Проверяем совместимость observation space перед загрузкой модели
        test_env = TradingEnv(self.config)
        current_obs_shape = test_env.observation_space.shape
        
        # Ищем существующую модель
        existing_model_path = None
        if self.resume_training:
            # Сначала проверяем точное совпадение с новым именованием
            for model_path in possible_model_paths:
                if os.path.exists(model_path):
                    existing_model_path = model_path
                    break
            
            # Если не найдено, ищем папки со старым именованием (с timestamp)
            if not existing_model_path and os.path.exists(self.save_dir):
                # Ищем папки вида SYMBOL_TIMEFRAME_YYYYMMDD_HHMMSS
                prefix = f"{self.config.symbol}_{self.config.timeframe}_"
                matching_dirs = []
                
                for item in os.listdir(self.save_dir):
                    item_path = os.path.join(self.save_dir, item)
                    if os.path.isdir(item_path) and item.startswith(prefix):
                        matching_dirs.append(item)
                
                if matching_dirs:
                    # Берем последнюю по времени создания (алфавитный порядок работает для timestamp)
                    latest_dir = sorted(matching_dirs)[-1]
                    old_model_dir = f"{self.save_dir}/{latest_dir}"
                    
                    # Проверяем модели в старой папке
                    old_possible_paths = [
                        f"{old_model_dir}/final_model.zip",
                        f"{old_model_dir}/best_model.zip",
                        f"{old_model_dir}/final_model",
                        f"{old_model_dir}/best_model"
                    ]
                    
                    for model_path in old_possible_paths:
                        if os.path.exists(model_path):
                            existing_model_path = model_path
                            break
                    
                    # Также проверяем checkpoints в старой папке
                    if not existing_model_path:
                        old_checkpoint_dir = f"{old_model_dir}/checkpoints"
                        if os.path.exists(old_checkpoint_dir):
                            checkpoint_files = [f for f in os.listdir(old_checkpoint_dir) if f.endswith('.zip')]
                            if checkpoint_files:
                                checkpoint_files.sort(key=lambda x: int(x.split('_')[-2]) if '_' in x and x.split('_')[-2].isdigit() else 0)
                                existing_model_path = os.path.join(old_checkpoint_dir, checkpoint_files[-1])
            
            # Проверяем папку checkpoints в новой папке
            if not existing_model_path:
                checkpoint_dir = f"{model_dir}/checkpoints"
                if os.path.exists(checkpoint_dir):
                    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
                    if checkpoint_files:
                        checkpoint_files.sort(key=lambda x: int(x.split('_')[-2]) if '_' in x and x.split('_')[-2].isdigit() else 0)
                        existing_model_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
        
        if existing_model_path:
            print(f"🔄 Найдена существующая модель: {existing_model_path}")
            
            # Проверяем совместимость observation space
            try:
                print(f"📊 Проверяем совместимость observation space...")
                print(f"   Текущий: {current_obs_shape}")
                
                # Попытка загрузить модель для проверки
                temp_agent = type(self.agent)(self.config)
                temp_agent.create_model(test_env)
                temp_agent.load(existing_model_path, test_env)
                
                print(f"✅ Observation space совместим, продолжаем обучение...")
                self.agent.load(existing_model_path, self.train_env)
                print(f"✅ Модель {agent_type} загружена для продолжения обучения")
                
            except Exception as e:
                if "Observation spaces do not match" in str(e):
                    print(f"⚠️ Observation space не совместим с существующей моделью")
                    print(f"   Ошибка: {e}")
                    print(f"🆕 Создаем новую модель...")
                    self.agent.create_model(self.train_env, model_config)
                    print(f"✅ Создана новая модель {agent_type} из-за несовместимости")
                else:
                    print(f"❌ Ошибка загрузки модели: {e}")
                    print(f"🆕 Создаем новую модель...")
                    self.agent.create_model(self.train_env, model_config)
                    print(f"✅ Создана новая модель {agent_type}")
        else:
            # Создаем новую модель
            self.agent.create_model(self.train_env, model_config)
            if self.resume_training:
                print(f"🆕 Существующая модель не найдена, создаем новую модель {agent_type}")
                print(f"💡 Искали в: {model_dir}")
            else:
                print(f"🆕 Создана новая модель {agent_type} с конфигурацией: {model_config or 'default'}")
        
        return self.agent
    
    def create_callbacks(self, eval_freq: int = 5000, save_freq: int = 10000):
        """Створити покращені callbacks для навчання."""
        callbacks = []
        
        # Покращений callback для оцінки моделі з частішою перевіркою
        eval_callback = EvalCallback(
            self.val_env,
            best_model_save_path=f"{self.save_dir}/{self.experiment_name}",
            log_path=f"logs/{self.experiment_name}",
            eval_freq=eval_freq,  # Частіше оцінювання
            n_eval_episodes=10,  # Більше епізодів для надійної оцінки
            deterministic=True,
            render=False,
            verbose=1,
            warn=False
        )
        callbacks.append(eval_callback)
        
        # Частіші checkpoint-и
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=f"{self.save_dir}/{self.experiment_name}/checkpoints",
            name_prefix="model",
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Ранній зупин для запобігання перенавчанню
        from .callbacks import EarlyStoppingCallback
        early_stopping_callback = EarlyStoppingCallback(
            patience=30,  # Зупинка після 30 оцінок без покращення
            min_improvement=0.005,  # Мінімальне покращення 0.5%
            verbose=1
        )
        callbacks.append(early_stopping_callback)
        
        # Кастомні callbacks
        trading_callback = TradingCallback(
            log_dir=f"logs/{self.experiment_name}",
            experiment_name=self.experiment_name
        )
        callbacks.append(trading_callback)
        
        # Tensorboard callback
        tensorboard_callback = TensorboardCallback(
            log_dir=f"logs/{self.experiment_name}/tensorboard"
        )
        callbacks.append(tensorboard_callback)
        
        # Моніторинг продуктивності
        from .callbacks import PerformanceMonitorCallback
        performance_callback = PerformanceMonitorCallback(
            log_freq=5000,
            verbose=1
        )
        callbacks.append(performance_callback)
        
        return CallbackList(callbacks)
    
    def train(self, total_timesteps: int = 500000, eval_freq: int = 5000, 
              save_freq: int = 10000, agent_type: str = "PPO", 
              model_config: Optional[Dict] = None):
        """Обучить агента."""
        
        print(f"🚀 Начинаем обучение {agent_type} агента")
        print(f"Эксперимент: {self.experiment_name}")
        print(f"Общее количество шагов: {total_timesteps:,}")
        print(f"Символ: {self.config.symbol}, Таймфрейм: {self.config.timeframe}")
        print(f"Схема наград: {self.config.reward_scheme}")
        print("-" * 50)
        
        # Подготавливаем среды
        self.prepare_environment()
        
        # Создаем агента
        self.create_agent(agent_type, model_config)
        
        # Создаем callbacks
        callbacks = self.create_callbacks(eval_freq, save_freq)
        
        # Сохраняем конфигурацию
        self.save_config()
        
        try:
            # Обучаем модель
            self.agent.train(
                total_timesteps=total_timesteps,
                callback=callbacks
            )
            
            # Сохраняем финальную модель
            final_model_path = f"{self.save_dir}/{self.experiment_name}/final_model"
            self.agent.save(final_model_path)
            
            print(f"✅ Обучение завершено!")
            print(f"Модель сохранена: {final_model_path}")
            
            return self.agent
            
        except Exception as e:
            print(f"❌ Ошибка во время обучения: {e}")
            raise
    
    def save_config(self):
        """Сохранить конфигурацию эксперимента."""
        config_dict = {
            'symbol': self.config.symbol,
            'timeframe': self.config.timeframe,
            'exchange': self.config.exchange,
            'initial_balance': self.config.initial_balance,
            'commission_rate': self.config.commission_rate,
            'slippage_rate': self.config.slippage_rate,
            'spread_rate': self.config.spread_rate,
            'reward_scheme': self.config.reward_scheme,
            'lookback_window': self.config.lookback_window,
            'experiment_name': self.experiment_name,
            'created_at': datetime.now().isoformat()
        }
        
        config_df = pd.DataFrame([config_dict])
        config_df.to_csv(f"logs/{self.experiment_name}/config.csv", index=False)
        
        # Также сохраняем как JSON для удобства
        import json
        with open(f"logs/{self.experiment_name}/config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)


def quick_train(symbol: str = "BTCUSDT", timeframe: str = "1d", 
                agent_type: str = "PPO", timesteps: int = 100000,
                reward_scheme: str = "optimized"):
    """Быстрый запуск обучения с минимальными настройками."""
    
    config = TradingConfig(
        symbol=symbol,
        timeframe=timeframe,
        reward_scheme=reward_scheme,
        initial_balance=10000.0
    )
    
    trainer = DRLTrainer(config)
    return trainer.train(
        total_timesteps=timesteps,
        agent_type=agent_type
    )


if __name__ == "__main__":
    # Пример использования
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение STAS_ML агента для торговли')
    parser.add_argument('--symbol', default='BTCUSDT', help='Торговая пара')
    parser.add_argument('--timeframe', default='1d', help='Таймфрейм')
    parser.add_argument('--agent', default='PPO', choices=['PPO', 'DQN'], help='Тип агента')
    parser.add_argument('--timesteps', type=int, default=500000, help='Количество шагов обучения')
    parser.add_argument('--reward', default='optimized', help='Схема наград')
    
    args = parser.parse_args()
    
    quick_train(
        symbol=args.symbol,
        timeframe=args.timeframe,
        agent_type=args.agent,
        timesteps=args.timesteps,
        reward_scheme=args.reward
    ) 