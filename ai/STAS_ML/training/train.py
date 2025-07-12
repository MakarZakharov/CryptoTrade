"""
Основний скрипт для навчання STAS_ML-агента.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
import torch

# Додаємо шлях до модулів проекту
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
    """Клас для навчання STAS_ML агентів."""
    
    def __init__(self, config: TradingConfig, save_dir: str = "models", resume_training: bool = True, custom_model_name: str = None):
        self.config = config
        self.save_dir = save_dir
        self.resume_training = resume_training
        
        # Дозволяємо користувачу вказати власне ім'я моделі
        if custom_model_name:
            self.experiment_name = custom_model_name
        else:
            # Використовуємо фіксоване ім'я без timestamp для постійного навчання однієї моделі
            self.experiment_name = f"{config.symbol}_{config.timeframe}_{config.reward_scheme}"
        
        # Створюємо директорії
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f"logs/{self.experiment_name}", exist_ok=True)
        
    def prepare_environment(self, train_split: float = 0.8, validation_split: float = 0.1):
        """Підготувати середовища для навчання та валідації."""
        # Створюємо базове середовище
        full_env = TradingEnv(self.config)
        
        # Розділяємо дані на train/validation/test
        data_len = len(full_env.data)
        train_end = int(data_len * train_split)
        val_end = int(data_len * (train_split + validation_split))
        
        # Створюємо конфігурації для різних періодів
        train_config = TradingConfig(**self.config.__dict__)
        val_config = TradingConfig(**self.config.__dict__)
        test_config = TradingConfig(**self.config.__dict__)
        
        # Створюємо середовища
        self.train_env = TradingEnv(train_config)
        self.train_env.data = self.train_env.data.iloc[:train_end]
        self.train_env = Monitor(self.train_env, f"logs/{self.experiment_name}/train")
        
        self.val_env = TradingEnv(val_config)
        self.val_env.data = self.val_env.data.iloc[train_end:val_end]
        self.val_env = Monitor(self.val_env, f"logs/{self.experiment_name}/val")
        
        self.test_env = TradingEnv(test_config)
        self.test_env.data = self.test_env.data.iloc[val_end:]
        
        print(f"Підготовлені середовища:")
        print(f"  Train: {len(self.train_env.unwrapped.data)} записів")
        print(f"  Validation: {len(self.val_env.unwrapped.data)} записів")
        print(f"  Test: {len(self.test_env.data)} записів")
        
        return self.train_env, self.val_env, self.test_env
    
    def create_agent(self, agent_type: str = "PPO", model_config: Optional[Dict] = None):
        """Створити агента."""
        if agent_type.upper() == "DQN":
            self.agent = DQNAgent(self.config)
        elif agent_type.upper() == "PPO":
            self.agent = PPOAgent(self.config)
        else:
            raise ValueError(f"Непідтримуваний тип агента: {agent_type}")
        
        # Перевіряємо, чи є існуюча модель для продовження навчання
        model_dir = f"{self.save_dir}/{self.experiment_name}"
        possible_model_paths = [
            f"{model_dir}/final_model.zip",
            f"{model_dir}/best_model.zip",
            f"{model_dir}/final_model",
            f"{model_dir}/best_model"
        ]
        
        # Шукаємо існуючу модель
        existing_model_path = None
        if self.resume_training:
            for model_path in possible_model_paths:
                if os.path.exists(model_path):
                    existing_model_path = model_path
                    break
            
            # Також перевіряємо checkpoints
            if not existing_model_path:
                checkpoint_dir = f"{model_dir}/checkpoints"
                if os.path.exists(checkpoint_dir):
                    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
                    if checkpoint_files:
                        checkpoint_files.sort(key=lambda x: int(x.split('_')[-2]) if '_' in x and x.split('_')[-2].isdigit() else 0)
                        existing_model_path = os.path.join(checkpoint_dir, checkpoint_files[-1])
        
        if existing_model_path:
            print(f"🔄 Знайдена існуюча модель: {existing_model_path}")
            try:
                self.agent.load(existing_model_path, self.train_env)
                print(f"✅ Модель {agent_type} завантажена для продовження навчання")
            except Exception as e:
                print(f"⚠️ Помилка завантаження моделі: {e}")
                print(f"🆕 Створюємо нову модель...")
                self.agent.create_model(self.train_env, model_config)
                print(f"✅ Створена нова модель {agent_type}")
        else:
            # Створюємо нову модель
            self.agent.create_model(self.train_env, model_config)
            print(f"🆕 Створена нова модель {agent_type}")
        
        return self.agent
    
    def create_callbacks(self, eval_freq: int = 10000, save_freq: int = 20000):
        """Створити callbacks для навчання - ОПТИМІЗОВАНО для швидкості."""
        callbacks = []
        
        # Callback для оцінки моделі - рідше оцінка для швидкості
        eval_callback = EvalCallback(
            self.val_env,
            best_model_save_path=f"{self.save_dir}/{self.experiment_name}",
            log_path=f"logs/{self.experiment_name}",
            eval_freq=eval_freq,  # Збільшено інтервал оцінки
            n_eval_episodes=5,  # Зменшено кількість епізодів оцінки
            deterministic=True,
            render=False,
            verbose=1,
            warn=False
        )
        callbacks.append(eval_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=f"{self.save_dir}/{self.experiment_name}/checkpoints",
            name_prefix="model",
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Кастомні callbacks з покращеним структурованим виводом
        trading_callback = TradingCallback(
            log_dir=f"logs/{self.experiment_name}",
            experiment_name=self.experiment_name
        )
        # Налаштовуємо інтервал звітності (кожні 5000 кроків для швидкості)
        trading_callback.report_interval = 5000
        callbacks.append(trading_callback)
        
        # Tensorboard callback
        tensorboard_callback = TensorboardCallback(
            log_dir=f"logs/{self.experiment_name}/tensorboard"
        )
        callbacks.append(tensorboard_callback)
        
        return CallbackList(callbacks)
    
    def train(self, total_timesteps: int = 500000, eval_freq: int = 5000, 
              save_freq: int = 10000, agent_type: str = "PPO", 
              model_config: Optional[Dict] = None):
        """Навчити агента."""
        
        print(f"🚀 Починаємо навчання {agent_type} агента")
        print(f"Експеримент: {self.experiment_name}")
        print(f"Загальна кількість кроків: {total_timesteps:,}")
        print(f"Символ: {self.config.symbol}, Таймфрейм: {self.config.timeframe}")
        print(f"Схема винагород: {self.config.reward_scheme}")
        print("-" * 50)
        
        # Підготовуємо середовища
        self.prepare_environment()
        
        # Створюємо агента
        self.create_agent(agent_type, model_config)
        
        # Створюємо callbacks
        callbacks = self.create_callbacks(eval_freq, save_freq)
        
        # Зберігаємо конфігурацію
        self.save_config()
        
        try:
            # Навчаємо модель
            self.agent.train(
                total_timesteps=total_timesteps,
                callback=callbacks
            )
            
            # Зберігаємо фінальну модель
            final_model_path = f"{self.save_dir}/{self.experiment_name}/final_model"
            self.agent.save(final_model_path)
            
            print(f"✅ Навчання завершено!")
            print(f"Модель збережена: {final_model_path}")
            
            return self.agent
            
        except Exception as e:
            print(f"❌ Помилка під час навчання: {e}")
            raise
    
    def save_config(self):
        """Зберегти конфігурацію експерименту."""
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
        
        # Також зберігаємо як JSON для зручності
        import json
        with open(f"logs/{self.experiment_name}/config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)


def quick_train(symbol: str = "BTCUSDT", timeframe: str = "1d", 
                agent_type: str = "PPO", timesteps: int = 200000,
                reward_scheme: str = "optimized"):
    """Швидкий запуск навчання з мінімальними налаштуваннями."""
    
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
    # Приклад використання
    import argparse
    
    parser = argparse.ArgumentParser(description='Навчання STAS_ML агента для торгівлі')
    parser.add_argument('--symbol', default='BTCUSDT', help='Торгова пара')
    parser.add_argument('--timeframe', default='1d', help='Таймфрейм')
    parser.add_argument('--agent', default='PPO', choices=['PPO', 'DQN'], help='Тип агента')
    parser.add_argument('--timesteps', type=int, default=200000, help='Кількість кроків навчання')
    parser.add_argument('--reward', default='optimized', help='Схема винагород')
    
    args = parser.parse_args()
    
    quick_train(
        symbol=args.symbol,
        timeframe=args.timeframe,
        agent_type=args.agent,
        timesteps=args.timesteps,
        reward_scheme=args.reward
    )