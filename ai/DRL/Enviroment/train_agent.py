"""
Скрипт для обучения DRL агента на криптовалютных данных.
Сохраняет модель и логи для последующего тестирования.
"""

import os
from datetime import datetime
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np

from env import CryptoTradingEnv, ActionSpace, RewardType


def create_directories():
    """Создать директории для сохранения."""
    dirs = ['models', 'logs', 'tensorboard']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def train_agent(
    symbol="BTCUSDT",
    timeframe="1d",
    algorithm="PPO",
    total_timesteps=100000,
    initial_balance=10000.0,
    reward_type=RewardType.RISK_ADJUSTED,
    learning_rate=3e-4,
    save_freq=10000
):
    """
    Обучить DRL агента.

    Args:
        symbol: Торговая пара
        timeframe: Таймфрейм
        algorithm: Алгоритм (PPO или A2C)
        total_timesteps: Количество шагов обучения
        initial_balance: Начальный баланс
        reward_type: Тип функции награды
        learning_rate: Learning rate
        save_freq: Частота сохранения чекпоинтов
    """
    print("\n" + "=" * 70)
    print("🚀 ОБУЧЕНИЕ DRL АГЕНТА")
    print("=" * 70)

    # Создаем директории
    create_directories()

    # Загружаем данные и делим на train/val
    print(f"\n📊 Загрузка данных: {symbol} {timeframe}")

    from data_loader import DataLoader
    loader = DataLoader(symbol=symbol, timeframe=timeframe)
    loader.load()

    total_length = len(loader)
    train_size = int(total_length * 0.8)

    print(f"  Всего данных: {total_length} свечей")
    print(f"  Train: {train_size} свечей (80%)")
    print(f"  Val: {total_length - train_size} свечей (20%)")

    # Train окружение
    print(f"\n🏋️ Создание train окружения...")
    train_env = CryptoTradingEnv(
        symbol=symbol,
        timeframe=timeframe,
        start_index=0,
        end_index=train_size,
        initial_balance=initial_balance,
        action_type=ActionSpace.DISCRETE,
        reward_type=reward_type,
        observation_window=50,
        add_indicators=True
    )

    train_env = Monitor(train_env, filename=f"logs/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    print(f"  Observation space: {train_env.observation_space.shape}")
    print(f"  Action space: {train_env.action_space}")
    print(f"  Reward type: {reward_type.value}")

    # Validation окружение
    print(f"\n📊 Создание validation окружения...")
    val_env = CryptoTradingEnv(
        symbol=symbol,
        timeframe=timeframe,
        start_index=train_size,
        initial_balance=initial_balance,
        action_type=ActionSpace.DISCRETE,
        reward_type=reward_type
    )

    val_env = Monitor(val_env, filename=f"logs/val_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # Создаем модель
    print(f"\n🤖 Создание модели: {algorithm}")

    if algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./tensorboard/"
        )
    elif algorithm == "A2C":
        model = A2C(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            verbose=1,
            tensorboard_log="./tensorboard/"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Callbacks
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path='./models/checkpoints/',
        name_prefix=f'{algorithm.lower()}_crypto'
    )

    # Обучение
    print(f"\n🎓 Начинаем обучение на {total_timesteps} шагов...")
    print("=" * 70)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )

        # Сохранение финальной модели
        model_name = f"models/{algorithm.lower()}_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        model.save(model_name)

        print("\n" + "=" * 70)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 70)
        print(f"📦 Модель сохранена: {model_name}")
        print(f"📊 Лучшая модель: models/best_model.zip")
        print(f"📈 TensorBoard логи: tensorboard/")

        # Тестирование на validation
        print("\n" + "=" * 70)
        print("📊 ТЕСТИРОВАНИЕ НА VALIDATION")
        print("=" * 70)

        obs, _ = val_env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = val_env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        metrics = val_env.get_metrics()

        print(f"\nРезультаты на Validation:")
        print(f"  Шагов: {steps}")
        print(f"  Общая награда: {total_reward:.2f}")
        print(f"  Total Return: {metrics.total_return_pct:.2f}%")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
        print(f"  Total Trades: {metrics.total_trades}")
        print(f"  Win Rate: {metrics.win_rate:.2f}%")

        # Сохраняем метрики
        import json
        metrics_dict = {
            'symbol': symbol,
            'timeframe': timeframe,
            'algorithm': algorithm,
            'total_timesteps': total_timesteps,
            'val_return': metrics.total_return_pct,
            'val_sharpe': metrics.sharpe_ratio,
            'val_max_dd': metrics.max_drawdown_pct,
            'val_trades': metrics.total_trades,
            'val_win_rate': metrics.win_rate,
            'model_path': model_name
        }

        with open(f"models/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print("\n🎉 Готово! Используйте test_agent.py для детального тестирования")
        print("   или manual_trading.py для ручной торговли\n")

        return model, metrics

    except KeyboardInterrupt:
        print("\n\n⚠️ Обучение прервано пользователем")
        print("Сохраняем промежуточную модель...")
        model.save(f"models/{algorithm.lower()}_interrupted.zip")
        print("✅ Модель сохранена: models/{algorithm.lower()}_interrupted.zip")

    finally:
        train_env.close()
        val_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Обучение DRL агента для крипто-трейдинга')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Торговая пара')
    parser.add_argument('--timeframe', type=str, default='1d', help='Таймфрейм')
    parser.add_argument('--algorithm', type=str, default='PPO', choices=['PPO', 'A2C'], help='Алгоритм')
    parser.add_argument('--timesteps', type=int, default=100000, help='Количество шагов обучения')
    parser.add_argument('--balance', type=float, default=10000.0, help='Начальный баланс')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')

    args = parser.parse_args()

    print("\n🎯 Параметры обучения:")
    print(f"  Symbol: {args.symbol}")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  Initial Balance: ${args.balance}")
    print(f"  Learning Rate: {args.lr}")

    train_agent(
        symbol=args.symbol,
        timeframe=args.timeframe,
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        initial_balance=args.balance,
        learning_rate=args.lr
    )
