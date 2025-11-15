"""
Скрипт для детального анализа Observation Space.
Показывает структуру, размеры, содержимое и как оно изменяется.
"""

import numpy as np
import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from env import CryptoTradingEnv, ActionSpace, RewardType
from data_loader import DataLoader


def analyze_observation_space():
    """
    Детальный анализ Observation Space.
    """
    print("\n" + "=" * 80)
    print("OBSERVATION SPACE ANALYSIS")
    print("=" * 80)

    # Создаем окружение
    print("\n1. Создание окружения...")
    env = CryptoTradingEnv(
        symbol="BTCUSDT",
        timeframe="1d",
        initial_balance=10000.0,
        observation_window=50,
        add_indicators=True,
        normalize_observations=True
    )

    print(f"   ✓ Окружение создано!")
    print(f"   - Symbol: {env.symbol}")
    print(f"   - Timeframe: {env.timeframe}")
    print(f"   - Initial Balance: ${env.initial_balance:.2f}")
    print(f"   - Observation Window: {env.observation_window}")
    print(f"   - Data loaded: {len(env.data_loader)} свечей")

    # Анализ Observation Space
    print("\n" + "-" * 80)
    print("2. ОБЩАЯ ИНФОРМАЦИЯ О OBSERVATION SPACE")
    print("-" * 80)

    obs_space = env.observation_space
    print(f"   Type: {type(obs_space).__name__}")
    print(f"   Shape: {obs_space.shape}")
    print(f"   Dtype: {obs_space.dtype}")
    print(f"   Low bound: {obs_space.low[0]:.4f} ...")
    print(f"   High bound: {obs_space.high[0]:.4f} ...")

    # Размер компонентов
    print("\n" + "-" * 80)
    print("3. СТРУКТУРА OBSERVATION")
    print("-" * 80)

    feature_count = len(env.data_loader.get_feature_names())
    window_features = env.observation_window * feature_count
    portfolio_state_size = 3
    position_state_size = 2
    episode_info_size = 2
    total_size = window_features + portfolio_state_size + position_state_size + episode_info_size

    print(f"\n   Компоненты Observation Space:")
    print(f"   ┌─────────────────────────────────────────────────────────────┐")
    print(f"   │ 1. Historical Window                                        │")
    print(f"   │    - Window size: {env.observation_window:3d} свечей         │")
    print(f"   │    - Features per candle: {feature_count:3d}                │")
    print(f"   │    - Total: {window_features:5d} значений                   │")
    print(f"   ├─────────────────────────────────────────────────────────────┤")
    print(f"   │ 2. Portfolio State                                          │")
    print(f"   │    - Normalized balance                                     │")
    print(f"   │    - Normalized crypto value                                │")
    print(f"   │    - Normalized total portfolio value                       │")
    print(f"   │    - Total: {portfolio_state_size:5d} значений              │")
    print(f"   ├─────────────────────────────────────────────────────────────┤")
    print(f"   │ 3. Position Info                                            │")
    print(f"   │    - Position ratio (доля позиции в портфеле)               │")
    print(f"   │    - Unrealized PnL % (нормализованный)                     │")
    print(f"   │    - Total: {position_state_size:5d} значений               │")
    print(f"   ├─────────────────────────────────────────────────────────────┤")
    print(f"   │ 4. Episode Info                                             │")
    print(f"   │    - Step ratio (прогресс эпизода 0-1)                      │")
    print(f"   │    - In position flag (0/1)                                 │")
    print(f"   │    - Total: {episode_info_size:5d} значений                 │")
    print(f"   ├─────────────────────────────────────────────────────────────┤")
    print(f"   │ ИТОГО: {total_size:5d} значений                             │")
    print(f"   └─────────────────────────────────────────────────────────────┘")

    # Названия фичей
    print("\n" + "-" * 80)
    print("4. ФИЧИ В OBSERVATION (из DataLoader)")
    print("-" * 80)

    feature_names = env.data_loader.get_feature_names()
    print(f"\n   Всего фичей: {len(feature_names)}")
    print(f"   Названия:")
    for i, name in enumerate(feature_names, 1):
        print(f"   {i:2d}. {name}")

    # Reset и первое наблюдение
    print("\n" + "-" * 80)
    print("5. ПЕРВОЕ НАБЛЮДЕНИЕ (после reset)")
    print("-" * 80)

    obs, info = env.reset()
    print(f"\n   Observation shape: {obs.shape}")
    print(f"   Observation dtype: {obs.dtype}")
    print(f"   Min value: {obs.min():.6f}")
    print(f"   Max value: {obs.max():.6f}")
    print(f"   Mean value: {obs.mean():.6f}")
    print(f"   Std value: {obs.std():.6f}")

    # Детальный разбор компонентов
    print("\n   Детальный разбор компонентов:")
    print(f"   ┌─────────────────────────────────────────────────────────────┐")

    # 1. Historical Window
    window_start = 0
    window_end = window_features
    window_data = obs[window_start:window_end]
    print(f"   │ 1. Historical Window [{window_start}:{window_end}]          │")
    print(f"   │    Shape: {window_data.shape}                               │")
    print(f"   │    Min: {window_data.min():.6f}, Max: {window_data.max():.6f} │")
    print(f"   │    Mean: {window_data.mean():.6f}, Std: {window_data.std():.6f}│")

    # 2. Portfolio State
    portfolio_start = window_end
    portfolio_end = portfolio_start + portfolio_state_size
    portfolio_data = obs[portfolio_start:portfolio_end]
    print(f"   ├─────────────────────────────────────────────────────────────┤")
    print(f"   │ 2. Portfolio State [{portfolio_start}:{portfolio_end}]      │")
    print(f"   │    Values: {portfolio_data}                                 │")
    print(f"   │    - Balance ratio: {portfolio_data[0]:.6f}                │")
    print(f"   │    - Crypto value ratio: {portfolio_data[1]:.6f}            │")
    print(f"   │    - Total value ratio: {portfolio_data[2]:.6f}             │")

    # 3. Position Info
    position_start = portfolio_end
    position_end = position_start + position_state_size
    position_data = obs[position_start:position_end]
    print(f"   ├─────────────────────────────────────────────────────────────┤")
    print(f"   │ 3. Position Info [{position_start}:{position_end}]          │")
    print(f"   │    Values: {position_data}                                  │")
    print(f"   │    - Position ratio: {position_data[0]:.6f}                 │")
    print(f"   │    - Unrealized PnL %: {position_data[1]:.6f}               │")

    # 4. Episode Info
    episode_start = position_end
    episode_end = episode_start + episode_info_size
    episode_data = obs[episode_start:episode_end]
    print(f"   ├─────────────────────────────────────────────────────────────┤")
    print(f"   │ 4. Episode Info [{episode_start}:{episode_end}]             │")
    print(f"   │    Values: {episode_data}                                   │")
    print(f"   │    - Step ratio: {episode_data[0]:.6f}                      │")
    print(f"   │    - In position: {episode_data[1]:.1f}                     │")
    print(f"   └─────────────────────────────────────────────────────────────┘")

    # Info
    print("\n   Info dict:")
    for key, value in info.items():
        if isinstance(value, float):
            print(f"   - {key}: {value:.4f}")
        else:
            print(f"   - {key}: {value}")

    # Тест изменения observation после действий
    print("\n" + "-" * 80)
    print("6. ИЗМЕНЕНИЕ OBSERVATION ПРИ ДЕЙСТВИЯХ")
    print("-" * 80)

    actions = [
        (0, "HOLD"),
        (1, "BUY"),
        (0, "HOLD"),
        (2, "SELL"),
        (0, "HOLD")
    ]

    print("\n   Выполняем последовательность действий:")
    for i, (action, action_name) in enumerate(actions):
        obs_new, reward, terminated, truncated, info_new = env.step(action)

        portfolio_data_new = obs_new[portfolio_start:portfolio_end]
        position_data_new = obs_new[position_start:position_end]
        episode_data_new = obs_new[episode_start:episode_end]

        print(f"\n   Step {i+1}: {action_name}")
        print(f"   ├─ Portfolio State: balance={portfolio_data_new[0]:.4f}, "
              f"crypto={portfolio_data_new[1]:.4f}, total={portfolio_data_new[2]:.4f}")
        print(f"   ├─ Position Info: ratio={position_data_new[0]:.4f}, "
              f"unrealized_pnl={position_data_new[1]:.4f}")
        print(f"   ├─ Episode Info: step_ratio={episode_data_new[0]:.4f}, "
              f"in_position={episode_data_new[1]:.1f}")
        print(f"   ├─ Reward: {reward:.6f}")
        print(f"   └─ Portfolio Value: ${info_new['portfolio_value']:.2f}")

        if terminated or truncated:
            print(f"   ⚠ Episode ended!")
            break

    # Визуализация наблюдения (первые несколько значений)
    print("\n" + "-" * 80)
    print("7. ВИЗУАЛИЗАЦИЯ OBSERVATION (первые 20 значений)")
    print("-" * 80)

    print("\n   Первые 20 значений observation:")
    for i in range(min(20, len(obs))):
        print(f"   [{i:4d}]: {obs[i]:10.6f}")

    # Статистика по всем наблюдениям
    print("\n" + "-" * 80)
    print("8. СТАТИСТИКА OBSERVATION (100 шагов)")
    print("-" * 80)

    obs, info = env.reset()
    obs_list = [obs]

    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        obs_list.append(obs)

        if terminated or truncated:
            break

    obs_array = np.array(obs_list)
    print(f"\n   Собрано наблюдений: {len(obs_list)}")
    print(f"   Shape: {obs_array.shape}")
    print(f"   Min across all obs: {obs_array.min():.6f}")
    print(f"   Max across all obs: {obs_array.max():.6f}")
    print(f"   Mean across all obs: {obs_array.mean():.6f}")
    print(f"   Std across all obs: {obs_array.std():.6f}")

    # Анализ изменчивости компонентов
    print("\n   Изменчивость компонентов:")
    portfolio_data_all = obs_array[:, portfolio_start:portfolio_end]
    position_data_all = obs_array[:, position_start:position_end]
    episode_data_all = obs_array[:, episode_start:episode_end]

    print(f"   - Portfolio State std: {portfolio_data_all.std(axis=0)}")
    print(f"   - Position Info std: {position_data_all.std(axis=0)}")
    print(f"   - Episode Info std: {episode_data_all.std(axis=0)}")

    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 80)

    env.close()


def quick_test():
    """
    Быстрый тест Observation Space.
    """
    print("\n" + "=" * 80)
    print("QUICK OBSERVATION SPACE TEST")
    print("=" * 80)

    env = CryptoTradingEnv(
        symbol="BTCUSDT",
        timeframe="1d",
        initial_balance=10000.0
    )

    print(f"\nObservation Space:")
    print(f"  Shape: {env.observation_space.shape}")
    print(f"  Type: {env.observation_space}")

    obs, info = env.reset()
    print(f"\nFirst Observation:")
    print(f"  Shape: {obs.shape}")
    print(f"  Range: [{obs.min():.4f}, {obs.max():.4f}]")

    # Простой цикл
    print(f"\nExecuting 5 steps...")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: Action={action}, Reward={reward:.4f}, "
              f"Portfolio=${info['portfolio_value']:.2f}")

    env.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        analyze_observation_space()

