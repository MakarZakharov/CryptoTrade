"""
Простой скрипт для просмотра Observation Space.
Просто запустите: python show_observation.py
"""

import sys
from pathlib import Path

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent))

from env import CryptoTradingEnv

def main():
    print("=" * 70)
    print("OBSERVATION SPACE - БЫСТРЫЙ ПРОСМОТР")
    print("=" * 70)

    # Создаем окружение
    print("\n1. Создание окружения...")
    env = CryptoTradingEnv(
        symbol="BTCUSDT",
        timeframe="1d",
        initial_balance=10000.0,
        observation_window=50
    )
    print("   ✓ Готово!")

    # Показываем Observation Space
    print("\n2. Observation Space:")
    print(f"   Размер: {env.observation_space.shape}")
    print(f"   Тип: {type(env.observation_space).__name__}")

    # Получаем первое наблюдение
    print("\n3. Получение первого наблюдения (reset)...")
    obs, info = env.reset()
    print(f"   ✓ Размер observation: {obs.shape}")
    print(f"   ✓ Диапазон значений: [{obs.min():.4f}, {obs.max():.4f}]")

    # Показываем структуру
    feature_count = len(env.data_loader.get_feature_names())
    window_size = env.observation_window
    total_features = window_size * feature_count

    print("\n4. Структура Observation:")
    print(f"   ┌─ Historical Window: {total_features} значений")
    print(f"   │  ({window_size} свечей × {feature_count} фичей)")
    print(f"   ├─ Portfolio State: 3 значения")
    print(f"   │  (баланс, крипта, общая стоимость)")
    print(f"   ├─ Position Info: 2 значения")
    print(f"   │  (доля позиции, unrealized PnL)")
    print(f"   └─ Episode Info: 2 значения")
    print(f"      (прогресс, флаг позиции)")
    print(f"   ИТОГО: {obs.shape[0]} значений")

    # Показываем значения компонентов
    window_end = total_features
    portfolio_start = window_end
    portfolio_end = portfolio_start + 3
    position_start = portfolio_end
    position_end = position_start + 2
    episode_start = position_end

    print("\n5. Значения компонентов (первое наблюдение):")
    print(f"   Portfolio State:")
    print(f"     - Balance ratio: {obs[portfolio_start]:.6f}")
    print(f"     - Crypto value ratio: {obs[portfolio_start+1]:.6f}")
    print(f"     - Total value ratio: {obs[portfolio_start+2]:.6f}")
    
    print(f"   Position Info:")
    print(f"     - Position ratio: {obs[position_start]:.6f}")
    print(f"     - Unrealized PnL: {obs[position_start+1]:.6f}")
    
    print(f"   Episode Info:")
    print(f"     - Step ratio: {obs[episode_start]:.6f}")
    print(f"     - In position: {obs[episode_start+1]:.1f}")

    # Показываем info
    print("\n6. Info (дополнительная информация):")
    for key, value in info.items():
        if isinstance(value, float):
            print(f"   - {key}: {value:.4f}")
        else:
            print(f"   - {key}: {value}")

    # Делаем несколько шагов
    print("\n7. Изменение Observation после действий:")
    print("   Выполняем: HOLD -> BUY -> HOLD -> SELL")
    
    actions = [0, 1, 0, 2]
    action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
    
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_value = info['portfolio_value']
        
        print(f"\n   Step {i+1}: {action_names[action]}")
        print(f"     Portfolio: ${portfolio_value:.2f}")
        print(f"     Reward: {reward:.6f}")
        print(f"     Balance: ${info['balance']:.2f}")
        print(f"     Crypto: {info['crypto_held']:.6f}")

    print("\n" + "=" * 70)
    print("ГОТОВО! Observation Space работает корректно.")
    print("=" * 70)
    
    env.close()

if __name__ == "__main__":
    main()

